import math
import os
import traceback
import uuid

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, Literal, List

from deck_generator import build_deck, build_deck_from_tetmesh, MATERIAL_PRESETS
from f06_parser import parse_f06
from solver_runner import run_job, SolveError
from tetmesh import mesh_stl_bytes, MeshError
import assistant

MESH_STORE = {}  # mesh_id -> {nodes, tets, boundary_faces, bbox} (in-memory, single-user local tool)

NASTRAN_ROOT = os.environ.get("NASTRAN_ROOT", "/n")

app = FastAPI(title="NASTRAN-95 Design Study API")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)


def von_mises(sx, sy, txy):
    return math.sqrt(max(0.0, sx * sx - sx * sy + sy * sy + 3 * txy * txy))


class AnalysisRequest(BaseModel):
    length_mm: float = Field(100, gt=0)
    width_mm: float = Field(20, gt=0)
    thickness_mm: float = Field(2, gt=0)
    nx: int = Field(6, ge=1, le=30)
    ny: int = Field(3, ge=1, le=12)
    material: str = Field("steel")
    material_custom: Optional[dict] = None  # {E, nu, rho} overrides `material` if set
    bc: Literal["cantilever", "simply_supported", "fixed_fixed"] = "cantilever"
    analysis: Literal["modes", "static", "freq_response"] = "modes"
    num_modes: int = Field(8, ge=1, le=30)
    freq_max_hz: float = Field(2.0e5, gt=0)
    load_n: float = Field(100.0)
    load_dir: Literal["x", "y", "z"] = "z"
    freq_start_hz: float = Field(50.0, ge=0)
    freq_end_hz: float = Field(300.0, gt=0)
    num_freq_points: int = Field(41, ge=2, le=200)
    damping_g: float = Field(0.02, ge=0, le=1)


@app.get("/materials")
def materials():
    return MATERIAL_PRESETS


@app.post("/analyze")
def analyze(req: AnalysisRequest):
    spec = req.dict()
    if req.material_custom:
        spec["material"] = req.material_custom
    try:
        deck_text, meta = build_deck(spec)
    except Exception as e:
        raise HTTPException(400, "deck generation failed: %s" % e)

    try:
        f06_text = run_job(deck_text, job_prefix="ds")
    except SolveError as e:
        raise HTTPException(500, "solver failed: %s" % e)
    except Exception:
        raise HTTPException(500, "unexpected solver error: %s" % traceback.format_exc())

    result = parse_f06(f06_text)
    if not result["ok"]:
        raise HTTPException(422, "solver reported an error: %s" % result["error"])

    nodes = {str(gid): list(coord) for gid, coord in meta["nodes"].items()}
    elements = [list(e) for e in meta["elements"]]

    response = dict(
        mesh=dict(nodes=nodes, elements=elements, fixed_ids=meta["fixed_ids"], nx=meta["nx"], ny=meta["ny"]),
        geometry=dict(length_mm=meta["L"], width_mm=meta["W"], thickness_mm=meta["t"], bc=meta["bc"]),
        analysis=result["analysis"],
    )
    if result["analysis"] == "modes":
        sorted_modes = sorted(result["modes"], key=lambda m: m["freq_hz"])[: req.num_modes]
        response["modes"] = [
            dict(mode=i + 1, freq_hz=m["freq_hz"],
                 vectors={str(k): v for k, v in m["vectors"].items()})
            for i, m in enumerate(sorted_modes)
        ]
    elif result["analysis"] == "static":
        response["static"] = dict(
            vectors={str(k): v for k, v in result["static"]["vectors"].items()}
        )
        # quick summary: max out-of-plane deflection
        vecs = result["static"]["vectors"]
        if vecs:
            max_def = max(abs(v[2]) for v in vecs.values())
            response["static"]["max_deflection_mm"] = max_def

        element_stress = {}
        max_vm = 0.0
        for eid, fibers in result["static"]["element_stress"].items():
            entry = {}
            vms = []
            for fiber_name in ("bottom", "top"):
                s = fibers.get(fiber_name)
                if not s:
                    continue
                vm = von_mises(s["normal_x"], s["normal_y"], s["shear_xy"])
                entry[fiber_name] = dict(**s, von_mises=vm)
                vms.append(vm)
            entry["von_mises_max"] = max(vms) if vms else 0.0
            max_vm = max(max_vm, entry["von_mises_max"])
            element_stress[str(eid)] = entry
        response["static"]["element_stress"] = element_stress
        response["static"]["max_von_mises_mpa"] = max_vm
    else:  # freq_response
        comp_idx = {"x": 0, "y": 1, "z": 2}[req.load_dir]
        node_id = meta["monitor_node"]
        pts = result["freq_response"]["points_by_node"].get(node_id, [])
        pts = sorted(pts, key=lambda p: p["freq_hz"])
        response["freq_response"] = dict(
            monitor_node=node_id,
            component=req.load_dir,
            points=[
                dict(freq_hz=p["freq_hz"], magnitude_mm=p["mag"][comp_idx], phase_deg=p["phase"][comp_idx])
                for p in pts
            ],
        )
        if pts:
            peak = max(pts, key=lambda p: p["mag"][comp_idx])
            response["freq_response"]["peak_freq_hz"] = peak["freq_hz"]
            response["freq_response"]["peak_magnitude_mm"] = peak["mag"][comp_idx]

    return response


class MeshAnalysisRequest(BaseModel):
    mesh_id: str
    fixed_ids: List[int] = Field(..., min_items=1)
    loaded_ids: List[int] = Field(..., min_items=1)
    material: str = Field("steel")
    material_custom: Optional[dict] = None
    analysis: Literal["modes", "static", "freq_response"] = "modes"
    num_modes: int = Field(6, ge=1, le=30)
    freq_max_hz: float = Field(2.0e5, gt=0)
    load_n: float = Field(500.0)
    load_dir: Literal["x", "y", "z"] = "z"
    freq_start_hz: float = Field(50.0, ge=0)
    freq_end_hz: float = Field(300.0, gt=0)
    num_freq_points: int = Field(41, ge=2, le=200)
    damping_g: float = Field(0.02, ge=0, le=1)


@app.post("/upload_geometry")
async def upload_geometry(file: UploadFile = File(...), mesh_size_max: Optional[float] = None):
    if not file.filename.lower().endswith(".stl"):
        raise HTTPException(400, "only .stl files are supported right now")
    raw = await file.read()
    if len(raw) > 25 * 1024 * 1024:
        raise HTTPException(400, "file too large (25MB limit)")
    try:
        mesh = mesh_stl_bytes(raw, mesh_size_max=mesh_size_max)
    except MeshError as e:
        raise HTTPException(422, str(e))
    except Exception:
        raise HTTPException(500, "meshing failed: %s" % traceback.format_exc())

    mesh_id = uuid.uuid4().hex[:12]
    MESH_STORE[mesh_id] = mesh
    return dict(
        mesh_id=mesh_id,
        n_nodes=len(mesh["nodes"]),
        n_tets=len(mesh["tets"]),
        nodes={str(k): list(v) for k, v in mesh["nodes"].items()},
        boundary_faces=[list(f) for f in mesh["boundary_faces"]],
        bbox=mesh["bbox"],
    )


@app.post("/analyze_mesh")
def analyze_mesh(req: MeshAnalysisRequest):
    mesh = MESH_STORE.get(req.mesh_id)
    if not mesh:
        raise HTTPException(404, "mesh_id not found -- upload the geometry again")

    spec = req.dict()
    if req.material_custom:
        spec["material"] = req.material_custom
    spec["nodes"] = mesh["nodes"]
    spec["tets"] = mesh["tets"]

    try:
        deck_text, meta = build_deck_from_tetmesh(spec)
    except Exception as e:
        raise HTTPException(400, "deck generation failed: %s" % e)

    try:
        f06_text = run_job(deck_text, job_prefix="tet")
    except SolveError as e:
        raise HTTPException(500, "solver failed: %s" % e)
    except Exception:
        raise HTTPException(500, "unexpected solver error: %s" % traceback.format_exc())

    result = parse_f06(f06_text)
    if not result["ok"]:
        raise HTTPException(422, "solver reported an error: %s" % result["error"])

    response = dict(
        mesh=dict(
            nodes={str(k): list(v) for k, v in mesh["nodes"].items()},
            boundary_faces=[list(f) for f in mesh["boundary_faces"]],
            fixed_ids=meta["fixed_ids"],
        ),
        analysis=result["analysis"],
    )
    if result["analysis"] == "modes":
        sorted_modes = sorted(result["modes"], key=lambda m: m["freq_hz"])[: req.num_modes]
        response["modes"] = [
            dict(mode=i + 1, freq_hz=m["freq_hz"],
                 vectors={str(k): v for k, v in m["vectors"].items()})
            for i, m in enumerate(sorted_modes)
        ]
    elif result["analysis"] == "static":
        vecs = result["static"]["vectors"]
        max_def = 0.0
        for v in vecs.values():
            mag = math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)
            max_def = max(max_def, mag)
        response["static"] = dict(
            vectors={str(k): v for k, v in vecs.items()},
            max_deflection_mm=max_def,
        )
    else:  # freq_response
        comp_idx = {"x": 0, "y": 1, "z": 2}[req.load_dir]
        node_id = meta["monitor_node"]
        pts = sorted(result["freq_response"]["points_by_node"].get(node_id, []),
                     key=lambda p: p["freq_hz"])
        response["freq_response"] = dict(
            monitor_node=node_id, component=req.load_dir,
            points=[dict(freq_hz=p["freq_hz"], magnitude_mm=p["mag"][comp_idx],
                         phase_deg=p["phase"][comp_idx]) for p in pts],
        )
        if pts:
            peak = max(pts, key=lambda p: p["mag"][comp_idx])
            response["freq_response"]["peak_freq_hz"] = peak["freq_hz"]
            response["freq_response"]["peak_magnitude_mm"] = peak["mag"][comp_idx]

    return response


class AssistantParseRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=2000)


class AssistantExplainRequest(BaseModel):
    error_text: str = Field(..., min_length=1, max_length=4000)
    deck_excerpt: Optional[str] = None


@app.post("/assistant/parse")
def assistant_parse(req: AssistantParseRequest):
    if not assistant.ollama_available():
        raise HTTPException(503, "Ollama isn't reachable at localhost:11434 -- is it running?")
    try:
        raw = assistant.parse_intent(req.text)
        # round-trip through AnalysisRequest so missing/invalid fields fall
        # back to sane defaults instead of breaking the form
        validated = AnalysisRequest(**raw)
    except Exception as e:
        raise HTTPException(422, "couldn't turn that into an analysis: %s" % e)
    return validated.dict()


@app.post("/assistant/explain")
def assistant_explain(req: AssistantExplainRequest):
    if not assistant.ollama_available():
        raise HTTPException(503, "Ollama isn't reachable at localhost:11434 -- is it running?")
    try:
        explanation = assistant.explain_error(req.error_text, req.deck_excerpt or "")
    except Exception as e:
        raise HTTPException(500, "assistant failed: %s" % e)
    return dict(explanation=explanation)


@app.get("/health")
def health():
    return dict(
        ok=True, nastran_root=NASTRAN_ROOT,
        nastran_bin_exists=os.path.exists(os.path.join(NASTRAN_ROOT, "bin", "nastran.x")),
        ollama_ok=assistant.ollama_available(),
    )
