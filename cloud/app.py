import math
import os
import time
import traceback
import uuid

import numpy as np
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, Literal, List

from deck_generator import build_deck, build_deck_from_tetmesh, MATERIAL_PRESETS
from f06_parser import parse_f06
from solver_runner import run_job, SolveError
from tetmesh import mesh_stl_bytes, MeshError
import assistant

from hpc_solver.mesh import box_mesh, face_node_ids, boundary_faces as hpc_boundary_faces
from hpc_solver.serial import solve_static as hpc_solve_static
from hpc_solver.assembly import element_stress_strain as hpc_element_stress_strain
from hpc_solver.modal import modal_analysis as hpc_modal_analysis
from hpc_solver.freq_response import frequency_response as hpc_frequency_response
from hpc_solver.fatigue import FATIGUE_PRESETS, life_from_stress as hpc_life_from_stress

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
    analysis: Literal["modes", "static", "freq_response", "nonlinear"] = "modes"
    num_modes: int = Field(8, ge=1, le=30)
    freq_max_hz: float = Field(2.0e5, gt=0)
    load_n: float = Field(100.0)
    load_dir: Literal["x", "y", "z"] = "z"
    freq_start_hz: float = Field(50.0, ge=0)
    freq_end_hz: float = Field(300.0, gt=0)
    num_freq_points: int = Field(41, ge=2, le=200)
    damping_g: float = Field(0.02, ge=0, le=1)
    yield_stress_mpa: float = Field(250.0, gt=0)
    tangent_modulus_mpa: float = Field(2000.0, gt=0)
    max_strain: float = Field(0.05, gt=0)
    num_load_steps: int = Field(4, ge=1, le=20)


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
    elif result["analysis"] == "nonlinear":
        monitor_node = meta["monitor_node"]
        comp_idx = {"x": 0, "y": 1}[meta["load_dir"]]
        curve = []
        for step in result["nonlinear"]["steps"]:
            v = step["vectors"].get(monitor_node)
            curve.append(dict(
                step=step["step"],
                load_factor=step["step"] / meta["num_load_steps"],
                load_n=meta["load_n"] * step["step"] / meta["num_load_steps"],
                monitor_deflection_mm=v[comp_idx] if v else None,
                epsilon=step["epsilon"],
            ))
        response["nonlinear"] = dict(
            monitor_node=monitor_node,
            component=meta["load_dir"],
            yield_stress_mpa=meta["yield_stress_mpa"],
            tangent_modulus_mpa=meta["tangent_modulus_mpa"],
            max_strain=meta["max_strain"],
            load_deflection_curve=curve,
            steps=[
                dict(step=s["step"], epsilon=s["epsilon"],
                     vectors={str(k): v for k, v in s["vectors"].items()})
                for s in result["nonlinear"]["steps"]
            ],
        )
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


@app.post("/analyze_deck")
async def analyze_deck(file: UploadFile = File(...)):
    """
    Power-user path: run a raw NASTRAN-95 bulk-data deck as-is, bypassing
    the parametric/STL generators entirely -- whatever SOL/elements/case
    control the file specifies. Pure I/O: no new solver math, just
    solver_runner + f06_parser reused directly. f06_parser was built
    against this project's own generated decks (CQUAD4/CTETRA/CTRMEM,
    standard DISPLACEMENT VECTOR tables), so an arbitrary hand-written or
    third-party deck using unusual output requests may not parse cleanly
    even if NASTRAN itself solves it without error -- check "ok" in the
    response.
    """
    if not file.filename.lower().endswith((".bdf", ".dat", ".inp", ".nas", ".txt")):
        raise HTTPException(400, "expected a NASTRAN bulk data file (.bdf/.dat/.inp/.nas/.txt)")
    raw = await file.read()
    if len(raw) > 5 * 1024 * 1024:
        raise HTTPException(400, "file too large (5MB limit)")
    try:
        deck_text = raw.decode("utf-8", errors="replace")
    except Exception:
        raise HTTPException(400, "could not decode file as text")

    try:
        f06_text = run_job(deck_text, job_prefix="deck")
    except SolveError as e:
        raise HTTPException(500, "solver failed: %s" % e)
    except Exception:
        raise HTTPException(500, "unexpected solver error: %s" % traceback.format_exc())

    result = parse_f06(f06_text)
    response = dict(filename=file.filename, ok=result["ok"], analysis=result["analysis"], error=result["error"])
    if result["analysis"] == "modes":
        response["modes"] = [
            dict(mode=i + 1, freq_hz=m["freq_hz"], vectors={str(k): v for k, v in m["vectors"].items()})
            for i, m in enumerate(sorted(result["modes"], key=lambda m: m["freq_hz"]))
        ]
    elif result["analysis"] == "static":
        response["static"] = dict(vectors={str(k): v for k, v in result["static"]["vectors"].items()})
    elif result["analysis"] == "nonlinear":
        response["nonlinear"] = dict(steps=[
            dict(step=s["step"], epsilon=s["epsilon"], vectors={str(k): v for k, v in s["vectors"].items()})
            for s in result["nonlinear"]["steps"]
        ])
    elif result["analysis"] == "freq_response":
        response["freq_response"] = dict(points_by_node={
            str(k): v for k, v in result["freq_response"]["points_by_node"].items()
        })
    return response


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


# ==================== Large-model (sparse/iterative) solver ====================
# A from-scratch CST-tetrahedron + sparse-matrix + Jacobi-preconditioned CG
# solver (hpc_solver/), built specifically because NASTRAN-95's compiled
# binary has a hard, compile-time working-memory ceiling: its entire in-core
# array is COMMON/ZZZZZZ/IZ(14000000) in src/nastrn.f, a fixed size baked in
# at build time that no request/setting can exceed. This path sidesteps
# that binary entirely for bigger models -- limited only by this container's
# RAM, not a fixed array. It reuses the same CST-tet element math NASTRAN's
# own CTETRA uses; hpc_solver/validate_serial.py cross-checked our assembly
# against NASTRAN-95's CTETRA solve directly (agreement converges from
# ~2.5% to ~0.9% as the mesh is refined -- the expected pattern for two
# independent, correct implementations of the same physics on a coarse
# mesh) before this was wired in.
#
# hpc_solver/dcg.py + driver.py implement genuine MPI-based distributed-
# memory domain decomposition (RCB partitioning, distributed CG with
# Allreduce/Allgatherv) and were validated locally to give bit-identical
# results across 1/2/4/8 MPI ranks. This free-tier container runs it as a
# single process (no MPI runtime in this image, and a free instance has no
# cluster to distribute across anyway) -- the MPI path is the architecture
# ready for whenever this runs on an actual multi-node cluster, not
# something this deployment exercises.
#
# MAX_HPC_NODES is set from live bisection against this actual free-tier
# instance, not a theoretical estimate. Confirmed working: 2,744 nodes
# (3.4s), 3,375 nodes (3.8s). Confirmed hanging (60s+, no response, no
# clean error) at 3,840 and 4,096 nodes -- a cliff this sharp (not a
# graceful slowdown) points to this instance running out of RAM and
# thrashing rather than failing cleanly, right around 3,400-3,800 nodes.
# 3,000 keeps real margin below the highest confirmed-good point. Raise
# this only after confirming headroom directly against the deployed
# instance -- do not extrapolate from local testing, this ceiling is
# specific to this free-tier container's actual RAM.
MAX_HPC_NODES = int(os.environ.get("MAX_HPC_NODES", "3000"))


# Modal and frequency-response solves are meaningfully more expensive than
# the static CG path per node (a sparse LU factorization for the eigensolver's
# shift-invert, or one such factorization PER frequency point for FRF) --
# kept separately capped, more conservatively, until live-tested the same
# way MAX_HPC_NODES was bisected against this actual instance.
MAX_HPC_NODES_ADVANCED = int(os.environ.get("MAX_HPC_NODES_ADVANCED", "800"))


class LargeModelRequest(BaseModel):
    nx: int = Field(20, ge=1, le=400)
    ny: int = Field(6, ge=1, le=400)
    nz: int = Field(6, ge=1, le=400)
    length_mm: float = Field(200.0, gt=0)
    width_mm: float = Field(40.0, gt=0)
    height_mm: float = Field(40.0, gt=0)
    material: str = Field("steel")
    material_custom: Optional[dict] = None
    analysis: Literal["static", "modal", "freq_response"] = "static"
    load_n: float = Field(5000.0)
    load_dir: Literal["x", "y", "z"] = "z"
    tol: float = Field(1e-8, gt=0, le=1e-2)
    max_iter: int = Field(3000, ge=100, le=5000)
    num_modes: int = Field(6, ge=1, le=20)
    freq_start_hz: float = Field(50.0, ge=0)
    freq_end_hz: float = Field(500.0, gt=0)
    num_freq_points: int = Field(11, ge=2, le=31)
    damping_g: float = Field(0.02, ge=0, le=1)
    compute_fatigue: bool = Field(True)


@app.post("/analyze_large")
def analyze_large(req: LargeModelRequest):
    n_nodes = (req.nx + 1) * (req.ny + 1) * (req.nz + 1)
    cap = MAX_HPC_NODES if req.analysis == "static" else MAX_HPC_NODES_ADVANCED
    if n_nodes > cap:
        raise HTTPException(
            400,
            "%d nodes requested for %s analysis, this free-tier instance is capped at %d "
            "-- reduce nx/ny/nz." % (n_nodes, req.analysis, cap),
        )

    mat = req.material_custom
    if not mat:
        mat = MATERIAL_PRESETS.get(req.material)
        if not mat:
            raise HTTPException(400, "unknown material: %s" % req.material)
    E, nu, rho = float(mat["E"]), float(mat["nu"]), float(mat["rho"])

    t0 = time.time()
    nodes, tets = box_mesh(req.nx, req.ny, req.nz, req.length_mm, req.width_mm, req.height_mm)
    t_mesh = time.time() - t0

    fixed = face_node_ids(nodes, axis=0, value=0.0)
    loaded = face_node_ids(nodes, axis=0, value=req.length_mm)
    comp_idx = {"x": 0, "y": 1, "z": 2}[req.load_dir]

    b_faces, b_owner = hpc_boundary_faces(tets)
    mesh_info = dict(
        nodes=len(nodes), tets=len(tets), ndof=len(nodes) * 3,
        geometry=dict(length_mm=req.length_mm, width_mm=req.width_mm, height_mm=req.height_mm),
        node_xyz=nodes.tolist(),
        boundary_faces=b_faces.tolist(),
        boundary_face_owner_tet=b_owner.tolist(),
        fixed_ids=[int(i) for i in fixed],
    )

    if req.analysis == "modal":
        t0 = time.time()
        try:
            modes = hpc_modal_analysis(nodes, tets, E, nu, rho, fixed, num_modes=req.num_modes)
        except MemoryError:
            raise HTTPException(507, "ran out of memory solving this mesh -- reduce nx/ny/nz or num_modes")
        except Exception as e:
            raise HTTPException(500, "modal solve failed: %s" % e)
        t_solve = time.time() - t0
        return dict(
            analysis="large_modal", mesh=mesh_info, timing=dict(mesh_s=t_mesh, solve_s=t_solve),
            modes=[dict(mode=i + 1, freq_hz=m["freq_hz"], vectors=m["vectors"].tolist())
                   for i, m in enumerate(modes)],
        )

    if req.analysis == "freq_response":
        per_node = req.load_n / len(loaded)
        loads = {}
        for nid in loaded:
            vec = [0.0, 0.0, 0.0]
            vec[comp_idx] = per_node
            loads[int(nid)] = tuple(vec)
        monitor_id = int(loaded[len(loaded) // 2])

        t0 = time.time()
        try:
            points_by_node = hpc_frequency_response(
                nodes, tets, E, nu, rho, fixed, loads,
                req.freq_start_hz, req.freq_end_hz, req.num_freq_points, req.damping_g, [monitor_id],
            )
        except MemoryError:
            raise HTTPException(507, "ran out of memory solving this mesh -- reduce nx/ny/nz or num_freq_points")
        except Exception as e:
            raise HTTPException(500, "frequency response solve failed: %s" % e)
        t_solve = time.time() - t0

        pts = sorted(points_by_node[monitor_id], key=lambda p: p["freq_hz"])
        points = [dict(freq_hz=p["freq_hz"], magnitude_mm=p["mag"][comp_idx], phase_deg=p["phase"][comp_idx])
                  for p in pts]
        resp = dict(
            analysis="large_freq_response", mesh=mesh_info, timing=dict(mesh_s=t_mesh, solve_s=t_solve),
            freq_response=dict(monitor_node=monitor_id, component=req.load_dir, points=points),
        )
        if points:
            peak = max(points, key=lambda p: p["magnitude_mm"])
            resp["freq_response"]["peak_freq_hz"] = peak["freq_hz"]
            resp["freq_response"]["peak_magnitude_mm"] = peak["magnitude_mm"]
        return resp

    # static (default)
    per_node = req.load_n / len(loaded)
    loads = {}
    for nid in loaded:
        vec = [0.0, 0.0, 0.0]
        vec[comp_idx] = per_node
        loads[int(nid)] = tuple(vec)

    t0 = time.time()
    try:
        u, info = hpc_solve_static(nodes, tets, E, nu, fixed, loads, method="cg", tol=req.tol, maxiter=req.max_iter)
    except MemoryError:
        raise HTTPException(507, "ran out of memory solving this mesh -- reduce nx/ny/nz")
    t_solve = time.time() - t0

    tip_disp = u[loaded, comp_idx]
    max_disp = float(np.abs(u).max())
    fields = dict(node_disp=u.tolist())

    fatigue_summary = None
    if req.compute_fatigue:
        ss = hpc_element_stress_strain(nodes, tets, u, E, nu)
        fat = FATIGUE_PRESETS.get(req.material, FATIGUE_PRESETS["steel"])
        life = hpc_life_from_stress(ss["von_mises"], fat["sigma_f_prime"], fat["b"])
        fields["element_von_mises_mpa"] = ss["von_mises"].tolist()
        fields["element_max_principal_mpa"] = ss["max_principal"].tolist()
        fields["element_max_strain"] = np.abs(ss["strain"]).max(axis=1).tolist()
        fields["element_life_cycles"] = life.tolist()
        fatigue_summary = dict(
            min_life_cycles=float(life.min()), max_von_mises_mpa=float(ss["von_mises"].max()),
            material_sigma_f_prime_mpa=fat["sigma_f_prime"], material_b=fat["b"],
        )

    return dict(
        analysis="large_static", mesh=mesh_info, timing=dict(mesh_s=t_mesh, solve_s=t_solve),
        converged=(info == 0), cg_info=int(info),
        tip_deflection_mm=float(tip_disp.mean()), max_deflection_mm=max_disp,
        fields=fields, fatigue=fatigue_summary,
    )


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
