from deck_generator import build_deck
from solver_runner import run_job
from f06_parser import parse_f06

spec = dict(
    length_mm=100.0, width_mm=20.0, thickness_mm=2.0, nx=6, ny=3,
    material="steel", bc="cantilever", analysis="nonlinear",
    load_n=800.0, load_dir="y",
    yield_stress_mpa=250.0, tangent_modulus_mpa=2000.0, max_strain=0.05,
    num_load_steps=4,
)
deck, meta = build_deck(spec)
print(deck)
print("---meta---")
print({k: v for k, v in meta.items() if k not in ("nodes", "elements")})

f06 = run_job(deck, job_prefix="nlgen")
with open("debug_nl_generalized.f06", "w") as fh:
    fh.write(f06)
print("FATAL" in f06.upper())
for ln in f06.splitlines():
    if "FATAL" in ln.upper():
        print(ln)

result = parse_f06(f06)
print("ok:", result["ok"], "analysis:", result["analysis"])
if result["nonlinear"]:
    for step in result["nonlinear"]["steps"]:
        gid = meta["monitor_node"]
        v = step["vectors"].get(gid)
        print("step", step["step"], "epsilon", step["epsilon"], "monitor node", gid, "disp", v)
