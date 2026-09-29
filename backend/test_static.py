from deck_generator import build_deck
from solver_runner import run_job
from f06_parser import parse_f06

spec = dict(
    length_mm=100, width_mm=20, thickness_mm=2,
    nx=6, ny=3, material="steel", bc="cantilever",
    analysis="static", load_n=50.0, load_dir="z",
)
deck, meta = build_deck(spec)
f06 = run_job(deck, job_prefix="stest")
with open("debug_static.f06", "w") as fh:
    fh.write(f06)
result = parse_f06(f06)
print("ok:", result["ok"], "error:", result["error"], "analysis:", result["analysis"])
if result["ok"] and result["analysis"] == "static":
    vecs = result["static"]["vectors"]
    print("n_vectors:", len(vecs))
    tip_ids = [gid for gid, (x, y, z) in meta["nodes"].items() if abs(x - meta["L"]) < 1e-6]
    for gid in tip_ids:
        print("tip node", gid, "T3(z) deflection mm =", vecs.get(gid, ["?"]*6)[2])
