from deck_generator import build_deck
from solver_runner import run_job
from f06_parser import parse_f06

spec = dict(
    length_mm=100, width_mm=20, thickness_mm=2,
    nx=6, ny=3, material="steel", bc="cantilever",
    analysis="modes", num_modes=6, freq_max_hz=2e5,
)
deck, meta = build_deck(spec)
print("=== DECK ===")
print(deck)
print("=== META ===")
print("nodes:", len(meta["nodes"]), "elements:", len(meta["elements"]), "n_mass_dof:", meta["n_mass_dof"])

f06 = run_job(deck, job_prefix="test")
with open("debug_last.f06", "w") as fh:
    fh.write(f06)
result = parse_f06(f06)
print("=== RESULT ===")
print("ok:", result["ok"], "error:", result["error"], "analysis:", result["analysis"])
if result["ok"] and result["analysis"] == "modes":
    for m in result["modes"]:
        print("mode", m["mode"], "freq_hz", m["freq_hz"], "n_vectors", len(m["vectors"]))
