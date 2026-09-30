"""
Empirical scale ceiling test for this NASTRAN-95 build: escalating CQUAD4
shell mesh sizes under a single static subcase (the cheapest analysis type
-- one direct decomposition, no eigensolve iteration, no load-step loop),
timing deck generation and solve, until something fails or gets too slow
to be practical. Not a routine benchmark -- a one-off answer to "how big a
model can this solver actually handle."

Writes results to bench_results.txt as it goes (flushed per line) so
partial results survive if the process is killed by an external timeout.
"""
import time
import traceback

from deck_generator import build_deck
from solver_runner import run_job, SolveError

LOG = open("bench_results.txt", "w")


def log(msg):
    print(msg)
    LOG.write(msg + "\n")
    LOG.flush()


targets = [100, 500, 2000, 8000, 32000, 128000, 512000, 1000000]
overall_budget_s = 25 * 60  # stop escalating past this cumulative wall time
t_start = time.time()

for n_target in targets:
    if time.time() - t_start > overall_budget_s:
        log("STOPPED: overall time budget (%ds) exhausted" % overall_budget_s)
        break

    k = max(1, round(n_target ** 0.5) - 1)
    spec = dict(
        length_mm=100.0, width_mm=100.0, thickness_mm=2.0,
        nx=k, ny=k, material="steel", bc="cantilever", analysis="static",
        load_n=1000.0, load_dir="z",
    )

    t0 = time.time()
    try:
        deck, meta = build_deck(spec)
    except Exception as e:
        log("target~%d: deck generation failed: %s" % (n_target, e))
        continue
    # MESSAGE 3019 (MAXIMUM LINE COUNT EXCEEDED) is an artificial output cap
    # (default 20,000 lines), not a solver capacity limit -- a big mesh's
    # bulk-data echo + ELSTRESS=ALL table blows past it trivially. Suppress
    # the echo and raise the cap so we're testing the real solver ceiling.
    deck = deck.replace("CEND\n", "CEND\n", 1)
    deck = deck.replace("BEGIN BULK\n", "MAXLINES = 50000000\nECHO   = NONE\nBEGIN BULK\n", 1)
    n_nodes = len(meta["nodes"])
    n_elems = len(meta["elements"])
    t_gen = time.time() - t0

    # scale timeout and in-core memory with problem size -- generous, since
    # the point is to find where it breaks, not to enforce production limits
    timeout_s = max(120, min(600, n_nodes // 50 + 120))
    dbmem = min(800_000_000, max(12_000_000, n_nodes * 4000))
    ocmem = dbmem // 10

    t1 = time.time()
    try:
        f06 = run_job(deck, job_prefix="bench", timeout=timeout_s, dbmem=dbmem, ocmem=ocmem)
    except SolveError as e:
        t_solve = time.time() - t1
        log("%d nodes (%d elems): SOLVER FAILED after %.1fs (dbmem=%d) -- %s" %
            (n_nodes, n_elems, t_solve, dbmem, str(e)[:400]))
        break
    except Exception as e:
        t_solve = time.time() - t1
        log("%d nodes (%d elems): TIMED OUT / ERROR after %.1fs (limit %ds) -- %s: %s" %
            (n_nodes, n_elems, t_solve, timeout_s, type(e).__name__, e))
        break
    t_solve = time.time() - t1

    fatal = "FATAL" in f06.upper()
    has_disp = "D I S P L A C E M E N T" in f06
    log("%d nodes (%d elems): deck_gen %.2fs, solve %.1fs, dbmem=%d, fatal=%s, has_results=%s" %
        (n_nodes, n_elems, t_gen, t_solve, dbmem, fatal, has_disp))
    if fatal:
        fatal_lines = [ln.strip() for ln in f06.splitlines() if "FATAL" in ln.upper()]
        log("  fatal detail: " + " | ".join(fatal_lines[:3]))
        break
    if not has_disp:
        log("  no displacement results found -- stopping")
        break

log("DONE")
LOG.close()
