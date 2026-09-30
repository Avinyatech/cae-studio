# NASTRAN Design Studio API (Linux / Docker)

Linux/Docker-deployable twin of `../backend/` — same solver capabilities
(parametric shell decks and uploaded-geometry tet meshes; normal modes,
static, frequency-response, and nonlinear piecewise-linear analysis),
packaged for Cloud Run, Render, Fly.io, or any container host that reads
`$PORT`. The LLM assistant endpoints (`/assistant/*`) work the same way
if you run Ollama as a sidecar reachable at `localhost:11434`; otherwise
they report a clear 503 rather than failing the whole service.

It also has one capability the local backend deliberately doesn't:
`POST /analyze_large`, a from-scratch sparse/iterative solver
(`hpc_solver/`) for models bigger than NASTRAN-95's compiled binary can
hold — see [Large-model solver](#large-model-solver) below.

`nastran_src/` is a minimal copy of the NASTRAN-95 source (just the
`rf`, `mds`, `mis`, `bd`, `include`, `src` directories and the `makefile`)
carrying the fixes needed to build with a modern gfortran:

- `-fallow-invalid-boz -fallow-argument-mismatch -fno-range-check
  -fno-automatic -std=legacy` for 1970s Fortran syntax gfortran now rejects
  by default.
- `libnas.a` built directly from object files rather than nesting the
  three sub-archives as a thin archive (`ar crT`), which GNU `ld` doesn't
  resolve the way this build originally assumed.

## Deploying

Build and run via the included `Dockerfile` on any container host that
reads the `PORT` env var (Cloud Run, Render, Railway, Fly.io, ...):

```bash
docker build -t nastran-api .
docker run -p 8080:8080 nastran-api
```

## Endpoints

- `POST /analyze` — run an analysis (see `app.py` for the request schema)
- `POST /analyze_large` — large-model sparse/iterative solver (see below)
- `POST /analyze_deck` — run a raw uploaded NASTRAN `.bdf`/`.dat`/`.inp` file
  as-is (both this deployment and the local backend support this; pure I/O,
  no new solver math)
- `POST /analyze_mesh`, `POST /upload_geometry` — uploaded-STL tet-mesh workflow
- `GET /materials` — list built-in material presets
- `GET /health` — confirm the solver binary is present and built

## Large-model solver

NASTRAN-95's compiled binary has a hard, compile-time working-memory
ceiling: its entire in-core array is `COMMON/ZZZZZZ/IZ(14000000)` in
`nastran_src/src/nastrn.f`, a fixed size baked into the binary that no
request or env var can exceed. Empirically this caps the solver
somewhere between 32,000 and 128,000 nodes, and even 32,000 nodes took
~4.7 minutes on its direct/banded decomposition.

`hpc_solver/` is a from-scratch alternative built to get past that, with
three analysis types (all reachable via `analysis` in the
`/analyze_large` request body):

- **static**: a sparse global stiffness matrix (`scipy.sparse`) with a
  Jacobi-preconditioned conjugate-gradient solve — no fixed-size array,
  no direct decomposition, limited only by available RAM. Also returns
  per-element stress/strain (recovered from the displacement solution)
  and a Basquin's-equation fatigue-life estimate (`fatigue.py`) —
  documented there as a simplified engineering estimate (fully-reversed
  loading, representative not certified material constants), since
  there's no NASTRAN-95 fatigue rigid format to validate against.
- **modal**: the generalized eigenproblem via `scipy.sparse.linalg.eigsh`
  (shift-invert), a lumped translational mass matrix matching how
  NASTRAN-95 treats `CTETRA`. Live-validated by mesh refinement
  (112→468 nodes dropped the first frequency 2266→1846 Hz, the correct
  direction for a displacement-based FEM discretization relaxing toward
  the true value).
- **freq_response**: direct complex-valued frequency sweep,
  `Z(omega) = K*(1+ig) - omega^2*M`, the same structural-damping
  formulation as the NASTRAN SOL 8 path. Live-validated against the
  modal solver on the same mesh: the resonance peak landed almost
  exactly on the second modal frequency, as it should since both derive
  from the same K/M matrices.

`hpc_solver/dcg.py`/`driver.py` additionally implement genuine MPI-based
distributed-memory domain decomposition (recursive coordinate bisection
partitioning, distributed CG via `Allreduce`/`Allgatherv`), validated
locally to give bit-identical results across 1/2/4/8 MPI ranks on a
216,000-node mesh. Not deployed here (no MPI runtime in this image, and
a free instance has no cluster to distribute across) — it's the
architecture ready for whenever this runs on real multi-node hardware,
not something this deployment exercises.

Static-path correctness was cross-checked directly against NASTRAN-95's
own `CTETRA` solve on identical mesh/material/BCs/load — agreement
improved from ~2.5% to ~0.9% as the mesh was refined, the expected
convergence pattern for two independent, correct implementations of the
same physics.

Three separate node caps, each set from live bisection testing against
the actual deployed free-tier instance rather than a theoretical
estimate — **do not reuse one analysis type's cap for another**, they
have fundamentally different cost profiles per node:

| Cap | Default | Why |
|---|---|---|
| `MAX_HPC_NODES` (static) | 3000 | 3,375 nodes solved in a few seconds; 3,840 hung 60s+ and needed a container restart — a cliff this sharp means the instance was thrashing out of RAM, not slowing gracefully. |
| `MAX_HPC_NODES_MODAL` | 800 | One sparse factorization regardless of node count; 735 nodes solved in 0.51s. |
| `MAX_HPC_NODES_FRF` | 250 | One factorization *per frequency point* — a fundamentally different (multiplicative) cost profile. Sharing modal's 800 cap let a 735-node request hang 60s+; bisected alone, 250 nodes/11 points solved in 12.4s but 396 nodes/11 points hung the same way. |

Raise any of these only after confirming headroom the same way, directly
against the deployed instance — never by extrapolating from another
analysis type's number or from local testing.
