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

`hpc_solver/` is a from-scratch alternative built to get past that:

- a 4-node constant-strain tetrahedron element (same family as
  NASTRAN's own `CTETRA`), fully vectorized with numpy so assembly
  stays practical into the millions of elements
- a sparse global stiffness matrix (`scipy.sparse`) with a Jacobi-
  preconditioned conjugate-gradient solve — no fixed-size array, no
  direct decomposition, limited only by available RAM
- genuine MPI-based distributed-memory domain decomposition
  (`dcg.py`/`driver.py` — recursive coordinate bisection partitioning,
  distributed CG via `Allreduce`/`Allgatherv`), validated locally to
  give bit-identical results across 1/2/4/8 MPI ranks on a 216,000-node
  mesh. Not deployed here (no MPI runtime in this image, and a free
  instance has no cluster to distribute across) -- it's the
  architecture ready for whenever this runs on real multi-node
  hardware, not something this deployment exercises.

Correctness was cross-checked directly against NASTRAN-95's own
`CTETRA` solve on identical mesh/material/BCs/load — agreement improved
from ~2.5% to ~0.9% as the mesh was refined, the expected convergence
pattern for two independent, correct implementations of the same
physics.

`MAX_HPC_NODES` (default 3000) caps request size, set from live
bisection testing against the actual deployed free-tier instance, not
a theoretical estimate: 3,375 nodes solved cleanly in a few seconds,
but 3,840 nodes hung for 60+ seconds with no response and needed a
container restart — a cliff this sharp means the instance was
thrashing out of RAM, not slowing down gracefully. Raise this env var
only after confirming headroom the same way, directly against the
deployed instance.
