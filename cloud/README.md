# NASTRAN Design Studio API (Linux / Docker)

Linux/Docker-deployable twin of `../backend/` — same solver capabilities
(parametric shell decks and uploaded-geometry tet meshes; normal modes,
static, and frequency-response analysis), packaged for Cloud Run, Render,
Fly.io, or any container host that reads `$PORT`. The LLM assistant
endpoints (`/assistant/*`) work the same way if you run Ollama as a
sidecar reachable at `localhost:11434`; otherwise they report a clear
503 rather than failing the whole service.

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
- `GET /materials` — list built-in material presets
- `GET /health` — confirm the solver binary is present and built
