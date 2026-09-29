# Backend

FastAPI service wrapping the NASTRAN-95 solver: parametric shell decks,
uploaded-geometry tet meshes, result parsing, and a local LLM assistant.
Targets Windows + MSYS2/mingw64 for local development (see `../cloud/`
for the Linux/Docker variant).

## Setup

You'll need a working `nastran.x` binary and its `rf/` rigid-format
directory. Get the source and build it:

```bash
git clone https://github.com/aerodme/nastran-95.git
# See the root README's "Hard-won lessons" for the exact makefile fixes
# needed (BOZ constants, argument mismatches, thin-archive linking) --
# apply them to the makefile, then, from an MSYS2 mingw64 shell:
make nastran
```

Point `NASTRAN_ROOT` in `app.py` at that checkout (or set the env var, if
you adapt the startup hook), then:

```bash
pip install -r requirements.txt
python -m uvicorn app:app --host 0.0.0.0 --port 8000
```

Optional: install [Ollama](https://ollama.com) and run
`ollama pull qwen2.5:7b-instruct` for the `/assistant/*` endpoints
(natural-language deck generation and error explanation). Everything else
works without it.

## Files

- `app.py` — FastAPI routes and request/response schemas
- `deck_generator.py` — parametric shell decks and uploaded-geometry tet
  decks (`build_deck` / `build_deck_from_tetmesh`)
- `solver_runner.py` — invokes `nastran.x` via MSYS2, manages FORTRAN unit
  file env vars and scratch directories
- `f06_parser.py` — parses displacements, eigenvectors, complex frequency
  response, and element stress out of `.f06` text
- `tetmesh.py` — STL → tetrahedral volume mesh via Gmsh
- `assistant.py` — Ollama-backed natural language ↔ analysis parameters,
  and error diagnosis grounded in the project's own debugging history

## Endpoints

| Route | Purpose |
|---|---|
| `POST /analyze` | Parametric shell analysis (modes / static / freq response) |
| `POST /upload_geometry` | Tet-mesh an uploaded STL |
| `POST /analyze_mesh` | Solve on an uploaded/meshed geometry |
| `POST /assistant/parse` | Natural language → analysis parameters |
| `POST /assistant/explain` | Explain a solver error |
| `GET /materials` | Built-in material presets |
| `GET /health` | Solver/Ollama reachability check |
