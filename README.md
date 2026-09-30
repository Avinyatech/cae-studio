# NASTRAN Design Studio

A full-stack structural analysis tool built around **NASTRAN-95** — the
1976 NASA/COSMIC finite element solver, released into the public domain
under the NASA Open Source Agreement. This project rebuilds that solver
from source for modern toolchains and wraps it in a real application: a
FastAPI backend, a web UI, a mobile app, an offline LLM assistant, and a
cloud deployment path.

It runs four of NASTRAN-95's rigid formats end to end, each validated
against hand-calculated theory or a second independent solve:

| Analysis | Rigid Format | Validated against |
|---|---|---|
| Normal modes (natural frequencies) | SOL 3 | Euler–Bernoulli beam theory: 167 Hz predicted, 167.6 Hz solved |
| Static deflection under load | SOL 1 | *P·L³/3EI*: 5.95 mm predicted, 5.81 mm solved |
| Direct frequency response | SOL 8 | Resonance peak at 168.75 Hz, matching the SOL 3 result |
| Nonlinear (piecewise-linear) static | SOL 6 | Bilinear-plastic material; deflection ratio between load steps exceeds the load-factor ratio past yield, the expected softening signature |

It also generates a tetrahedral volume mesh (via Gmsh) from an uploaded
STL file, so you're not limited to simple parametric shapes — arbitrary
"medium complexity" geometry works too, with boundary conditions assigned
by clicking faces in an interactive 3D viewer.

For models too big for NASTRAN-95's compiled binary to hold at all (its
in-core working memory is a fixed-size array set at compile time — see
[Hard-won lessons](#hard-won-lessons) #11), the cloud deployment adds a
from-scratch sparse/iterative solver with genuine MPI-based distributed
domain decomposition — see `cloud/README.md`'s
[Large-model solver](cloud/README.md#large-model-solver) section.

## Why this exists

NASTRAN-95's original build environment (Sun/Solaris/VMS-era Fortran
toolchains) doesn't exist anymore. Getting it to compile and run correctly
on a modern machine, and produce *correct* results rather than silently
wrong ones, took real debugging. That debugging record is worth more than
the code — see [Hard-won lessons](#hard-won-lessons) below.

## Repository layout

```
backend/    FastAPI service: deck generation, solver invocation, result
            parsing, local LLM assistant, STL-to-tet-mesh pipeline.
            Targets Windows + MSYS2/mingw64 for local development.
web/        The design studio itself: a single-file web app (parametric
            and uploaded-geometry workflows, 3D viewers, live charts).
mobile/     Expo/React Native client for the same backend API.
cloud/      Linux/Docker-deployable variant of the backend (Cloud Run,
            Render, Fly.io, or any host that reads $PORT), with a minimal
            copy of the NASTRAN-95 source needed to build it.
```

Each directory has its own README with setup instructions.

## Quickstart (local, Windows)

```bash
# 1. Get NASTRAN-95 source and build it (see backend/README.md for the
#    full toolchain setup: MSYS2, gfortran, the specific build fixes).
# 2. Start the backend
cd backend
pip install -r requirements.txt
python -m uvicorn app:app --host 0.0.0.0 --port 8000

# 3. Open web/index.html in a browser, or run the mobile app:
cd mobile
npm install
npx expo start
```

The web UI talks to `http://127.0.0.1:8000` by default (editable in the
page itself). Optional: install [Ollama](https://ollama.com) and pull
`qwen2.5:7b-instruct` for the natural-language assistant features — the
app works without it, just without that panel.

## Hard-won lessons

Every one of these was a real failure with a cryptic message, tracked
down by comparing against NASA's own unmodified demo decks. They're
encoded directly in the code (`backend/deck_generator.py`'s docstring,
`backend/assistant.py`'s knowledge base) so the LLM assistant can
diagnose them too, not just this README.

1. **A flat shell mesh has zero drilling stiffness.** `CQUAD4` elements
   provide no resistance to rotation about their own normal (component 6)
   in a purely planar mesh. Left unconstrained, the stiffness matrix is
   singular (`ATTEMPT TO PERFORM CHOLESKY DECOMPOSITION ON A NEGATIVE
   DEFINITE MATRIX`). Fix: `GRDSET` with `PS=6`.

2. **`CTETRA` solids have zero rotational stiffness at every node**, not
   just one drilling component — they're pure translational elements.
   Same symptom, same fix, but `PS=456`.

3. **`CQUAD4`'s lumped mass matrix has zero rotary inertia.** The Givens
   eigensolver needs a non-singular full mass matrix and fails
   (`SYMMETRIC DECOMPOSITION OF DATA BLOCK MAA ... SINGULAR`). Fix: use
   `METHOD=INV` (inverse power), which auto-omits massless DOF via static
   condensation.

4. **`EIGR`'s `NE` field must be a genuine upper bound on the root count**,
   not just how many modes you want. An underestimate makes the solver
   silently converge on the *wrong* — too high frequency — subset of
   roots, with only an easy-to-miss informational note
   (`N EIGENVALUE(S) AT LOW FREQ. END NOT FOUND`) as a clue. Fix: set
   `NE`/`ND` to the true mass-bearing DOF count, then truncate to what the
   caller asked for after solving.

5. **`SOL 1,0` and `SOL 8,1`** (any `SOL` card with a `,N` approach-code
   suffix) **crash on an internally auto-generated `SBST` card** —
   `USER FATAL MESSAGE 8020, SYNTAX ERROR NEAR COLUMN 16`. Confirmed as a
   genuine bug in this build by running NASA's own unmodified demo decks.
   Fix: drop the approach code (`SOL 1`, `SOL 8`).

6. **1970s BOZ hex/octal constants** (bit-mask literals in postfix
   notation) **are rejected by modern gfortran by default.** Fix:
   `-fallow-invalid-boz`. Legacy argument-type mismatches between caller
   and callee need `-fallow-argument-mismatch` alongside it.

7. **Building `libnas.a` as a nested "thin" archive** (`ar crT`, combining
   three sub-archives into one) **fails to link** on a modern toolchain —
   `error adding symbols: file format not recognized`. GNU `ld` here
   doesn't resolve nested thin-archive members the way the original build
   environment did. Fix: build the archive directly from `.o` files.

8. **Bulk data cards are fixed-8-column format.** A manually typed card
   with a miscounted space shifts every following field by one column,
   producing cryptic parse errors (`ILLEGAL NUMBER OF WORDS`, `POSSIBLE
   ERROR IN EXPONENT`, continuation cards with `NO PARENTS`). Always
   generate cards programmatically with exact padding — see `_card()` in
   `deck_generator.py`.

9. **Internal FORTRAN filename buffers are fixed-length (~45 chars).** A
   long working-directory path truncates mid-string (`SYSTEM FATAL
   MESSAGE, RFOPEN CAN NOT OPEN ...`). Run the solver from a short path
   (locally, a `subst`-mapped drive letter; in the cloud image, `/n`).

10. **`.f06` output wraps long tables across pages, re-emitting the
    section header on each page.** A naive parser that stops at the first
    page break silently drops most of the data — including when the
    *same* header repeats mid-table with no new content in between (a
    plain static run, not just per-load-step nonlinear results): the
    naive fix of "keep whichever page had the most rows" quietly drops
    every page but one. `f06_parser.py` does a two-pass scan: find every
    header occurrence first, then merge rows across all of them by node
    ID, regardless of what page they're on.

11. **NASTRAN-95's entire in-core working memory is one fixed-size array,
    baked in at compile time.** `COMMON/ZZZZZZ/IZ(14000000)` in
    `src/nastrn.f` — the `DBMEM`/`OCMEM` env vars only divide that fixed
    14-million-word pool, they can't exceed it (`LARGEST VALUE FOR OPEN
    CORE ALLOWED IS: 14000000`). Empirically this caps the solver
    somewhere between 32,000 and 128,000 nodes; going bigger needs a
    genuinely different solver, not a config change — see
    `cloud/README.md`'s Large-model solver section. Separately, any
    non-trivial mesh's bulk-data echo and result tables also hit an
    *unrelated* default output cap (`MAXLINES=20000`,
    `USER FATAL MESSAGE 3019`) well before that real ceiling — fixed by
    adding `ECHO=NONE` and a much higher `MAXLINES` to every generated
    deck's case control.

12. **`scipy`/`numpy`'s OpenBLAS-linked wheels need `libgomp.so.1` (GNU
    OpenMP runtime) at import time**, which `python:3.11-slim` doesn't
    ship. Without it, `import scipy` crash-loops the whole container on
    startup (`OSError: libgomp.so.1: cannot open shared object file`) —
    it looks like nothing is wrong with the code, because nothing is;
    the base image is just missing a system library. Fix: `apt-get
    install libgomp1` alongside the other runtime deps.

## License

This project's own code (everything outside `cloud/nastran_src` and any
vendored NASTRAN-95 source) is original work. The NASTRAN-95 solver
itself is public domain under the
[NASA Open Source Agreement v1.3](https://github.com/nasa/NASTRAN-95),
originally released by NASA/COSMIC; this project uses the Windows/Linux
build fixes contributed by the
[AeroDME fork](https://github.com/aerodme/nastran-95) (MIT licensed) on
top of that base. See `cloud/nastran_src/` for the minimal source subset
vendored for the Docker build.
