# --- Stage 1: compile NASTRAN-95 (COSMIC, 1976) for Linux ---
# Same source and gfortran fixes proven on Windows (BOZ/argument-mismatch
# flags, direct-object libnas.a instead of a nested thin archive); none of
# the Windows-only workarounds (MSYS2, subst drive) are needed here.
FROM debian:bookworm-slim AS builder
RUN apt-get update && apt-get install -y --no-install-recommends \
    gfortran make binutils && rm -rf /var/lib/apt/lists/*
COPY nastran_src /n
WORKDIR /n
RUN make nastran

# --- Stage 2: runtime ---
FROM python:3.11-slim
# short root path: this 1970s Fortran uses fixed-length filename buffers
# (~45 chars) that truncate long absolute paths, so keep it at /n exactly
# like the local Windows dev setup uses a short `subst N:` drive.
RUN mkdir -p /n/bin /n/rf
COPY --from=builder /n/bin/nastran.x /n/bin/nastran.x
COPY --from=builder /n/rf /n/rf
COPY --from=builder /n/NASINFO /n/NASINFO

# gmsh's Linux wheel dynamically links against these even for headless
# (no-GUI) use -- omitting them fails at `import gmsh`, not at call time.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglu1-mesa libxrender1 libxcursor1 libxft2 libxinerama1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app.py deck_generator.py f06_parser.py solver_runner.py assistant.py tetmesh.py ./
COPY hpc_solver ./hpc_solver

ENV NASTRAN_ROOT=/n
EXPOSE 8080
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-8080}"]
