"""
Linux solver runner for Cloud Run. Simpler than the Windows version: no
MSYS2/bash wrapper needed, just a direct subprocess call with the FORTRAN
unit-file environment variables set. NASTRAN_ROOT is baked in short (/n)
in the container image for the same reason we used `subst N:` on Windows:
this 1970s Fortran uses fixed-length filename buffers (~45 chars) that
truncate long absolute paths, so job scratch dirs also stay short (/tmp/j*).
"""
import os
import shutil
import subprocess
import uuid

NASTRAN_ROOT = os.environ.get("NASTRAN_ROOT", "/n")
NASTRAN_BIN = os.path.join(NASTRAN_ROOT, "bin", "nastran.x")
RFDIR = os.path.join(NASTRAN_ROOT, "rf")

FTN_FILES = [
    ("NPTPNM", "a.nptp"), ("PLTNM", "a.plt"), ("DICTNM", "a.dict"), ("PUNCHNM", "a.pch"),
    ("OPTPNM", "a.opt"), ("LOGNM", "a.f04"), ("IN12", "a.in12"), ("OUT11", "a.out11"),
    ("FTN11", "a.f11"), ("FTN12", "a.f12"), ("FTN13", "a.f13"), ("FTN14", "a.f14"),
    ("FTN15", "a.f15"), ("FTN16", "a.f16"), ("FTN17", "a.f17"), ("FTN18", "a.f18"),
    ("FTN19", "a.f19"), ("FTN20", "a.f20"), ("FTN21", "a.f21"), ("FTN22", "a.f22"),
    ("FTN23", "a.f23"), ("SOF1", "a.sof1"), ("SOF2", "a.sof2"),
]


class SolveError(Exception):
    pass


def run_job(deck_text, job_prefix="job", timeout=120, dbmem=12000000, ocmem=2000000):
    """
    dbmem/ocmem are NASTRAN's in-core working-set sizes (words), carved out
    of this build's fixed COMMON/ZZZZZZ/IZ(14000000) array -- 14,000,000
    words total is a hard, compile-time ceiling (see src/nastrn.f) that no
    env var can exceed. timeout/dbmem/ocmem exist as overrides for scale
    testing (see bench_scale.py), not for routine use.
    """
    job_dir = "/tmp/j" + uuid.uuid4().hex[:8]
    os.makedirs(job_dir, exist_ok=True)
    inp_path = os.path.join(job_dir, "in.inp")
    f06_path = os.path.join(job_dir, "out.f06")

    with open(inp_path, "w", newline="\n") as f:
        f.write(deck_text)

    env = dict(os.environ)
    env["DIRCTY"] = job_dir
    env["RFDIR"] = RFDIR
    for name, fname in FTN_FILES:
        env[name] = os.path.join(job_dir, fname)
    env["DBMEM"] = str(dbmem)
    env["OCMEM"] = str(ocmem)

    try:
        with open(inp_path) as inf, open(f06_path, "w") as outf:
            proc = subprocess.run(
                [NASTRAN_BIN], stdin=inf, stdout=outf, stderr=subprocess.PIPE,
                env=env, cwd=job_dir, timeout=timeout, text=True,
            )
    except subprocess.TimeoutExpired:
        shutil.rmtree(job_dir, ignore_errors=True)
        raise SolveError("solver timed out after %ds" % timeout)

    if not os.path.exists(f06_path):
        stderr = proc.stderr if proc else ""
        shutil.rmtree(job_dir, ignore_errors=True)
        raise SolveError(".f06 output not produced; stderr=%r" % stderr)

    with open(f06_path, errors="replace") as f:
        f06_text = f.read()

    shutil.rmtree(job_dir, ignore_errors=True)
    return f06_text
