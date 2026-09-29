import os
import re
import subprocess
import tempfile
import time
import uuid

BASH = r"C:\msys64\usr\bin\bash.exe"
NROOT = "/n"  # subst'd drive mapped to the NASTRAN-95 install (see README.md)

RUN_SCRIPT = """#!/bin/sh
set -e
cd {nroot}
mkdir -p OUTPUT
mkdir -p /tmp/{jobtmp}
export DIRCTY=/tmp/{jobtmp}
export RFDIR={nroot}/rf
export NPTPNM=OUTPUT/{job}.nptp
export PLTNM=OUTPUT/{job}.plt
export DICTNM=OUTPUT/{job}.dict
export PUNCHNM=OUTPUT/{job}.pch
export OPTPNM=OUTPUT/{job}.opt
export LOGNM=OUTPUT/{job}.f04
export IN12=OUTPUT/{job}.in12
export OUT11=OUTPUT/{job}.out11
export FTN11=OUTPUT/{job}.f11
export FTN12=OUTPUT/{job}.f12
export FTN13=OUTPUT/{job}.f13
export FTN14=OUTPUT/{job}.f14
export FTN15=OUTPUT/{job}.f15
export FTN16=OUTPUT/{job}.f16
export FTN17=OUTPUT/{job}.f17
export FTN18=OUTPUT/{job}.f18
export FTN19=OUTPUT/{job}.f19
export FTN20=OUTPUT/{job}.f20
export FTN21=OUTPUT/{job}.f21
export FTN22=OUTPUT/{job}.f22
export FTN23=OUTPUT/{job}.f23
export SOF1=OUTPUT/{job}.sof1
export SOF2=OUTPUT/{job}.sof2
export DBMEM=12000000
export OCMEM=2000000
{nroot}/bin/nastran.x < {nroot}/inp/{job}.inp > OUTPUT/{job}.f06 2>OUTPUT/{job}.stderr
echo "EXITCODE:$?"
"""


class SolveError(Exception):
    pass


def _win_path(nroot_relpath):
    """/n/foo -> N:\\foo -- caller must know N: is subst'd to the nastran root."""
    return "N:\\" + nroot_relpath.replace("/", "\\")


def run_job(deck_text, job_prefix="job"):
    """
    Writes deck_text to N:/inp/<job>.inp, runs nastran.x, returns the .f06
    text. Raises SolveError on subprocess failure (the caller should still
    check the parsed result's ok/error, since NASTRAN itself reports
    modeling FATAL errors inside a .f06 that exits 0).
    """
    job = "%s_%s" % (job_prefix, uuid.uuid4().hex[:8])
    inp_path = _win_path("inp/%s.inp" % job)
    f06_path = _win_path("OUTPUT/%s.f06" % job)

    os.makedirs(os.path.dirname(inp_path), exist_ok=True)
    with open(inp_path, "w", newline="\n") as f:
        f.write(deck_text)

    script = RUN_SCRIPT.format(nroot=NROOT, job=job, jobtmp=job)
    script_win_path = _win_path("run_%s.sh" % job)
    with open(script_win_path, "w", newline="\n") as f:
        f.write(script)

    env = dict(os.environ)
    env["MSYSTEM"] = "MINGW64"
    proc = subprocess.run(
        [BASH, "-lc", "sh %s/run_%s.sh" % (NROOT, job)],
        env=env, capture_output=True, text=True, timeout=120,
    )

    try:
        os.remove(script_win_path)
    except OSError:
        pass

    if "EXITCODE:0" not in proc.stdout:
        raise SolveError("solver process failed: stdout=%r stderr=%r" % (proc.stdout, proc.stderr))

    if not os.path.exists(f06_path):
        raise SolveError(".f06 output not found at %s" % f06_path)

    with open(f06_path, "r", errors="replace") as f:
        f06_text = f.read()

    for ext in ("inp", "f06", "stderr", "f04", "nptp", "plt", "dict", "pch", "opt",
                "in12", "out11", "f11", "f12", "f13", "f14", "f15", "f16", "f17",
                "f18", "f19", "f20", "f21", "f22", "f23", "sof1", "sof2"):
        p = _win_path("OUTPUT/%s.%s" % (job, ext)) if ext != "inp" else inp_path
        try:
            os.remove(p)
        except OSError:
            pass

    return f06_text
