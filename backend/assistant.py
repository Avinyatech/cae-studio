"""
Local LLM assistant for the design studio, backed by Ollama running on
this machine (http://localhost:11434) -- no external API calls, no data
leaves the PC. Two jobs:

  1. parse_intent(text): natural language -> AnalysisRequest JSON, so a
     user can type "200mm aluminum cantilever, 10 elements, first 5 modes"
     instead of filling in the form by hand.
  2. explain_error(...): given a failed run's error and deck, explain it
     in plain English and suggest a fix. The system prompt below encodes
     the actual NASTRAN-95-on-modern-gfortran quirks discovered by hand
     during this project, so the answers are grounded in what's really
     going on in this specific solver build rather than generic textbook
     NASTRAN advice that wouldn't apply here.
"""
import json
import re

import requests

OLLAMA_URL = "http://localhost:11434"
MODEL = "qwen2.5:7b-instruct"

SCHEMA_PROMPT = """You translate a plain-English structural design request into a JSON object
matching this exact schema (all fields required unless noted):

{
  "length_mm": float,        // along the beam/plate length
  "width_mm": float,
  "thickness_mm": float,
  "nx": int,                 // mesh divisions along length, 1-30
  "ny": int,                 // mesh divisions across width, 1-12
  "material": "steel" | "aluminum" | "titanium",
  "bc": "cantilever" | "simply_supported" | "fixed_fixed",
  "analysis": "modes" | "static" | "freq_response",
  "num_modes": int,          // only meaningful for analysis="modes", default 6
  "load_n": float,           // only meaningful for analysis="static"/"freq_response", default 50
  "load_dir": "x" | "y" | "z",  // default "z" (out-of-plane / transverse bending)
  "freq_start_hz": float,    // only for analysis="freq_response", default 50
  "freq_end_hz": float,      // only for analysis="freq_response", default 300
  "num_freq_points": int,    // only for analysis="freq_response", default 41
  "damping_g": float         // only for analysis="freq_response", default 0.02
}

Rules:
- Output ONLY the JSON object. No markdown fences, no prose, no explanation.
- Fill every field with a sensible value even if the user didn't mention it
  (use the defaults noted above).
- "cantilever" = fixed at one end. "simply supported" = pinned both ends.
  "fixed-fixed" / "clamped-clamped" = fully fixed both ends.
- If the user asks for natural frequencies / mode shapes / vibration, use
  analysis="modes". If they ask for deflection / stress under a load, use
  "static". If they ask for resonance / frequency sweep / harmonic response
  / FRF, use "freq_response".
"""

DOMAIN_NOTES = """Known, verified quirks of this specific NASTRAN-95 (COSMIC, 1976) build,
compiled with modern gfortran via MSYS2/mingw64. Use these to give SPECIFIC,
grounded diagnoses -- don't give generic NASTRAN advice that doesn't apply here.

1. Flat CQUAD4 shell meshes have ZERO stiffness against drilling rotation
   (RZ, component 6) at every node. Without a GRDSET card constraining
   component 6 globally, the stiffness matrix is singular and Cholesky
   decomposition fails ("ATTEMPT TO PERFORM CHOLESKY DECOMPOSITION ON A
   NEGATIVE DEFINITE MATRIX" or "SYMMETRIC DECOMPOSITION ... ABORTED").
   Fix: add a GRDSET card with PS (field 8) = "6".

2. CQUAD4's lumped mass matrix has zero rotary inertia (R1, R2 have no
   mass). The Givens (GIV) eigensolver needs a non-singular full mass
   matrix and fails ("SYMMETRIC DECOMPOSITION OF DATA BLOCK MAA ...
   SINGULAR"). Fix: use METHOD=INV (inverse power) instead of GIV, which
   auto-omits massless DOF via static condensation.

3. For EIGR with METHOD=INV/DET, the NE field (estimated root count) MUST
   be a genuine upper bound on the number of roots in [F1,F2] -- an
   under-estimate makes the solver silently converge on the WRONG (too
   high frequency) subset of roots with NO error message, just a
   "N EIGENVALUE(S) AT LOW FREQ. END NOT FOUND" informational note easy to
   miss. Fix: set NE/ND to the true count of mass-bearing DOF (3 per free
   grid: T1,T2,T3) rather than just how many modes the user wants.

4. F1 on EIGR should be slightly above 0.0 (e.g. 0.1), not exactly 0.0, to
   avoid a rigid-body/zero-frequency numerical edge case.

5. "SOL 1,0" and "SOL 8,1" (any SOL card with a ",N" approach-code suffix)
   crash with "USER FATAL MESSAGE 8020, SYNTAX ERROR NEAR COLUMN 16" on an
   internally auto-generated "SBST" (substructure preface) card -- this is
   a genuine bug in this build's substructure-preface logic, confirmed by
   running NASA's own unmodified demo decks. Fix: use the SOL number alone
   with no approach code ("SOL 1", "SOL 8").

6. 1970s-style BOZ hex/octal constants (e.g. postfix notation used for bit
   masks) are rejected by modern gfortran by default ("BOZ constant ...
   uses nonstandard postfix syntax"). Fix: compile with
   -fallow-invalid-boz. Similarly, legacy argument-type mismatches between
   caller and callee need -fallow-argument-mismatch.

7. Building libnas.a by nesting three sub-archives into one "thin" archive
   (ar crT) fails to link on this toolchain ("error adding symbols: file
   format not recognized") -- GNU ld here doesn't resolve nested thin-
   archive members the way the original build environment did. Fix: build
   libnas.a directly from the .o files instead of from the three .a files.

8. NASTRAN bulk data cards are fixed-8-column format. A field that is
   exactly 8 characters wide has no visible trailing space before the next
   field starts -- that's correct, not a bug. But a MANUALLY typed card
   with a miscounted space shifts every subsequent field by one column and
   causes cryptic parse errors ("ILLEGAL NUMBER OF WORDS", "POSSIBLE ERROR
   IN EXPONENT", continuation cards with "NO PARENTS"). Always generate
   cards programmatically with exact 8-char padding per field.

9. This build's internal FORTRAN filename buffers are fixed-length
   (~45 chars). A long absolute working-directory path truncates mid-string
   ("SYSTEM FATAL MESSAGE, RFOPEN CAN NOT OPEN ..."). Fix: run the solver
   from a short path.
"""


def _ollama_generate(prompt, system, temperature=0.1):
    resp = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json=dict(model=MODEL, prompt=prompt, system=system, stream=False,
                   options=dict(temperature=temperature)),
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()["response"]


def _extract_json(text):
    text = text.strip()
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        raise ValueError("model did not return a JSON object: %r" % text[:300])
    return json.loads(m.group(0))


def ollama_available():
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=3)
        return r.status_code == 200
    except requests.RequestException:
        return False


def parse_intent(text):
    """Natural language -> dict matching AnalysisRequest."""
    raw = _ollama_generate(prompt=text, system=SCHEMA_PROMPT, temperature=0.1)
    return _extract_json(raw)


def explain_error(error_text, deck_excerpt=""):
    """Failed-run error text (+ optional deck excerpt) -> plain-English diagnosis."""
    system = (
        "You are a NASTRAN troubleshooting assistant for this specific solver build. "
        "Use the following verified knowledge base to ground your answer. If the error "
        "matches one of these known issues, say so specifically and give the exact fix. "
        "If it doesn't match any of them, say that plainly rather than guessing.\n\n"
        + DOMAIN_NOTES
    )
    prompt = "Solver error:\n%s\n" % error_text
    if deck_excerpt:
        prompt += "\nRelevant deck excerpt:\n%s\n" % deck_excerpt
    prompt += "\nExplain what went wrong and how to fix it, in 3-5 sentences."
    return _ollama_generate(prompt=prompt, system=system, temperature=0.2)
