import re

FATAL_RE = re.compile(r"FATAL", re.IGNORECASE)
MODE_HDR_RE = re.compile(r"EIGENVALUE\s*=\s*([\d.+\-E]+)\s*\(CYCLIC FREQUENCY\s*=\s*([\d.+\-E]+)\s*HZ\)")
VECTOR_HDR_RE = re.compile(r"R E A L   E I G E N V E C T O R   N O\.\s*(\d+)")
DISP_HDR_RE = re.compile(r"D I S P L A C E M E N T   V E C T O R")
ROW_RE = re.compile(
    r"^\s*(\d+)\s+G\s+([\d.\-+E]+)\s+([\d.\-+E]+)\s+([\d.\-+E]+)\s+([\d.\-+E]+)\s+([\d.\-+E]+)\s+([\d.\-+E]+)\s*$"
)

LOAD_FACTOR_HDR_RE = re.compile(r"LOAD\s+FACTOR\s+(\d+)")
EPSILON_RE = re.compile(r"FOR SUBCASE NUMBER\s+(\d+),\s*EPSILON SUB E\s*=\s*([\d.\-+E]+)")

POINT_HDR_RE = re.compile(r"POINT-ID\s*=\s*(\d+)")
# Each frequency row is preceded by a standalone Fortran carriage-control "0"
# (double-space-before marker) as its own whitespace-separated token, distinct
# from the actual frequency value that follows -- optionally consume it.
FREQ_ROW_RE = re.compile(
    r"^\s*(?:0\s+)?([\d.\-+E]+)\s+G\s+([\d.\-+E]+)\s+([\d.\-+E]+)\s+([\d.\-+E]+)\s+([\d.\-+E]+)\s+([\d.\-+E]+)\s+([\d.\-+E]+)\s*$"
)
NUM6_RE = re.compile(r"^\s*([\d.\-+E]+)\s+([\d.\-+E]+)\s+([\d.\-+E]+)\s+([\d.\-+E]+)\s+([\d.\-+E]+)\s+([\d.\-+E]+)\s*$")

STRESS_HDR_RE = re.compile(r"S T R E S S E S   I N   G E N E R A L   Q U A D R I L A T E R A L   E L E M E N T S")
# Bottom-fiber row: element id + fibre distance + 7 values (normal-x, normal-y,
# shear-xy, angle, major, minor, max shear). Same leading carriage-control "0"
# as the frequency rows above.
STRESS_ROW1_RE = re.compile(
    r"^\s*(?:0\s+)?(\d+)\s+([\d.\-+E]+)\s+([\d.\-+E]+)\s+([\d.\-+E]+)\s+([\d.\-+E]+)\s+([\d.\-+E]+)\s+([\d.\-+E]+)\s+([\d.\-+E]+)\s+([\d.\-+E]+)\s*$"
)
# Top-fiber continuation row: same 8 fields, no element id.
STRESS_ROW2_RE = re.compile(
    r"^\s+([\d.\-+E]+)\s+([\d.\-+E]+)\s+([\d.\-+E]+)\s+([\d.\-+E]+)\s+([\d.\-+E]+)\s+([\d.\-+E]+)\s+([\d.\-+E]+)\s+([\d.\-+E]+)\s*$"
)


def _f(x):
    return float(x)


def _collect_vectors(lines, start, end):
    """Scan lines[start:end] for POINT ID rows, tolerating page breaks /
    repeated column headers in between (NASTRAN wraps long tables across
    pages)."""
    vectors = {}
    for k in range(start, end):
        rm = ROW_RE.match(lines[k])
        if rm:
            gid = int(rm.group(1))
            vectors[gid] = [_f(rm.group(x)) for x in range(2, 8)]
    return vectors


def _collect_freq_points(lines, start, end):
    """Scan lines[start:end] for (frequency, magnitude[6], phase[6]) rows
    from a COMPLEX DISPLACEMENT VECTOR (SORT2, MAGNITUDE/PHASE) block,
    tolerating page breaks the same way as _collect_vectors."""
    points = []
    k = start
    while k < end:
        rm = FREQ_ROW_RE.match(lines[k])
        if rm:
            freq_hz = _f(rm.group(1))
            mag = [_f(rm.group(x)) for x in range(2, 8)]
            phase = [0.0] * 6
            if k + 1 < end:
                pm = NUM6_RE.match(lines[k + 1])
                if pm:
                    phase = [_f(pm.group(x)) for x in range(1, 7)]
            points.append(dict(freq_hz=freq_hz, mag=mag, phase=phase))
        k += 1
    return points


def _stress_row(vals):
    return dict(normal_x=vals[0], normal_y=vals[1], shear_xy=vals[2], angle=vals[3],
                major=vals[4], minor=vals[5], max_shear=vals[6])


def _collect_stress(lines, start, end):
    """Scan lines[start:end] for CQUAD4 stress rows (bottom fiber + top
    fiber continuation line per element), tolerating page breaks."""
    stress = {}
    k = start
    while k < end:
        rm = STRESS_ROW1_RE.match(lines[k])
        if rm:
            eid = int(rm.group(1))
            bottom = _stress_row([_f(rm.group(x)) for x in range(3, 10)])
            top = None
            if k + 1 < end:
                pm = STRESS_ROW2_RE.match(lines[k + 1])
                if pm:
                    top = _stress_row([_f(pm.group(x)) for x in range(2, 9)])
            stress[eid] = dict(bottom=bottom, top=top)
        k += 1
    return stress


def parse_f06(text):
    """
    Returns dict:
      ok: bool
      error: str or None
      analysis: "modes" | "static" | "freq_response" | None
      modes: [ {mode, freq_hz, eigenvalue, vectors: {node_id: [T1..R3]}} ]
      static: {vectors: {node_id: [T1..R3]}} or None
      freq_response: {points_by_node: {node_id: [{freq_hz, mag, phase}, ...]}} or None
    """
    if FATAL_RE.search(text):
        fatal_lines = [ln.strip() for ln in text.splitlines() if "FATAL" in ln.upper()]
        return dict(ok=False, error=" | ".join(fatal_lines[:3]), analysis=None,
                    modes=[], static=None, freq_response=None)

    lines = text.splitlines()
    n = len(lines)

    # Pass 1: find every section-start line (mode header, static disp header,
    # or complex-frequency-response point header). A nonlinear (SOL 6) run
    # prints one "LOAD FACTOR N" banner before each step's displacement
    # table -- track the most recently seen one so each "static" start can
    # be tagged with the load step it belongs to.
    starts = []  # (line_idx, kind, extra)
    current_load_step = None
    for idx, ln in enumerate(lines):
        lm = LOAD_FACTOR_HDR_RE.search(ln)
        if lm:
            current_load_step = int(lm.group(1))
        m = MODE_HDR_RE.search(ln)
        if m:
            mode_no = None
            for j in range(idx + 1, min(idx + 6, n)):
                mv = VECTOR_HDR_RE.search(lines[j])
                if mv:
                    mode_no = int(mv.group(1))
                    break
            starts.append((idx, "mode", dict(freq_hz=_f(m.group(2)), mode_no=mode_no)))
            continue
        pm = POINT_HDR_RE.search(ln)
        if pm:
            starts.append((idx, "freq_point", dict(node_id=int(pm.group(1)))))
            continue
        if STRESS_HDR_RE.search(ln):
            starts.append((idx, "stress_table", {}))
            continue
        if DISP_HDR_RE.search(ln) and "C O M P L E X" not in ln.upper():
            starts.append((idx, "static", dict(load_step=current_load_step)))

    epsilons = {int(mo.group(1)): _f(mo.group(2)) for mo in EPSILON_RE.finditer(text)}

    modes = []
    static_vectors = None
    nonlinear_steps = {}  # load_step -> vectors (merged across page wraps)
    freq_points_by_node = {}
    element_stress = {}

    for i, (idx, kind, extra) in enumerate(starts):
        end = starts[i + 1][0] if i + 1 < len(starts) else n
        if kind == "mode":
            vectors = _collect_vectors(lines, idx, end)
            modes.append(dict(mode=extra["mode_no"] or (len(modes) + 1),
                               freq_hz=extra["freq_hz"], vectors=vectors))
        elif kind == "static":
            vectors = _collect_vectors(lines, idx, end)
            step = extra.get("load_step")
            if step is not None:
                existing = nonlinear_steps.get(step)
                if existing is None or len(vectors) > len(existing):
                    nonlinear_steps[step] = vectors
            else:
                if static_vectors is None or len(vectors) > len(static_vectors):
                    static_vectors = vectors
        elif kind == "freq_point":
            pts = _collect_freq_points(lines, idx, end)
            node_id = extra["node_id"]
            freq_points_by_node.setdefault(node_id, [])
            # a point's table wraps across many pages, each re-emitting the
            # "POINT-ID =" header, so accumulate rather than overwrite
            freq_points_by_node[node_id].extend(pts)
        elif kind == "stress_table":
            # the table wraps across pages under a repeated header, each page
            # covering a different slice of element ids -- merge them all
            element_stress.update(_collect_stress(lines, idx, end))

    if modes:
        analysis = "modes"
    elif nonlinear_steps:
        analysis = "nonlinear"
    elif freq_points_by_node:
        analysis = "freq_response"
    elif static_vectors:
        analysis = "static"
    else:
        analysis = None

    ok = analysis is not None
    nonlinear = None
    if nonlinear_steps:
        steps = []
        for step in sorted(nonlinear_steps):
            steps.append(dict(step=step, vectors=nonlinear_steps[step], epsilon=epsilons.get(step)))
        nonlinear = dict(steps=steps)

    return dict(
        ok=ok,
        error=None if ok else "no results found in .f06 (check deck/echo for a non-fatal parse issue)",
        analysis=analysis,
        modes=modes,
        static=dict(vectors=static_vectors, element_stress=element_stress) if static_vectors else None,
        freq_response=dict(points_by_node=freq_points_by_node) if freq_points_by_node else None,
        nonlinear=nonlinear,
    )
