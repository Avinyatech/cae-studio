"""
Parametric NASTRAN-95 bulk-data deck generator for rectangular shell
"beam/plate" design studies: a flat CQUAD4 mesh of given length/width,
thickness, mesh density and material, with a chosen boundary condition,
solved either as a normal-modes (SOL 3) or static (SOL 1) analysis.

Lessons baked in from interactive debugging of this NASTRAN-95 build:
  - A flat shell mesh has zero stiffness against drilling rotation (RZ,
    component 6) at every node; must be constrained globally via GRDSET,
    or the stiffness matrix is singular (Cholesky failure).
  - CQUAD4's lumped mass matrix carries zero rotary inertia, so the
    Givens (GIV) eigensolver hits a singular mass matrix. Use the
    inverse-power method (INV) instead, which auto-omits massless DOF.
  - INV needs NE (estimated root count) to be a real upper bound on the
    number of roots in [F1, F2], or low-frequency roots get skipped
    entirely (no error -- just silently wrong results). We estimate
    the true mass-bearing DOF count (3 translations per free node) and
    use that as NE/ND, capped by what the user actually asked for.
  - F1 must be slightly above 0.0 (not exactly 0.0) to avoid the
    rigid-body/zero-frequency numerical edge case.
  - SOL 1,0 and SOL 8,1 (the ",N" approach-code suffix) both crash with a
    "SYNTAX ERROR NEAR COLUMN 16" on an internally auto-generated "SBST"
    card -- a bug in this build's substructure-preface logic, confirmed by
    running NASA's own unmodified demo decks. Plain "SOL 1" / "SOL 8" (no
    approach code) sidesteps that code path entirely and works correctly.
"""
import math

MATERIAL_PRESETS = {
    "steel":    dict(E=210000.0, nu=0.3,  rho=7.85e-9),
    "aluminum": dict(E=69000.0,  nu=0.33, rho=2.70e-9),
    "titanium": dict(E=114000.0, nu=0.34, rho=4.43e-9),
}


def _card(fields, cont_out=None):
    row = "".join(str(f if f is not None else "").ljust(8) for f in (list(fields) + [""] * 9)[:9])
    if cont_out:
        row += str(cont_out).ljust(8)
    return row + "\n"


def _fnum(x):
    """Format a float compactly to fit an 8-char NASTRAN field."""
    x = float(x)
    if x == 0:
        return "0.0"
    ax = abs(x)
    if 1e-3 <= ax < 1e6:
        s = repr(round(x, 6))
        if len(s) <= 8:
            return s
    # NASTRAN's exponent-without-E shorthand, e.g. 7.85e-9 -> "7.850-9"
    exp = math.floor(math.log10(ax))
    mant = x / (10 ** exp)
    for prec in (3, 2, 1, 0):
        s = ("%." + str(prec) + "f%+d") % (mant, exp)
        if len(s) <= 8:
            return s
    return s[:8]


def grid_id(ix, iy, ny):
    return ix * (ny + 1) + iy + 1


def _tables1_cards(tid, pairs):
    """
    Build a TABLES1 card plus continuations for a list of (x, y) pairs,
    packing up to 4 pairs (8 data values) per continuation line.

    Under-packing this (2 pairs/line spread across 3+ chained continuation
    cards) was found by trial to trigger a "USER FATAL MESSAGE 316 ILLEGAL
    DATA ON TABLES1" / "MESSAGE 315 FORMAT ERROR ON PLFACT" pair in this
    NASTRAN-95 build, even though every individual field was byte-aligned
    correctly -- packing each continuation line as full as it will hold
    (matching how NASA's own demo decks do it) avoids the bug.
    """
    tid = int(tid)
    vals = []
    for x, y in pairs:
        vals.append(_fnum(x))
        vals.append(_fnum(y))
    vals.append("ENDT")
    chunks = [vals[i:i + 8] for i in range(0, len(vals), 8)]
    label_base = "+T%d" % tid
    out = []
    header_cont = (label_base + "A")[:8] if len(chunks) > 1 else None
    out.append(_card(["TABLES1", str(tid)], cont_out=header_cont))
    for i, chunk in enumerate(chunks):
        this_label = (label_base + chr(ord("A") + i))[:8]
        next_label = (label_base + chr(ord("A") + i + 1))[:8] if i + 1 < len(chunks) else None
        out.append(_card([this_label] + chunk, cont_out=next_label))
    return out


def _build_nonlinear_deck(spec, L, W, t, nx, ny, bc, E, nu, rho, nodes, elements,
                           left_edge, right_edge, target_edge, monitor_node, n_mass_dof):
    """
    SOL 6 piecewise-linear static analysis: a bilinear (elastic, then
    reduced-tangent) stress-strain curve applied via MATS1/TABLES1, with
    the load ramped up in fractional increments via PLFACT. Validated
    against NASA's own d06011a.inp cracked-panel demo before generalizing.
    """
    load_dir = spec.get("load_dir", "y")
    if load_dir not in ("x", "y"):
        raise ValueError(
            "nonlinear analysis uses in-plane membrane elements (CTRMEM) with no "
            "out-of-plane stiffness -- load_dir must be 'x' or 'y', not 'z'"
        )
    yield_stress = float(spec.get("yield_stress_mpa", 250.0))
    Et = float(spec.get("tangent_modulus_mpa", E / 100.0))
    max_strain = float(spec.get("max_strain", 0.05))
    num_load_steps = max(1, min(int(spec.get("num_load_steps", 4)), 20))
    load_n = float(spec.get("load_n", 100.0))

    yield_strain = yield_stress / E
    if max_strain <= yield_strain:
        raise ValueError("max_strain must be greater than the elastic yield_strain (yield_stress/E)")
    max_stress = yield_stress + Et * (max_strain - yield_strain)

    # Triangulate the quad mesh: CTRMEM (triangular membrane) is what this
    # NASTRAN-95 build's SOL 6 nonlinear-material DMAP path actually works
    # with -- see build_deck's docstring note on why CQUAD4 is avoided here.
    tris = []
    for g1, g2, g3, g4 in elements:
        tris.append((g1, g2, g3))
        tris.append((g1, g3, g4))

    fixed_ids = sorted(set(left_edge) if bc == "cantilever" else set(left_edge) | set(right_edge))
    free_ids = [g for g in nodes if g not in fixed_ids]

    title = "NONLINEAR %.0fx%.0fx%.1fMM, %dx%d MESH, %s, YIELD=%.0fMPA" % (
        L, W, t, nx, ny, bc.upper(), yield_stress)

    lines = []
    lines.append("ID    PARAM95,NASTRAN".ljust(80) + "\n")
    lines.append("APP   DISPLACEMENT".ljust(80) + "\n")
    lines.append("SOL   6".ljust(80) + "\n")
    lines.append("TIME  30".ljust(80) + "\n")
    lines.append("CEND\n")
    lines.append("TITLE    = %s\n" % title[:72])
    lines.append("SPC   = 10\n")
    lines.append("LOAD  = 20\n")
    lines.append("PLCOEFFICIENT = 30\n")
    lines.append("DISP  = ALL\n")
    lines.append("BEGIN BULK\n")

    lines.append("$ CTRMEM is a pure in-plane membrane (no bending, no out-of-plane\n")
    lines.append("$ stiffness) -- globally fix T3 and all rotations, nothing solves for them.\n")
    lines.append(_card(["GRDSET", "", "", "", "", "", "", "3456"]))

    lines.append("$ Grid points\n")
    for gid in sorted(nodes):
        x, y, z = nodes[gid]
        lines.append(_card(["GRID", str(gid), "", _fnum(x), _fnum(y), _fnum(z)]))

    lines.append("$ Elements: %d CTRMEM triangular membranes (%dx%d mesh, 2 tris/quad)\n" % (len(tris), nx, ny))
    for i, (g1, g2, g3) in enumerate(tris, start=1):
        lines.append(_card(["CTRMEM", str(i), "1", str(g1), str(g2), str(g3)]))
    lines.append(_card(["PTRMEM", "1", "1", _fnum(t), "0.0"]))

    lines.append("$ Material: linear-elastic base (MAT1) plus a bilinear stress-strain\n")
    lines.append("$ curve (MATS1 -> TABLES1) that makes it nonlinear past yield\n")
    lines.append(_card(["MAT1", "1", _fnum(E), "", _fnum(nu), _fnum(rho)]))
    lines.append(_card(["MATS1", "1", "101"]))
    pairs = [
        (-max_strain, -max_stress), (-yield_strain, -yield_stress),
        (0.0, 0.0),
        (yield_strain, yield_stress), (max_strain, max_stress),
    ]
    lines.extend(_tables1_cards(101, pairs))

    lines.append("$ Constraints (GRDSET already fixes T3 + all rotations everywhere;\n")
    lines.append("$ this fixes the in-plane translations at the support edge(s))\n")
    fix_groups = [left_edge] if bc == "cantilever" else [left_edge, right_edge]
    for grp in fix_groups:
        rest = list(grp)
        chunk = rest[:6]
        rest = rest[6:]
        lines.append(_card(["SPC1", "10", "12"] + [str(g) for g in chunk]))
        while rest:
            lines.append(_card(["SPC1", "10", "12"] + [str(g) for g in rest[:7]]))
            rest = rest[7:]

    lines.append("$ In-plane point loads distributed over the load edge\n")
    comp_vec = {"x": (1, 0, 0), "y": (0, 1, 0)}[load_dir]
    per_node = load_n / len(target_edge)
    for gid in target_edge:
        lines.append(_card(["FORCE", "20", str(gid), "0", _fnum(per_node),
                             _fnum(comp_vec[0]), _fnum(comp_vec[1]), _fnum(comp_vec[2])]))

    lines.append("$ Load ramp: fractional multipliers of the base FORCE load\n")
    steps = [round((i + 1) / num_load_steps, 4) for i in range(num_load_steps)]
    chunk = steps[:6]
    rest = steps[6:]
    plfact_cont = "+PL1" if rest else None
    lines.append(_card(["PLFACT", "30"] + [str(s) for s in chunk], cont_out=plfact_cont))
    i = 1
    while rest:
        this_chunk = rest[:7]
        rest = rest[7:]
        next_cont = "+PL%d" % (i + 1) if rest else None
        lines.append(_card(["+PL%d" % i] + [str(s) for s in this_chunk], cont_out=next_cont))
        i += 1

    lines.append("ENDDATA\n")

    meta = dict(
        nodes=nodes, elements=tris, fixed_ids=fixed_ids, free_ids=sorted(free_ids),
        n_mass_dof=n_mass_dof, nx=nx, ny=ny, L=L, W=W, t=t, bc=bc, analysis="nonlinear",
        target_edge=target_edge, monitor_node=monitor_node,
        yield_stress_mpa=yield_stress, tangent_modulus_mpa=Et, max_strain=max_strain,
        num_load_steps=num_load_steps, load_n=load_n, load_dir=load_dir,
    )
    return "".join(lines), meta


def build_deck(spec):
    """
    spec: dict with keys
      length_mm, width_mm, thickness_mm (float)
      nx, ny (int, mesh divisions along length / width, >=1)
      material: name in MATERIAL_PRESETS, or dict(E=,nu=,rho=)
      bc: "cantilever" | "simply_supported" | "fixed_fixed"
      analysis: "modes" | "static" | "freq_response" | "nonlinear"
      num_modes (int, only for modes)
      freq_max_hz (float, only for modes, default 2e5)
      load_n (float, for static/freq_response/nonlinear, total load magnitude in Newtons)
      load_dir: "z" | "y" | "x" (for static/freq_response, default "z";
                for nonlinear must be "x" or "y" -- see note below)
      freq_start_hz, freq_end_hz (float, only for freq_response)
      num_freq_points (int, only for freq_response, default 41)
      damping_g (float, only for freq_response, structural damping coeff, default 0.02)
      yield_stress_mpa (float, only for nonlinear, default 250.0)
      tangent_modulus_mpa (float, only for nonlinear, post-yield tangent
                            modulus, default E/100)
      max_strain (float, only for nonlinear, curve extent, default 0.05)
      num_load_steps (int, only for nonlinear, default 4)
    Returns (deck_text, meta) where meta has grid/element bookkeeping
    the caller needs to interpret the .f06 output (node ids, coords,
    fixed node ids, free node ids, element list).

    Nonlinear analysis (SOL 6, piecewise-linear stress-dependent material
    via MATS1/TABLES1, incrementally loaded via PLFACT) uses CTRMEM/PTRMEM
    (triangular membrane) elements instead of CQUAD4/PSHELL, because this
    NASTRAN-95 build's SOL 6 DMAP path was found empirically to reject
    CQUAD4 with cryptic MATS1/PLFACT card-format errors while CTRMEM works
    cleanly with an identical material/load setup. CTRMEM has no bending or
    out-of-plane stiffness, so load_dir must be in-plane ("x" or "y"), and
    every node's T3,R1,R2,R3 are globally fixed via GRDSET (there's no
    stiffness there to solve for).
    """
    L = float(spec["length_mm"])
    W = float(spec["width_mm"])
    t = float(spec["thickness_mm"])
    nx = max(1, int(spec.get("nx", 4)))
    ny = max(1, int(spec.get("ny", 2)))
    bc = spec.get("bc", "cantilever")
    analysis = spec.get("analysis", "modes")

    mat = spec.get("material", "steel")
    if isinstance(mat, str):
        mat = MATERIAL_PRESETS[mat]
    E, nu, rho = float(mat["E"]), float(mat["nu"]), float(mat["rho"])

    nnx, nny = nx + 1, ny + 1
    nodes = {}
    for ix in range(nnx):
        for iy in range(nny):
            gid = grid_id(ix, iy, ny)
            nodes[gid] = (ix * L / nx, iy * W / ny, 0.0)

    elements = []
    for ix in range(nx):
        for iy in range(ny):
            g1 = grid_id(ix, iy, ny)
            g2 = grid_id(ix + 1, iy, ny)
            g3 = grid_id(ix + 1, iy + 1, ny)
            g4 = grid_id(ix, iy + 1, ny)
            elements.append((g1, g2, g3, g4))

    left_edge = [grid_id(0, iy, ny) for iy in range(nny)]
    right_edge = [grid_id(nx, iy, ny) for iy in range(nny)]

    if bc == "cantilever":
        fixed_full = left_edge
        fixed_partial = []  # (grid_ids, components)
    elif bc == "simply_supported":
        fixed_full = []
        fixed_partial = [(left_edge, "123"), (right_edge, "123")]
    elif bc == "fixed_fixed":
        fixed_full = left_edge + right_edge
        fixed_partial = []
    else:
        raise ValueError("unknown bc: %s" % bc)

    fixed_ids = set(fixed_full) | {g for grp, _ in fixed_partial for g in grp}
    free_ids = [g for g in nodes if g not in fixed_ids]
    n_mass_dof = 3 * len(free_ids)  # T1,T2,T3 carry mass; R1,R2 auto-omitted; R3 globally fixed

    # load/monitor edge: the free end for a cantilever, mid-span otherwise
    target_edge = right_edge if bc == "cantilever" else [grid_id(nx // 2, iy, ny) for iy in range(nny)]
    monitor_node = target_edge[len(target_edge) // 2]

    if analysis == "nonlinear":
        return _build_nonlinear_deck(spec, L, W, t, nx, ny, bc, E, nu, rho, nodes, elements,
                                      left_edge, right_edge, target_edge, monitor_node, n_mass_dof)

    lines = []
    title = "PARAMETRIC SHELL %.0fx%.0fx%.1fMM, %dx%d MESH, %s" % (L, W, t, nx, ny, bc.upper())
    lines.append(("ID    PARAM95,NASTRAN").ljust(80) + "\n")
    lines.append("APP   DISPLACEMENT".ljust(80) + "\n")

    if analysis == "modes":
        lines.append("SOL   3,1".ljust(80) + "\n")
    elif analysis == "static":
        lines.append("SOL   1".ljust(80) + "\n")
    elif analysis == "freq_response":
        lines.append("SOL   8".ljust(80) + "\n")
    else:
        raise ValueError("unknown analysis: %s" % analysis)

    lines.append("TIME  15".ljust(80) + "\n")
    lines.append("CEND\n")
    lines.append("TITLE    = %s\n" % title[:72])
    lines.append("SPC   = 10\n")

    if analysis == "modes":
        lines.append("DISP  = ALL\n")
        lines.append("METHOD= 5\n")
    elif analysis == "static":
        lines.append("DISP  = ALL\n")
        lines.append("ELSTRESS = ALL\n")
        lines.append("SUBCASE 1\n")
        lines.append("LOAD  = 20\n")
    elif analysis == "freq_response":
        # restrict output to the monitor node to keep the .f06 small and
        # parsing simple -- SORT2/PHASE prints one block per point spanning
        # every frequency, which is what the parser expects.
        lines.append("SET 1 = %d\n" % monitor_node)
        lines.append("DISPLACEMENT(SORT2,PHASE) = 1\n")
        lines.append("DLOAD = 20\n")
        lines.append("FREQUENCY = 30\n")
    lines.append("BEGIN BULK\n")

    lines.append("$ Global default: constrain drilling DOF (RZ) on every grid -- a flat\n")
    lines.append("$ shell mesh has no stiffness there.\n")
    lines.append(_card(["GRDSET", "", "", "", "", "", "", "6"]))

    lines.append("$ Grid points\n")
    for gid in sorted(nodes):
        x, y, z = nodes[gid]
        lines.append(_card(["GRID", str(gid), "", _fnum(x), _fnum(y), _fnum(z)]))

    lines.append("$ Elements: %d CQUAD4 shells (%dx%d mesh)\n" % (len(elements), nx, ny))
    for i, (g1, g2, g3, g4) in enumerate(elements, start=1):
        lines.append(_card(["CQUAD4", str(i), "1", str(g1), str(g2), str(g3), str(g4)]))

    lines.append("$ Property: PSHELL, uniform thickness, isotropic material (MID=1)\n")
    lines.append(_card(["PSHELL", "1", "1", _fnum(t), "1", "1.0", "1"]))

    lines.append("$ Material\n")
    lines.append(_card(["MAT1", "1", _fnum(E), "", _fnum(nu), _fnum(rho)]))

    lines.append("$ Constraints\n")
    if fixed_full:
        lines.append(_card(["SPC1", "10", "123456"] + [str(g) for g in fixed_full[:5]]))
        rest = fixed_full[5:]
        while rest:
            lines.append(_card(["SPC1", "10", "123456"] + [str(g) for g in rest[:7]]))
            rest = rest[7:]
    for grp, comp in fixed_partial:
        chunk = grp[:6]
        rest = grp[6:]
        lines.append(_card(["SPC1", "10", comp] + [str(g) for g in chunk]))
        while rest:
            lines.append(_card(["SPC1", "10", comp] + [str(g) for g in rest[:7]]))
            rest = rest[7:]

    if analysis == "modes":
        num_modes = min(int(spec.get("num_modes", 8)), n_mass_dof)
        freq_max = float(spec.get("freq_max_hz", 2.0e5))
        # NE/ND must be a real upper bound on the number of roots in [F1,F2],
        # not just how many the caller wants -- an under-estimate makes the
        # solver silently converge on the WRONG (too-high) subset of roots
        # instead of the true lowest ones. Extract generously, then the
        # caller truncates to num_modes after sorting by frequency.
        extract_count = n_mass_dof
        lines.append("$ Real eigenvalue extraction: inverse power method. NE/ND is deliberately\n")
        lines.append("$ larger than num_modes requested -- an under-estimated root count makes\n")
        lines.append("$ INV silently skip the true low-frequency roots. Caller truncates after.\n")
        lines.append(_card(["EIGR", "5", "INV", "0.1", _fnum(freq_max), str(extract_count), str(extract_count)], cont_out="+EG5"))
        lines.append(_card(["+EG5", "MAX"]))
    elif analysis == "static":
        load_n = float(spec.get("load_n", 100.0))
        load_dir = spec.get("load_dir", "z")
        comp_vec = {"z": (0, 0, 1), "y": (0, 1, 0), "x": (1, 0, 0)}[load_dir]
        per_node = load_n / len(target_edge)
        lines.append("$ Static point loads distributed over the load edge\n")
        for gid in target_edge:
            lines.append(_card(["FORCE", "20", str(gid), "0", _fnum(per_node),
                                 _fnum(comp_vec[0]), _fnum(comp_vec[1]), _fnum(comp_vec[2])]))

    else:  # freq_response
        load_n = float(spec.get("load_n", 50.0))
        load_dir = spec.get("load_dir", "z")
        comp_idx = {"x": 1, "y": 2, "z": 3}[load_dir]
        f_start = float(spec.get("freq_start_hz", 50.0))
        f_end = float(spec.get("freq_end_hz", 300.0))
        n_pts = max(2, int(spec.get("num_freq_points", 41)))
        damping_g = float(spec.get("damping_g", 0.02))
        per_node = load_n / len(target_edge)

        lines.append("$ Frequency sweep: f_start, increment, number of increments\n")
        df = (f_end - f_start) / (n_pts - 1)
        lines.append(_card(["FREQ1", "30", _fnum(f_start), _fnum(df), str(n_pts - 1)]))

        lines.append("$ Harmonic point loads distributed over the load edge (unit magnitude\n")
        lines.append("$ vs. frequency, via TABLED1 50; RLOAD1 combines DAREA 40 x TABLED1 50)\n")
        for gid in target_edge:
            lines.append(_card(["DAREA", "40", str(gid), str(comp_idx), _fnum(per_node)]))
        lines.append(_card(["RLOAD1", "20", "40", "", "", "50"]))
        lines.append(_card(["TABLED1", "50"], cont_out="+TB1"))
        lines.append(_card(["+TB1", _fnum(f_start), "1.0", _fnum(f_end), "1.0", "ENDT"]))

        lines.append("$ Uniform structural damping (direct formulation uses G directly,\n")
        lines.append("$ no W3/W4 conversion needed for SOL 8)\n")
        lines.append(_card(["PARAM", "G", _fnum(damping_g)]))

    lines.append("ENDDATA\n")

    meta = dict(
        nodes=nodes, elements=elements, fixed_ids=sorted(fixed_ids),
        free_ids=sorted(free_ids), n_mass_dof=n_mass_dof,
        nx=nx, ny=ny, L=L, W=W, t=t, bc=bc, analysis=analysis,
        target_edge=target_edge, monitor_node=monitor_node,
    )
    return "".join(lines), meta


def build_deck_from_tetmesh(spec):
    """
    Parametric-shell counterpart for uploaded "medium complexity" geometry:
    build a CTETRA solid-element deck from an arbitrary tet mesh (from
    tetmesh.mesh_stl_bytes) plus user-picked fixed/loaded node sets (from
    3D click-selection in the UI, not a bounding-box heuristic).

    spec keys:
      nodes: {node_id(int): (x,y,z)}
      tets: [(n1,n2,n3,n4), ...]
      fixed_ids: [node_id, ...]           -- SPC1 123, all translations
      loaded_ids: [node_id, ...]          -- FORCE/DAREA split across these
      material: name in MATERIAL_PRESETS, or dict(E=,nu=,rho=)
      analysis: "modes" | "static" | "freq_response"
      (other fields identical in meaning to build_deck's spec)

    Lesson learned specific to solids: CTETRA provides zero rotational
    stiffness at every node (it's a pure translational 3D solid element,
    unlike the shell's single drilling-DOF gap) -- GRDSET must constrain
    components 4,5,6 globally, not just component 6.
    """
    nodes = {int(k): tuple(v) for k, v in spec["nodes"].items()}
    tets = [tuple(int(x) for x in t) for t in spec["tets"]]
    fixed_ids = sorted(set(int(x) for x in spec["fixed_ids"]))
    loaded_ids = sorted(set(int(x) for x in spec["loaded_ids"]))
    if not fixed_ids:
        raise ValueError("no fixed nodes selected")
    if not loaded_ids:
        raise ValueError("no loaded nodes selected")

    analysis = spec.get("analysis", "modes")
    mat = spec.get("material", "steel")
    if isinstance(mat, str):
        mat = MATERIAL_PRESETS[mat]
    E, nu, rho = float(mat["E"]), float(mat["nu"]), float(mat["rho"])

    free_ids = [g for g in nodes if g not in fixed_ids]
    n_mass_dof = 3 * len(free_ids)  # T1,T2,T3 only -- CTETRA has no rotational stiffness at all
    monitor_node = loaded_ids[0]

    lines = []
    lines.append("ID    PARAM95,NASTRAN".ljust(80) + "\n")
    lines.append("APP   DISPLACEMENT".ljust(80) + "\n")
    if analysis == "modes":
        lines.append("SOL   3,1".ljust(80) + "\n")
    elif analysis == "static":
        lines.append("SOL   1".ljust(80) + "\n")
    elif analysis == "freq_response":
        lines.append("SOL   8".ljust(80) + "\n")
    else:
        raise ValueError("unknown analysis: %s" % analysis)
    lines.append("TIME  30".ljust(80) + "\n")
    lines.append("CEND\n")
    lines.append("TITLE    = UPLOADED GEOMETRY, %d TETS, %d NODES\n" % (len(tets), len(nodes)))
    lines.append("SPC   = 10\n")

    if analysis == "modes":
        lines.append("DISP  = ALL\n")
        lines.append("METHOD= 5\n")
    elif analysis == "static":
        lines.append("DISP  = ALL\n")
        lines.append("SUBCASE 1\n")
        lines.append("LOAD  = 20\n")
    else:
        lines.append("SET 1 = %d\n" % monitor_node)
        lines.append("DISPLACEMENT(SORT2,PHASE) = 1\n")
        lines.append("DLOAD = 20\n")
        lines.append("FREQUENCY = 30\n")
    lines.append("BEGIN BULK\n")

    lines.append("$ CTETRA has zero rotational stiffness at every node (pure solid\n")
    lines.append("$ element) -- constrain components 4,5,6 globally or the stiffness\n")
    lines.append("$ matrix is singular.\n")
    lines.append(_card(["GRDSET", "", "", "", "", "", "", "456"]))

    lines.append("$ Grid points (from uploaded/meshed geometry)\n")
    for gid in sorted(nodes):
        x, y, z = nodes[gid]
        lines.append(_card(["GRID", str(gid), "", _fnum(x), _fnum(y), _fnum(z)]))

    lines.append("$ Elements: %d CTETRA solids\n" % len(tets))
    for i, (g1, g2, g3, g4) in enumerate(tets, start=1):
        lines.append(_card(["CTETRA", str(i), "1", str(g1), str(g2), str(g3), str(g4)]))

    lines.append("$ Material (CTETRA references MAT1 directly, no PSOLID in this build)\n")
    lines.append(_card(["MAT1", "1", _fnum(E), "", _fnum(nu), _fnum(rho)]))

    lines.append("$ Constraints: fully fix translations at user-picked support nodes\n")
    rest = fixed_ids
    while rest:
        lines.append(_card(["SPC1", "10", "123"] + [str(g) for g in rest[:7]]))
        rest = rest[7:]

    if analysis == "modes":
        num_modes = min(int(spec.get("num_modes", 8)), n_mass_dof)
        freq_max = float(spec.get("freq_max_hz", 2.0e5))
        extract_count = n_mass_dof
        lines.append("$ Real eigenvalue extraction: inverse power, NE/ND = full mass-DOF\n")
        lines.append("$ count (see deck_generator module docstring for why)\n")
        lines.append(_card(["EIGR", "5", "INV", "0.1", _fnum(freq_max), str(extract_count), str(extract_count)], cont_out="+EG5"))
        lines.append(_card(["+EG5", "MAX"]))
    elif analysis == "static":
        load_n = float(spec.get("load_n", 100.0))
        load_dir = spec.get("load_dir", "z")
        comp_vec = {"z": (0, 0, 1), "y": (0, 1, 0), "x": (1, 0, 0)}[load_dir]
        per_node = load_n / len(loaded_ids)
        lines.append("$ Static point loads distributed over the user-picked load nodes\n")
        for gid in loaded_ids:
            lines.append(_card(["FORCE", "20", str(gid), "0", _fnum(per_node),
                                 _fnum(comp_vec[0]), _fnum(comp_vec[1]), _fnum(comp_vec[2])]))
    else:  # freq_response
        load_n = float(spec.get("load_n", 50.0))
        load_dir = spec.get("load_dir", "z")
        comp_idx = {"x": 1, "y": 2, "z": 3}[load_dir]
        f_start = float(spec.get("freq_start_hz", 50.0))
        f_end = float(spec.get("freq_end_hz", 300.0))
        n_pts = max(2, int(spec.get("num_freq_points", 41)))
        damping_g = float(spec.get("damping_g", 0.02))
        per_node = load_n / len(loaded_ids)

        df = (f_end - f_start) / (n_pts - 1)
        lines.append(_card(["FREQ1", "30", _fnum(f_start), _fnum(df), str(n_pts - 1)]))
        for gid in loaded_ids:
            lines.append(_card(["DAREA", "40", str(gid), str(comp_idx), _fnum(per_node)]))
        lines.append(_card(["RLOAD1", "20", "40", "", "", "50"]))
        lines.append(_card(["TABLED1", "50"], cont_out="+TB1"))
        lines.append(_card(["+TB1", _fnum(f_start), "1.0", _fnum(f_end), "1.0", "ENDT"]))
        lines.append(_card(["PARAM", "G", _fnum(damping_g)]))

    lines.append("ENDDATA\n")

    meta = dict(
        nodes=nodes, tets=tets, fixed_ids=fixed_ids, loaded_ids=loaded_ids,
        free_ids=sorted(free_ids), n_mass_dof=n_mass_dof, monitor_node=monitor_node,
        analysis=analysis,
    )
    return "".join(lines), meta
