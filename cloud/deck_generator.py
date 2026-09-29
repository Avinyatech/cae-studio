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


def build_deck(spec):
    """
    spec: dict with keys
      length_mm, width_mm, thickness_mm (float)
      nx, ny (int, mesh divisions along length / width, >=1)
      material: name in MATERIAL_PRESETS, or dict(E=,nu=,rho=)
      bc: "cantilever" | "simply_supported" | "fixed_fixed"
      analysis: "modes" | "static" | "freq_response"
      num_modes (int, only for modes)
      freq_max_hz (float, only for modes, default 2e5)
      load_n (float, for static/freq_response, total load magnitude in Newtons)
      load_dir: "z" | "y" | "x" (for static/freq_response, default "z")
      freq_start_hz, freq_end_hz (float, only for freq_response)
      num_freq_points (int, only for freq_response, default 41)
      damping_g (float, only for freq_response, structural damping coeff, default 0.02)
    Returns (deck_text, meta) where meta has grid/element bookkeeping
    the caller needs to interpret the .f06 output (node ids, coords,
    fixed node ids, free node ids, element list).
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
