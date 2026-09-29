from deck_generator import build_deck, _card, _fnum, grid_id, MATERIAL_PRESETS
from solver_runner import run_job

# Hand-built prototype deck reusing the validated 6x3 cantilever mesh geometry
L, W, t, nx, ny = 100.0, 20.0, 2.0, 6, 3
mat = MATERIAL_PRESETS["steel"]
nnx, nny = nx + 1, ny + 1

nodes = {}
for ix in range(nnx):
    for iy in range(nny):
        nodes[grid_id(ix, iy, ny)] = (ix * L / nx, iy * W / ny, 0.0)
elements = []
for ix in range(nx):
    for iy in range(ny):
        g1 = grid_id(ix, iy, ny); g2 = grid_id(ix + 1, iy, ny)
        g3 = grid_id(ix + 1, iy + 1, ny); g4 = grid_id(ix, iy + 1, ny)
        elements.append((g1, g2, g3, g4))
left_edge = [grid_id(0, iy, ny) for iy in range(nny)]
right_edge = [grid_id(nx, iy, ny) for iy in range(nny)]

lines = []
lines.append("ID    FREQTEST,NASTRAN".ljust(80) + "\n")
lines.append("APP   DISPLACEMENT".ljust(80) + "\n")
lines.append("SOL   8".ljust(80) + "\n")
lines.append("TIME  15".ljust(80) + "\n")
lines.append("CEND\n")
lines.append("TITLE    = FREQ RESPONSE PROTOTYPE\n")
lines.append("SPC   = 10\n")
lines.append("DLOAD = 20\n")
lines.append("FREQUENCY= 30\n")
lines.append("DISPLACEMENT(SORT2,PHASE) = ALL\n")
lines.append("BEGIN BULK\n")
lines.append(_card(["GRDSET", "", "", "", "", "", "", "6"]))
for gid in sorted(nodes):
    x, y, z = nodes[gid]
    lines.append(_card(["GRID", str(gid), "", _fnum(x), _fnum(y), _fnum(z)]))
for i, (g1, g2, g3, g4) in enumerate(elements, start=1):
    lines.append(_card(["CQUAD4", str(i), "1", str(g1), str(g2), str(g3), str(g4)]))
lines.append(_card(["PSHELL", "1", "1", _fnum(t), "1", "1.0", "1"]))
lines.append(_card(["MAT1", "1", _fnum(mat["E"]), "", _fnum(mat["nu"]), _fnum(mat["rho"])]))
lines.append(_card(["SPC1", "10", "123456"] + [str(g) for g in left_edge]))

# frequency sweep 50-300 Hz, 51 points, straddling the known mode-1 ~167.6 Hz
lines.append(_card(["FREQ1", "30", "50.0", "5.0", "50"]))

# unit harmonic point load at tip, Z direction, split across tip nodes
per_node = 10.0 / len(right_edge)
for i, gid in enumerate(right_edge, start=1):
    lines.append(_card(["DAREA", "40", str(gid), "3", _fnum(per_node)]))
lines.append(_card(["RLOAD1", "20", "40", "", "", "50"], cont_out=""))
lines.append(_card(["TABLED1", "50"], cont_out="+TB1"))
lines.append(_card(["+TB1", "50.0", "1.0", "300.0", "1.0", "ENDT"]))
lines.append(_card(["PARAM", "G", "0.02"]))
lines.append("ENDDATA\n")

deck = "".join(lines)
print(deck)
f06 = run_job(deck, job_prefix="freqproto")
with open("debug_freq_proto.f06", "w") as fh:
    fh.write(f06)
print("FATAL" in f06.upper())
for ln in f06.splitlines():
    if "FATAL" in ln.upper():
        print(ln)
