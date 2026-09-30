"""
Direct frequency response: solves Z(omega)*u = F at each frequency in a
sweep, where Z(omega) = K*(1+i*g) - omega^2*M -- uniform structural
damping via a complex stiffness multiplier, the same formulation the
NASTRAN-95 SOL 8 path in deck_generator.py uses (PARAM,G direct, no
W3/W4 conversion needed). Each point is an independent complex sparse
solve; no factorization is reused across frequencies since Z changes
with omega, but that's fine at the node counts this solver targets.
"""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .assembly import assemble_global, lumped_mass_diag


def frequency_response(nodes, tets, E, nu, rho, fixed_node_ids, loads,
                        freq_start_hz, freq_end_hz, num_points, damping_g, monitor_node_ids):
    K = assemble_global(nodes, tets, E, nu)
    mass = lumped_mass_diag(nodes, tets, rho)
    ndof = nodes.shape[0] * 3

    F = np.zeros(ndof, dtype=complex)
    for nid, (fx, fy, fz) in loads.items():
        F[3 * nid] += fx
        F[3 * nid + 1] += fy
        F[3 * nid + 2] += fz

    fixed_dofs = set()
    for nid in fixed_node_ids:
        fixed_dofs.update((3 * nid, 3 * nid + 1, 3 * nid + 2))
    free_mask = np.ones(ndof, dtype=bool)
    free_mask[list(fixed_dofs)] = False
    free_idx = np.nonzero(free_mask)[0]

    K_ff = K[free_idx][:, free_idx].astype(complex)
    M_ff_diag = mass[free_idx]
    F_f = F[free_idx]
    Kc = K_ff * (1 + 1j * damping_g)

    freqs = np.linspace(freq_start_hz, freq_end_hz, max(2, num_points))
    points_by_node = {int(nid): [] for nid in monitor_node_ids}

    for f_hz in freqs:
        omega = 2 * np.pi * f_hz
        Z = (Kc - (omega ** 2) * sp.diags(M_ff_diag)).tocsc()
        u_f = spla.spsolve(Z, F_f)
        u = np.zeros(ndof, dtype=complex)
        u[free_idx] = u_f
        u = u.reshape(-1, 3)
        for nid in monitor_node_ids:
            mag = np.abs(u[nid]).tolist()
            phase = np.angle(u[nid], deg=True).tolist()
            points_by_node[int(nid)].append(dict(freq_hz=float(f_hz), mag=mag, phase=phase))

    return points_by_node
