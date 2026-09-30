"""
Modal (normal-modes) analysis: the generalized eigenproblem K*phi = omega^2
* M * phi, solved on the free-DOF subspace after removing fixed BCs the
same way serial.py does for statics. Uses a lumped (diagonal) mass matrix
-- matches how NASTRAN-95 treats CTETRA (translational mass only, no
rotary inertia) and keeps the eigensolver's shift-invert factorization
cheap since M is diagonal.
"""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .assembly import assemble_global, lumped_mass_diag


def modal_analysis(nodes, tets, E, nu, rho, fixed_node_ids, num_modes=6):
    """
    Returns a list of {freq_hz, vectors} dicts, sorted ascending by
    frequency. vectors is (N,3), zero at fixed nodes.
    """
    K = assemble_global(nodes, tets, E, nu)
    mass = lumped_mass_diag(nodes, tets, rho)
    ndof = nodes.shape[0] * 3

    fixed_dofs = set()
    for nid in fixed_node_ids:
        fixed_dofs.update((3 * nid, 3 * nid + 1, 3 * nid + 2))
    free_mask = np.ones(ndof, dtype=bool)
    free_mask[list(fixed_dofs)] = False
    free_idx = np.nonzero(free_mask)[0]

    K_ff = K[free_idx][:, free_idx].tocsc()
    M_ff = sp.diags(mass[free_idx])

    k = min(num_modes, K_ff.shape[0] - 2)
    # shift-invert around sigma=0: with rigid-body DOF removed by the BCs
    # above, K_ff is SPD (no zero eigenvalues), so this is the standard
    # "smallest eigenvalues of an SPD pencil" recipe, not a degenerate case.
    vals, vecs = spla.eigsh(K_ff, k=k, M=M_ff, sigma=0.0, which="LM")

    order = np.argsort(vals)
    vals, vecs = vals[order], vecs[:, order]
    freqs_hz = np.sqrt(np.maximum(vals, 0.0)) / (2 * np.pi)

    modes = []
    for m in range(len(vals)):
        u = np.zeros(ndof)
        u[free_idx] = vecs[:, m]
        modes.append(dict(freq_hz=float(freqs_hz[m]), vectors=u.reshape(-1, 3)))
    return modes
