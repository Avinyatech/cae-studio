"""Single-process reference solve: assemble + apply BCs/loads + CG (or direct) solve."""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .assembly import assemble_global


def solve_static(nodes, tets, E, nu, fixed_node_ids, loads, method="cg", tol=1e-8, maxiter=20000):
    """
    fixed_node_ids: iterable of node row indices, fully clamped (ux=uy=uz=0).
    loads: dict node_row_index -> (fx,fy,fz).
    Returns (u, info): u is (N,3) displacement array, info is solver status
    (0 = converged, for CG; >0 = iterations to converge without hitting tol
    for scipy's cg, matching scipy's own return convention).
    """
    K = assemble_global(nodes, tets, E, nu)
    ndof = nodes.shape[0] * 3

    F = np.zeros(ndof)
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

    K_ff = K[free_idx][:, free_idx].tocsr()
    F_f = F[free_idx]

    if method == "direct":
        u_f = spla.spsolve(K_ff, F_f)
        info = 0
    else:
        precond = sp.diags(1.0 / K_ff.diagonal())
        u_f, info = spla.cg(K_ff, F_f, rtol=tol, maxiter=maxiter, M=precond)

    u = np.zeros(ndof)
    u[free_idx] = u_f
    return u.reshape(-1, 3), info
