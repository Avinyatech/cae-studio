"""
Vectorized global stiffness assembly: builds every element's 12x12 CST-tet
stiffness matrix in batched numpy operations (no per-element Python loop),
then scatters into a sparse COO matrix. This is what lets assembly stay
practical up to millions of elements -- a pure Python per-element loop would
dominate runtime at that scale.
"""
import numpy as np
import scipy.sparse as sp


def elasticity_matrix(E, nu):
    lam = E * nu / ((1 + nu) * (1 - 2 * nu))
    mu = E / (2 * (1 + nu))
    D = np.zeros((6, 6))
    D[0, 0] = D[1, 1] = D[2, 2] = lam + 2 * mu
    D[0, 1] = D[1, 0] = D[0, 2] = D[2, 0] = D[1, 2] = D[2, 1] = lam
    D[3, 3] = D[4, 4] = D[5, 5] = mu
    return D


def element_stiffness_batch(nodes, tets, E, nu):
    """
    nodes: (N,3), tets: (M,4) int -> Ke_all (M,12,12), vol (M,), dofs (M,12)
    (dofs are global DOF indices, 3 per node, in tet-local node order).
    """
    coords = nodes[tets]                      # (M,4,3)
    M = np.ones((tets.shape[0], 4, 4))
    M[:, :, 1:] = coords
    Minv = np.linalg.inv(M)                   # (M,4,4), batched
    vol = np.abs(np.linalg.det(M)) / 6.0       # (M,)
    grads = Minv[:, 1:, :].transpose(0, 2, 1)  # (M,4,3): grads[m,i]=(bi,ci,di)

    Bs = np.zeros((tets.shape[0], 6, 12))
    for i in range(4):
        c = 3 * i
        bi, ci, di = grads[:, i, 0], grads[:, i, 1], grads[:, i, 2]
        Bs[:, 0, c + 0] = bi
        Bs[:, 1, c + 1] = ci
        Bs[:, 2, c + 2] = di
        Bs[:, 3, c + 0] = ci
        Bs[:, 3, c + 1] = bi
        Bs[:, 4, c + 1] = di
        Bs[:, 4, c + 2] = ci
        Bs[:, 5, c + 0] = di
        Bs[:, 5, c + 2] = bi

    D = elasticity_matrix(E, nu)
    DB = np.einsum('kl,mlj->mkj', D, Bs)               # (M,6,12)
    Ke_all = np.einsum('mik,mkj->mij', Bs.transpose(0, 2, 1), DB)  # (M,12,12)
    Ke_all *= vol[:, None, None]

    dofs = np.empty((tets.shape[0], 12), dtype=np.int64)
    for i in range(4):
        dofs[:, 3 * i + 0] = 3 * tets[:, i]
        dofs[:, 3 * i + 1] = 3 * tets[:, i] + 1
        dofs[:, 3 * i + 2] = 3 * tets[:, i] + 2

    return Ke_all, vol, dofs


def assemble_local_rows(nodes, tets, node_start, node_count, E, nu, free_mask):
    """
    Builds this rank's row-block of the global stiffness matrix: only rows
    for nodes [node_start, node_start+node_count) (this rank's owned DOFs
    after renumber_by_partition), with GLOBAL column indices -- shape
    (3*node_count, 3*N). BCs are folded in directly (see hpc_solver/dcg.py
    docstring on why symmetry must be preserved): fixed-DOF columns are
    dropped everywhere, and this rank's own fixed rows become identity
    rows, both applied here so the returned block is solve-ready.

    Every tet touching ANY node this rank owns gets (redundantly, cheaply)
    recomputed here rather than communicated -- avoids any assembly-time
    MPI traffic at the cost of a little duplicate element-level work at
    partition boundaries.
    """
    ndof = nodes.shape[0] * 3
    row_lo, row_hi = node_start * 3, (node_start + node_count) * 3

    owned_ids = np.arange(node_start, node_start + node_count)
    elem_mask = np.isin(tets, owned_ids).any(axis=1)
    local_tets = tets[elem_mask]

    b_local = np.zeros(node_count * 3)
    if local_tets.shape[0] == 0:
        rows_local = cols_local = vals_local = np.array([], dtype=np.int64)
    else:
        Ke_all, _, dofs = element_stiffness_batch(nodes, local_tets, E, nu)
        M = local_tets.shape[0]
        rows_full = np.broadcast_to(dofs[:, :, None], (M, 12, 12)).reshape(-1)
        cols_full = np.broadcast_to(dofs[:, None, :], (M, 12, 12)).reshape(-1)
        vals_full = Ke_all.reshape(-1)

        owned_row = (rows_full >= row_lo) & (rows_full < row_hi)
        keep = owned_row & free_mask[rows_full] & free_mask[cols_full]
        rows_local = rows_full[keep] - row_lo
        cols_local = cols_full[keep]
        vals_local = vals_full[keep]

    diag_extra_rows = np.nonzero(~free_mask[row_lo:row_hi])[0]  # local row indices that are fixed
    if diag_extra_rows.size:
        rows_local = np.concatenate([rows_local, diag_extra_rows])
        cols_local = np.concatenate([cols_local, diag_extra_rows + row_lo])
        vals_local = np.concatenate([vals_local, np.ones(diag_extra_rows.size)])

    K_local = sp.coo_matrix((vals_local, (rows_local, cols_local)),
                             shape=(node_count * 3, ndof)).tocsr()
    return K_local, b_local


def assemble_global(nodes, tets, E, nu):
    """Full global sparse stiffness matrix, (3N,3N) CSR. Serial/single-rank use."""
    Ke_all, _, dofs = element_stiffness_batch(nodes, tets, E, nu)
    M = tets.shape[0]
    rows = np.broadcast_to(dofs[:, :, None], (M, 12, 12)).reshape(-1)
    cols = np.broadcast_to(dofs[:, None, :], (M, 12, 12)).reshape(-1)
    vals = Ke_all.reshape(-1)
    ndof = nodes.shape[0] * 3
    K = sp.coo_matrix((vals, (rows, cols)), shape=(ndof, ndof)).tocsr()
    return K
