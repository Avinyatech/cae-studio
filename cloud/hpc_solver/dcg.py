"""
Distributed, Jacobi-preconditioned Conjugate Gradient. Every rank owns a
contiguous row-block of the global stiffness matrix (never materialized
in full on any single rank) and only ever computes with that block.

The two MPI primitives this relies on are exactly the ones real distributed
solvers (PETSc's KSP, Trilinos) are built on:
  - Allreduce(SUM) for the two global scalar dot products per iteration.
  - a vector sync (here: Allgatherv) so every rank has the full search-
    direction vector before its *local* sparse matvec.

Allgatherv broadcasts the whole vector to every rank each iteration, which
is simpler and easier to get correct than point-to-point neighbor (halo/
ghost) exchange, at the cost of O(P) communication volume per rank instead
of O(neighbors) -- fine at the rank counts a single workstation can run,
but the thing to replace with neighbor-only halo exchange before scaling
this to a real many-node cluster.
"""
import numpy as np
from mpi4py import MPI


def global_dot(comm, a, b):
    local = float(np.dot(a, b))
    return comm.allreduce(local, op=MPI.SUM)


def global_norm(comm, a):
    return global_dot(comm, a, a) ** 0.5


def allgather_vector(comm, local_vals, counts, displs, ndof):
    full = np.empty(ndof)
    comm.Allgatherv(np.ascontiguousarray(local_vals), [full, counts, displs, MPI.DOUBLE])
    return full


def distributed_cg(comm, K_local, b_local, counts, displs, ndof, precond_diag_local,
                    tol=1e-8, maxiter=20000, callback=None):
    """
    K_local: scipy.sparse CSR, shape (row_count, ndof) -- this rank's owned
             rows only, but with GLOBAL column indices (cheap: sparse
             storage cost depends on nnz, not on ndof).
    b_local: (row_count,) RHS for this rank's owned rows.
    counts/displs: DOF counts/offsets per rank (from renumber_by_partition).
    Returns (x_local, iterations, converged).
    """
    rank = comm.Get_rank()
    row_count = K_local.shape[0]

    x_local = np.zeros(row_count)
    x_full = allgather_vector(comm, x_local, counts, displs, ndof)
    r_local = b_local - K_local @ x_full
    z_local = r_local / precond_diag_local
    p_local = z_local.copy()
    rz_old = global_dot(comm, r_local, z_local)
    b_norm = global_norm(comm, b_local) or 1.0

    for it in range(1, maxiter + 1):
        p_full = allgather_vector(comm, p_local, counts, displs, ndof)
        Ap_local = K_local @ p_full
        pAp = global_dot(comm, p_local, Ap_local)
        if pAp <= 0:
            break
        alpha = rz_old / pAp
        x_local += alpha * p_local
        r_local -= alpha * Ap_local

        r_norm = global_norm(comm, r_local)
        if callback and rank == 0:
            callback(it, r_norm / b_norm)
        if r_norm / b_norm < tol:
            return x_local, it, True

        z_local = r_local / precond_diag_local
        rz_new = global_dot(comm, r_local, z_local)
        beta = rz_new / rz_old
        p_local = z_local + beta * p_local
        rz_old = rz_new

    return x_local, maxiter, False
