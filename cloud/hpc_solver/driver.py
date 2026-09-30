"""
MPI driver for the distributed CST-tet solver. Run with:

    mpiexec -n <P> py -3.11 hpc_solver/driver.py --nx 40 --ny 10 --nz 10

Each rank independently (and redundantly -- see partition.py/assembly.py
docstrings for why this is an acceptable, documented tradeoff) generates
the full mesh and partition, then keeps only its own row-block of the
stiffness matrix and solves via distributed CG. No rank ever materializes
the full global stiffness matrix.
"""
import argparse
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from mpi4py import MPI

from hpc_solver.mesh import box_mesh, face_node_ids
from hpc_solver.partition import rcb_partition, renumber_by_partition
from hpc_solver.assembly import assemble_local_rows
from hpc_solver.dcg import distributed_cg, allgather_vector

MATERIAL_PRESETS = {
    "steel":    dict(E=210000.0, nu=0.3),
    "aluminum": dict(E=69000.0,  nu=0.33),
    "titanium": dict(E=114000.0, nu=0.34),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nx", type=int, default=20)
    ap.add_argument("--ny", type=int, default=6)
    ap.add_argument("--nz", type=int, default=6)
    ap.add_argument("--lx", type=float, default=200.0)
    ap.add_argument("--ly", type=float, default=40.0)
    ap.add_argument("--lz", type=float, default=40.0)
    ap.add_argument("--material", default="steel", choices=list(MATERIAL_PRESETS))
    ap.add_argument("--load-n", type=float, default=5000.0)
    ap.add_argument("--tol", type=float, default=1e-8)
    ap.add_argument("--maxiter", type=int, default=20000)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    comm = MPI.COMM_WORLD
    rank, P = comm.Get_rank(), comm.Get_size()
    mat = MATERIAL_PRESETS[args.material]

    def log(msg):
        if rank == 0:
            print(msg, flush=True)

    t0 = MPI.Wtime()
    nodes, tets = box_mesh(args.nx, args.ny, args.nz, args.lx, args.ly, args.lz)
    t_mesh = MPI.Wtime() - t0
    log("mesh: %d nodes, %d tets (%.2fs)" % (len(nodes), len(tets), t_mesh))

    t0 = MPI.Wtime()
    owner = rcb_partition(nodes, P)
    nodes2, tets2, _, counts, displs = renumber_by_partition(nodes, tets, owner, P)
    t_part = MPI.Wtime() - t0
    ndof = nodes2.shape[0] * 3
    node_start, node_count = displs[rank] // 3, counts[rank] // 3
    local_min = comm.allreduce(node_count, op=MPI.MIN)
    local_max = comm.allreduce(node_count, op=MPI.MAX)
    log("partition: %d ranks, owned nodes/rank min=%d max=%d (%.2fs)" % (P, local_min, local_max, t_part))

    fixed_ids = face_node_ids(nodes2, axis=0, value=0.0)
    loaded_ids = face_node_ids(nodes2, axis=0, value=args.lx)
    free_mask = np.ones(ndof, dtype=bool)
    for nid in fixed_ids:
        free_mask[3 * nid:3 * nid + 3] = False

    t0 = MPI.Wtime()
    K_local, b_local = assemble_local_rows(nodes2, tets2, node_start, node_count,
                                            mat["E"], mat["nu"], free_mask)
    per_node = args.load_n / len(loaded_ids)
    for nid in loaded_ids:
        if node_start <= nid < node_start + node_count and free_mask[3 * nid + 2]:
            b_local[3 * (nid - node_start) + 2] += per_node
    t_asm = MPI.Wtime() - t0
    nnz_total = comm.allreduce(K_local.nnz, op=MPI.SUM)
    log("assembly: %d total nonzeros across ranks (%.2fs)" % (nnz_total, t_asm))

    row_idx = np.arange(node_count * 3)
    col_idx = node_start * 3 + row_idx
    diag_local = np.asarray(K_local[row_idx, col_idx]).flatten()
    diag_local[diag_local == 0] = 1.0

    iters_box = [0]

    def cb(it, relres):
        if it % 50 == 0 or it == 1:
            print("  iter %5d  rel.residual %.3e" % (it, relres), flush=True)
        iters_box[0] = it

    t0 = MPI.Wtime()
    x_local, iters, converged = distributed_cg(
        comm, K_local, b_local, counts, displs, ndof, diag_local,
        tol=args.tol, maxiter=args.maxiter, callback=cb,
    )
    t_solve = MPI.Wtime() - t0
    log("solve: %s in %d iterations (%.2fs)" % ("converged" if converged else "DID NOT CONVERGE", iters, t_solve))

    x_full = allgather_vector(comm, x_local, counts, displs, ndof)
    if rank == 0:
        tip_z = x_full[3 * loaded_ids + 2].mean()
        log("mean tip Z deflection = %.6f mm" % tip_z)
        summary = dict(
            ranks=P, nodes=len(nodes), tets=len(tets), ndof=ndof,
            t_mesh_s=t_mesh, t_partition_s=t_part, t_assembly_s=t_asm, t_solve_s=t_solve,
            iterations=iters, converged=bool(converged), tip_z_mm=float(tip_z),
        )
        print(json.dumps(summary), flush=True)
        if args.out:
            with open(args.out, "w") as f:
                json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
