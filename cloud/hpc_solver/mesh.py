"""
Structured box mesh: (nx+1)x(ny+1)x(nz+1) grid of nodes, each hex cell split
into 6 tets sharing the cell's main diagonal. Deterministic node/element
count from (nx,ny,nz), so it's used both for correctness validation (small
sizes, cross-checked against NASTRAN-95's CTETRA) and for scale/performance
benchmarking (large sizes, far beyond what the compiled solver can hold).
"""
import numpy as np

# 6 tets per hex cell, all sharing the diagonal from local corner 0 to
# corner 6 -- a standard conforming decomposition (no cracks/overlaps when
# tiled across a grid). Local corners: 0=(0,0,0) 1=(1,0,0) 2=(1,1,0) 3=(0,1,0)
# 4=(0,0,1) 5=(1,0,1) 6=(1,1,1) 7=(0,1,1)
_CELL_TETS = [
    (0, 1, 2, 6),
    (0, 2, 3, 6),
    (0, 3, 7, 6),
    (0, 7, 4, 6),
    (0, 4, 5, 6),
    (0, 5, 1, 6),
]


def _node_index(ix, iy, iz, nny, nnz):
    return ix * nny * nnz + iy * nnz + iz


def box_mesh(nx, ny, nz, Lx, Ly, Lz):
    """
    nx,ny,nz: element divisions along each axis (>=1). Lx,Ly,Lz: box extents.
    Returns (nodes, tets): nodes is (N,3) float64 array (0-indexed rows),
    tets is (M,4) int64 array of 0-indexed node rows.
    """
    nnx, nny, nnz = nx + 1, ny + 1, nz + 1
    xs = np.linspace(0.0, Lx, nnx)
    ys = np.linspace(0.0, Ly, nny)
    zs = np.linspace(0.0, Lz, nnz)

    nodes = np.empty((nnx * nny * nnz, 3))
    for ix in range(nnx):
        for iy in range(nny):
            for iz in range(nnz):
                nodes[_node_index(ix, iy, iz, nny, nnz)] = (xs[ix], ys[iy], zs[iz])

    tets = np.empty((nx * ny * nz * 6, 4), dtype=np.int64)
    t = 0
    for ix in range(nx):
        for iy in range(ny):
            for iz in range(nz):
                corners = [
                    _node_index(ix + dx, iy + dy, iz + dz, nny, nnz)
                    for dx, dy, dz in [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
                                        (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)]
                ]
                for a, b, c, d in _CELL_TETS:
                    tets[t] = (corners[a], corners[b], corners[c], corners[d])
                    t += 1
    return nodes, tets


def face_node_ids(nodes, axis, value, tol=1e-9):
    """Node row indices whose `axis` (0=x,1=y,2=z) coordinate is ~= value."""
    return np.nonzero(np.abs(nodes[:, axis] - value) < tol)[0]
