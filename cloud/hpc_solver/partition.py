"""
Recursive coordinate bisection (RCB): a simple, well-established geometric
domain-decomposition method for well-shaped meshes (used by real HPC
partitioners such as Zoltan when a full graph partitioner like METIS isn't
warranted). Repeatedly splits the node set along its longest bounding-box
axis into two groups sized proportionally to the number of MPI ranks each
side will serve, until every leaf holds exactly one rank's share.
"""
import numpy as np


def rcb_partition(nodes, num_parts):
    """nodes: (N,3) -> owner: (N,) int64 in [0, num_parts)."""
    n = nodes.shape[0]
    owner = np.empty(n, dtype=np.int64)

    def recurse(node_ids, part_start, part_count):
        if part_count == 1 or len(node_ids) <= 1:
            owner[node_ids] = part_start
            return
        coords = nodes[node_ids]
        spans = coords.max(axis=0) - coords.min(axis=0)
        axis = int(np.argmax(spans))
        order = np.argsort(coords[:, axis], kind="stable")
        sorted_ids = node_ids[order]
        left_count = part_count // 2
        split = max(1, min(len(sorted_ids) - 1, round(len(sorted_ids) * left_count / part_count)))
        recurse(sorted_ids[:split], part_start, left_count)
        recurse(sorted_ids[split:], part_start + left_count, part_count - left_count)

    recurse(np.arange(n), 0, num_parts)
    return owner


def renumber_by_partition(nodes, tets, owner, num_parts):
    """
    Reorders nodes so every rank's owned DOFs are contiguous in the global
    numbering (rank 0's nodes first, then rank 1's, ...) -- this is what
    lets each rank's row-block be described by a single (start, count) pair
    instead of an arbitrary scattered index set, which is what makes
    Allgatherv-based vector sync simple and correct.

    Returns (nodes_new, tets_new, counts, displs) where counts[r]/displs[r]
    are DOF (not node) counts/offsets per rank.
    """
    owned_by_rank = [np.nonzero(owner == r)[0] for r in range(num_parts)]
    ordering = np.concatenate(owned_by_rank)  # old node id, in new-id order
    new_id_of_old = np.empty(nodes.shape[0], dtype=np.int64)
    new_id_of_old[ordering] = np.arange(nodes.shape[0])

    nodes_new = nodes[ordering]
    tets_new = new_id_of_old[tets]

    node_counts = np.array([len(x) for x in owned_by_rank], dtype=np.int64)
    counts = node_counts * 3
    displs = np.zeros(num_parts, dtype=np.int64)
    displs[1:] = np.cumsum(counts)[:-1]
    return nodes_new, tets_new, new_id_of_old, counts, displs
