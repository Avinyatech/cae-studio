"""
STL -> tetrahedral volume mesh, via Gmsh. Used for "medium complexity"
uploaded geometry, as opposed to the parametric rectangular-strip meshes
in deck_generator.py.

Gmsh's own OpenCASCADE-free STL workflow: merge the STL, classify its
triangles into surface patches (needed because a raw STL is just a soup
of triangles with no surface/curve topology), build a discrete geometry
from that classification, then mesh a volume bounded by it.
"""
import math
import tempfile
import os

import gmsh

TET_ELEM_TYPE = 4    # gmsh element type id: 4-node tetrahedron
TRI_ELEM_TYPE = 2    # gmsh element type id: 3-node triangle

_lock_active = False


class MeshError(Exception):
    pass


def mesh_stl_bytes(stl_bytes, mesh_size_max=None, angle_deg=40):
    """
    Tet-mesh an uploaded STL file's raw bytes.

    Returns dict:
      nodes: {node_id: (x, y, z)}
      tets: [(n1, n2, n3, n4), ...]          -- CTETRA connectivity
      boundary_faces: [(n1, n2, n3), ...]    -- outer skin, for 3D preview
      bbox: {xmin, xmax, ymin, ymax, zmin, zmax}
    Raises MeshError with a plain-English reason on failure (e.g. a
    non-manifold or leaky STL that Gmsh can't close into a volume).
    """
    global _lock_active
    if _lock_active:
        raise MeshError("a mesh is already being generated; try again shortly")
    _lock_active = True

    fd, stl_path = tempfile.mkstemp(suffix=".stl")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(stl_bytes)

        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)
        try:
            gmsh.model.add("uploaded")
            gmsh.merge(stl_path)

            n_tri_raw = len(gmsh.model.getEntities(2))
            if n_tri_raw == 0:
                raise MeshError("the STL file contains no readable triangles")

            gmsh.model.mesh.classifySurfaces(angle_deg * math.pi / 180.0, True, True)
            gmsh.model.mesh.createGeometry()

            surfaces = gmsh.model.getEntities(2)
            if not surfaces:
                raise MeshError("could not classify any surfaces from this STL")

            try:
                surface_loop = gmsh.model.geo.addSurfaceLoop([e[1] for e in surfaces])
                gmsh.model.geo.addVolume([surface_loop])
            except Exception as e:
                raise MeshError(
                    "could not close this surface into a solid volume -- the STL is "
                    "likely non-manifold or has gaps/holes (common with quick CAD "
                    "exports). Try re-exporting with a smaller tolerance or repairing "
                    "the mesh first. (%s)" % e
                )
            gmsh.model.geo.synchronize()

            if mesh_size_max:
                gmsh.option.setNumber("Mesh.MeshSizeMax", float(mesh_size_max))

            try:
                gmsh.model.mesh.generate(3)
            except Exception as e:
                raise MeshError("volume meshing failed: %s" % e)

            node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
            if len(node_tags) == 0:
                raise MeshError("meshing produced no nodes")
            nodes = {}
            for i, tag in enumerate(node_tags):
                nodes[int(tag)] = (
                    node_coords[3 * i], node_coords[3 * i + 1], node_coords[3 * i + 2]
                )

            elem_types, elem_tags, elem_node_tags = gmsh.model.mesh.getElements(3)
            tets = []
            for et, tags, ents in zip(elem_types, elem_tags, elem_node_tags):
                if et == TET_ELEM_TYPE:
                    for i in range(len(tags)):
                        tets.append(tuple(int(ents[4 * i + k]) for k in range(4)))
            if not tets:
                raise MeshError("volume meshing produced zero tetrahedra")

            btypes, btags, bnode_tags = gmsh.model.mesh.getElements(2)
            boundary_faces = []
            for et, tags, ents in zip(btypes, btags, bnode_tags):
                if et == TRI_ELEM_TYPE:
                    for i in range(len(tags)):
                        boundary_faces.append(tuple(int(ents[3 * i + k]) for k in range(3)))

            xs = [c[0] for c in nodes.values()]
            ys = [c[1] for c in nodes.values()]
            zs = [c[2] for c in nodes.values()]
            bbox = dict(xmin=min(xs), xmax=max(xs), ymin=min(ys), ymax=max(ys),
                        zmin=min(zs), zmax=max(zs))

            return dict(nodes=nodes, tets=tets, boundary_faces=boundary_faces, bbox=bbox)
        finally:
            gmsh.finalize()
    finally:
        try:
            os.remove(stl_path)
        except OSError:
            pass
        _lock_active = False
