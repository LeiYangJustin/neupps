import numpy as np
import trimesh


def ccw_boundary(mesh, b_edges):

    vids = [e[0] for e in b_edges]
    center = np.mean(mesh.vertices[vids],axis=0)
    v0 = mesh.vertices[b_edges[0][0]] - center
    v1 = mesh.vertices[b_edges[0][1]] - mesh.vertices[b_edges[0][0]]
    cp = np.cross(v0, v1)
    all_edges = mesh.edges 
    for eid, e in enumerate(all_edges):
        if b_edges[0][0] == e[0] and b_edges[0][1] == e[1]:
            fid = mesh.edges_face[eid]
            normal = mesh.face_normals[fid]
            return np.sum(cp * normal) > 0


def get_border_edges(patch_mesh, traverse_sorted=True, has_multiple_connected_comp=False):
    return get_border_edges_with_faces(patch_mesh.faces, traverse_sorted, has_multiple_connected_comp)

def get_border_edges_with_faces(faces, traverse_sorted=True, has_multiple_connected_comp=False):
    edges = trimesh.geometry.faces_to_edges(faces)
    unique_edge_ids = trimesh.grouping.group_rows(np.sort(edges, axis=1), require_count=1)
    unique_edges = edges[unique_edge_ids]
    
    if len(unique_edges) == 0:
        print("unique_edges", len(unique_edges))
        return None

    boundary_curves = [[]] ## work like a pointer
    if traverse_sorted:
        unique_edges = unique_edges.tolist()
        sorted_edges = boundary_curves[-1] ## work like a pointer
        sorted_edges.append(unique_edges[0])
        unique_edges.remove(unique_edges[0])
        while len(unique_edges) > 0:
            found = False
            for e in unique_edges:
                if e[0] == sorted_edges[-1][1]:
                    sorted_edges.append(e)
                    unique_edges.remove(e)
                    found = True
                    break
            if not found:
                if has_multiple_connected_comp:
                    boundary_curves.append([])
                    sorted_edges = boundary_curves[-1] ## work like a pointer
                    sorted_edges.append(e)
                    unique_edges.remove(e)
                else:
                    # print(edges)
                    # print(len(unique_edges))
                    # print(sorted_edges[-1])
                    print("cannot find any match; may have multiple connected components")
                    # verts = patch_mesh.vertices
                    # vids = [e[0] for e in sorted_edges]
                    # write_obj_file("edges.obj", verts[vids])
                    raise AssertionError
        if has_multiple_connected_comp:
            return boundary_curves
        else:
            return boundary_curves[-1]
    return unique_edges