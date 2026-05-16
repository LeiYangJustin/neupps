import sys
import os
sys.path.append(os.getcwd())

from geometry_tools.get_border_edges import get_border_edges, get_border_edges_with_faces
import trimesh 
from common_tools.io_tools import *
import pickle as pkl

def match_mesh_boundaries_to_cells(mesh, graph):
    node_ids = graph['node_ids']
    cells = graph['cells']
    node_ids = np.array(node_ids).astype(np.int32)
    
    boundary_curves = get_border_edges(mesh, traverse_sorted=True, has_multiple_connected_comp=True)
    boundary_verts_list = []
    for i, boundary_curve in enumerate(boundary_curves):
        boundary_verts = []
        for be in boundary_curve:
            boundary_verts.append(be[0])
        # boundary_verts = mesh.vertices[boundary_verts]
        # boundary_verts = np.array(boundary_verts)
        boundary_verts_list.append(boundary_verts)
        # write_obj_file(f"boundary_{i}.obj", boundary_verts)

    for i, cell in enumerate(cells):
        cell_nodes = np.array(mesh.vertices[node_ids[cell]])            
        for j, boundary_verts in enumerate(boundary_verts_list):
            mesh_boundary_verts = np.array(mesh.vertices[boundary_verts])
            dist = np.linalg.norm(cell_nodes[:,None,:] - mesh_boundary_verts[None,:,:], axis=2)
            min_dists = np.min(dist, axis=1)
            print(i, j, min_dists)
            ## find the index of min_dist that is less than 0.00001
            min_dist_ids = np.where(min_dists < 0.00001)[0]

    return




"""
assume mesh is not a manifold mesh
mask is a list of face lists that correspond to the patches
graph is a dictionary that contains the node ids and cells
"""
def match_segmentation_boudanry_to_cell_nonmanifold(mesh, graph, mask):
    nodes = graph['nodes']
    cells = graph['cells']
    nodes = np.array(nodes)

    pq_mesh = trimesh.proximity.ProximityQuery(mesh)

    ## for each patch
    patch_arcs = []
    for id, m in enumerate(mask):
        # faces = mesh.faces[m]
        submesh = mesh.submesh([m], only_watertight=False)[0]

        boundary_edges = get_border_edges_with_faces(
            submesh.faces, 
            traverse_sorted=True, 
            has_multiple_connected_comp=True)

        ## sort by length if there are more than 1 boundary edges
        if len(boundary_edges) > 1:
            boundary_edges.sort(key=len, reverse=True)

        boundary_verts = []
        for be in boundary_edges[0]:
            boundary_verts.append(be[0])
        boundary_verts.append(boundary_edges[0][-1][1])

        submesh_boundary_verts = np.array(submesh.vertices[boundary_verts])
        _, boundary_verts = pq_mesh.vertex(submesh_boundary_verts)
        mesh_boundary_verts = np.array(mesh.vertices[boundary_verts])

        pq_submesh = trimesh.proximity.ProximityQuery(submesh)
        _, vids = pq_submesh.vertex(nodes[cells[id]])
        cell_nodes = np.array(submesh.vertices[vids])
        
        # mesh_boundary_verts = np.array(mesh.vertices[boundary_verts])
        # cell_nodes = np.array(mesh.vertices[node_ids[cells[id]]])

        dist = np.linalg.norm(cell_nodes[:,None,:] - mesh_boundary_verts[None,:,:], axis=2)
        min_dist_ids = np.argmin(dist, axis=1)

        # chain of vertices
        arcs = []
        for i in range(len(min_dist_ids)):
            j = (i+1)%len(min_dist_ids)
            if min_dist_ids[i] > min_dist_ids[j]:
                arc = boundary_verts[min_dist_ids[i]:-1]
                arc = np.concatenate([arc, boundary_verts[0:min_dist_ids[j]+1]]).astype(int).tolist()
            else:
                arc = boundary_verts[min_dist_ids[i]:min_dist_ids[j]+1] ## +1 do not miss the last one
            
            arcs.append(arc)
        
        patch_arcs.append(arcs)
    return patch_arcs


"""
assume mesh is a manifold mesh
mask is a list of face lists that correspond to the patches
graph is a dictionary that contains the node ids and cells
"""
def match_segmentation_boudanry_to_cell(mesh, graph, mask):
    node_ids = graph['node_ids']
    cells = graph['cells']
    node_ids = np.array(node_ids).astype(np.int32)

    ## for each patch
    patch_arcs = []
    for id, m in enumerate(mask):
        faces = mesh.faces[m]
        boundary_edges = get_border_edges_with_faces(
            faces, 
            traverse_sorted=True, 
            has_multiple_connected_comp=True)

        ## sort by length if there are more than 1 boundary edges
        if len(boundary_edges) > 1:
            boundary_edges.sort(key=len, reverse=True)

        boundary_verts = []
        for be in boundary_edges[0]:
            boundary_verts.append(be[0])
        boundary_verts.append(boundary_edges[0][-1][1])

        mesh_boundary_verts = np.array(mesh.vertices[boundary_verts])
        cell_nodes = np.array(mesh.vertices[node_ids[cells[id]]])

        dist = np.linalg.norm(cell_nodes[:,None,:] - mesh_boundary_verts[None,:,:], axis=2)
        min_dist_ids = np.argmin(dist, axis=1)

        # chain of vertices
        arcs = []
        for i in range(len(min_dist_ids)):
            j = (i+1)%len(min_dist_ids)
            if min_dist_ids[i] > min_dist_ids[j]:
                arc = boundary_verts[min_dist_ids[i]:-1]
                arc = np.concatenate([arc, boundary_verts[0:min_dist_ids[j]+1]]).astype(int).tolist()
            else:
                arc = boundary_verts[min_dist_ids[i]:min_dist_ids[j]+1] ## +1 do not miss the last one
            
            arcs.append(arc)
            # write_obj_file(f"arc_{id}_{i}.obj", mesh.vertices[arc])
        
        patch_arcs.append(arcs)
    return patch_arcs


if __name__ == "__main__":

    folder = 'ph_toothpaste'
    fname = 'mesh.obj'
    graph = read_json(f'data_new/{folder}/topology_graph.json')
    mask = read_json(f'data_new/{folder}/mask.json')

    mesh = trimesh.load(
        f'data_new/{folder}/single/{fname}', process=False, maintain_order=True)

    patch_arcs = match_segmentation_boudanry_to_cell(mesh, graph, mask)
    # with open(f'data_new/{folder}/patch_arcs.pkl', 'wb') as f:
    #     pkl.dump(patch_arcs, f)
    write_obj_file("nodes.obj", mesh.vertices[graph['node_ids']])


    print("\n\npatchessssssssss")
    # with open(f'data_processed/{folder}/patch_arcs.pkl', 'rb') as f:
    #     patch_arcs = pkl.load(f)
    for j, arcs in enumerate(patch_arcs):
        for i in range(len(arcs)):
            write_obj_file(f"arc_{j}_{i}.obj", mesh.vertices[arcs[i]])


    