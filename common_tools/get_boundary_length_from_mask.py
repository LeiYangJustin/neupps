import argparse
import sys
import os
sys.path.append(os.getcwd())

from common_tools.get_border_edges import get_border_edges, get_border_edges_with_faces
import trimesh 
from common_tools.io_tools import *
import argparse

def get_boundary_length_from_mask(mesh, mask, graph, folder=None):
    node_ids = graph['node_ids']
    cells = graph['cells']
    node_ids = np.array(node_ids).astype(np.int32)

    cell_arc_lengths =[]
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
        boundary_length = []
        for be in boundary_edges[0]:
            boundary_verts.append(be[0])
            boundary_length.append(np.linalg.norm(mesh.vertices[be[0]] - mesh.vertices[be[1]]))

        # print("debug")
        # print(boundary_length)
        # print(len(boundary_verts))

        boundary_verts = np.array(mesh.vertices[boundary_verts])
        cell_nodes = np.array(mesh.vertices[node_ids[cells[id]]])

        dist = np.linalg.norm(cell_nodes[:,None,:] - boundary_verts[None,:,:], axis=2)
        min_dist_ids = np.argmin(dist, axis=1)

        arc_lengths = []
        for i in range(len(min_dist_ids)):
            j = (i+1)%len(min_dist_ids)
            if min_dist_ids[i] > min_dist_ids[j]:
                # print(min_dist_ids[i], min_dist_ids[j])
                # print(boundary_length[min_dist_ids[i]], np.sum(boundary_length[min_dist_ids[i]:]))
                # print(boundary_length[min_dist_ids[j]], np.sum(boundary_length[0:min_dist_ids[j]]))
                arc_length = np.sum(boundary_length[min_dist_ids[i]:])
                arc_length += np.sum(boundary_length[0:min_dist_ids[j]])
            else:
                print(min_dist_ids[i], min_dist_ids[j])
                arc_length = np.sum(boundary_length[min_dist_ids[i]:min_dist_ids[j]])
            arc_lengths.append(arc_length)

        print(cells[id])
        print(min_dist_ids)
        
        # print(id)
        # print("arc_length", arc_lengths)
        # input()

        arc_lengths = np.array(arc_lengths)
        arc_lengths = arc_lengths / np.sum(arc_lengths)
        arc_lengths = np.concatenate([arc_lengths[-1:], arc_lengths[:-1]])
        cell_arc_lengths.append(arc_lengths.tolist())
        
    return cell_arc_lengths



def get_boundary_length_from_mask_nonmanifold(mesh, mask, graph, folder=None):
    nodes = graph['nodes']
    cells = graph['cells']
    nodes = np.array(nodes)

    cell_arc_lengths =[]
    for id, m in enumerate(mask):
        submesh = mesh.submesh([m], only_watertight=False)[0]
        
        boundary_edges = get_border_edges_with_faces(
            submesh.faces, 
            traverse_sorted=True, 
            has_multiple_connected_comp=True)

        ## sort by length if there are more than 1 boundary edges
        if len(boundary_edges) > 1:
            boundary_edges.sort(key=len, reverse=True)

        boundary_verts = []
        boundary_length = []
        for be in boundary_edges[0]:
            boundary_verts.append(be[0])
            boundary_length.append(np.linalg.norm(mesh.vertices[be[0]] - mesh.vertices[be[1]]))

        boundary_verts = np.array(submesh.vertices[boundary_verts])

        pq_submesh = trimesh.proximity.ProximityQuery(submesh)
        _, vids = pq_submesh.vertex(nodes[cells[id]])
        cell_nodes = np.array(submesh.vertices[vids])

        dist = np.linalg.norm(cell_nodes[:,None,:] - boundary_verts[None,:,:], axis=2)
        min_dist_ids = np.argmin(dist, axis=1)

        arc_lengths = []
        for i in range(len(min_dist_ids)):
            j = (i+1)%len(min_dist_ids)
            if min_dist_ids[i] > min_dist_ids[j]:
                arc_length = np.sum(boundary_length[min_dist_ids[i]:-1])
                arc_length += np.sum(boundary_length[0:min_dist_ids[j]])
            else:
                arc_length = np.sum(boundary_length[min_dist_ids[i]:min_dist_ids[j]])
            arc_lengths.append(arc_length)

        print(cells[id])
        print(min_dist_ids)
        
        arc_lengths = np.array(arc_lengths)
        arc_lengths = arc_lengths / np.sum(arc_lengths)
        arc_lengths = np.concatenate([arc_lengths[-1:], arc_lengths[:-1]])
        cell_arc_lengths.append(arc_lengths.tolist())

    return cell_arc_lengths



def parse_args():
    parser = argparse.ArgumentParser(description="Modeling 3D shapes with neural patches")
    parser.add_argument("--model_name",
                        required=True,
                        type=str,
                        help="path to config"
                        )
    
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser('Get boundary length from mask.')
    parser.add_argument('--folder', type=str, default='data/lc_bunny/data')
    args = parser.parse_args()
    
    folder = os.path.join(args.folder, 'data')
    mesh_path = f'{folder}/single/mesh.obj'
    if not os.path.exists(mesh_path):
        mesh_path = f'{folder}/single/mesh.ply'
    mesh = trimesh.load(mesh_path, process=False, maintain_order=True)

    print(len(mesh.vertices), len(mesh.faces))

    mask = read_json(f'{folder}/mask.json')
    
    # ## masks are ids of vertices
    # mask = [np.arange(len(mesh.faces))]
    graph = read_json(f'{folder}/topology_graph.json')

    
    cell_arc_lengths = get_boundary_length_from_mask(mesh, mask, graph, folder)
    write_json(cell_arc_lengths, f'{folder}/cell_arc_lengths.json')

