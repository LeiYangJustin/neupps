import sys
import os
sys.path.append(os.getcwd())
import shutil

from common_tools.io_tools import *
import trimesh
from common_tools.get_border_edges import get_border_edges, get_border_edges_with_faces

from PIL import Image
import json
from collections import OrderedDict
import matplotlib
import pickle
import argparse

class Complex():

    def __init__(self, base_mesh, mask, savefolder='out') -> None:
        self.base = base_mesh
        self.mask = mask
        self.savefolder = savefolder

    # def init_scaffold_vertices_new(self):
    #     boundary_vert_ids = []
    #     for i, m in enumerate(self.mask):
    #         faces = self.base.faces[m]
    #         has_multiple_connected_comp = False
    #         bedges = get_border_edges(self.base, has_multiple_connected_comp=has_multiple_connected_comp)
    #         if bedges is not None:
    #             if has_multiple_connected_comp:
    #                 bvids = np.unique(np.concatenate(bedges, axis=0).reshape(-1))
    #             else:
    #                 bvids = np.unique(np.array(bedges).reshape(-1))
    #         # vert_ids = faces.reshape(-1)
    #         # unique_vids, counts = np.unique(vert_ids, return_counts=True)
    #         # boundary_vert_ids.extend(unique_vids[counts==1].tolist())
        
    #     print(boundary_vert_ids)
    #     write_obj_file("boundary_verts.obj", mesh.vertices[boundary_vert_ids])

        

    def init_scaffold_vertices(self):
        print("init scaffold vertices")
        # pq_base = trimesh.proximity.ProximityQuery(self.base)
        # ## sample points from each hand patches
        # vert_counts = np.zeros(self.base.vertices.shape[0])
        # for patch_name, patch_mesh in self.dict_patches.items():
        #     face_centers = np.mean(patch_mesh.vertices[patch_mesh.faces], axis=1)
        #     _, _, fids = pq_base.on_surface(face_centers)
        #     vids = np.unique(self.base.faces[fids.tolist()])
        #     vert_counts[vids] += 1
        
        vert_counts = np.zeros(self.base.vertices.shape[0])
        for i, m in enumerate(self.mask):
            faces = self.base.faces[m]
            has_multiple_connected_comp = False
            bedges = get_border_edges_with_faces(faces, has_multiple_connected_comp=has_multiple_connected_comp)
            if bedges is not None:
                if has_multiple_connected_comp:
                    bvids = np.unique(np.concatenate(bedges, axis=0).reshape(-1))
                else:
                    bvids = np.unique(np.array(bedges).reshape(-1))
            vert_counts[bvids] += 1

        ## find boundary edges from hand mesh
        has_multiple_connected_comp = True
        bedges = get_border_edges(self.base, has_multiple_connected_comp=has_multiple_connected_comp)
        if bedges is not None:
            if has_multiple_connected_comp:
                bvids = np.unique(np.concatenate(bedges, axis=0).reshape(-1))
            else:
                bvids = np.unique(np.array(bedges).reshape(-1))
    
            ## add one count to boundary vertices
            vert_counts[bvids] += 1
        else:
            bvids = []

        
        ## find duplicated vertices
        self.scaffold_vids = np.where(vert_counts > 1)[0]
        ## find duplicated vertices with more than 2 counts
        self.scaffold_corner_ids = np.where(vert_counts > 2)[0]
        self.scaffold_corner_ids = self.scaffold_corner_ids.tolist()

        assert len(self.scaffold_corner_ids) > 0

        ## get duplicated vertices' coordinates
        dup_vert_coords = self.base.vertices[self.scaffold_vids]
        write_obj_file(f'./{self.savefolder}/dup_verts.obj', dup_vert_coords)
        dup_vert_coords_larger_than_2 = self.base.vertices[self.scaffold_corner_ids]
        if len(dup_vert_coords_larger_than_2) > 0:
            write_obj_file(f'./{self.savefolder}/dup_verts_larger_than_2.obj', dup_vert_coords_larger_than_2)
        if len(bvids) > 0:
            write_obj_file(
                f'./{self.savefolder}/mesh_boundary_vertices.obj', 
                self.base.vertices[bvids])

        return None


    def arc_dist(self, arc_verts_0, arc_verts_1):
        dist = np.linalg.norm(arc_verts_0[None,:,:] - arc_verts_1[:,None,:], axis=-1) 
        min_dist = np.min(dist, axis=-1) ## scaffold vids cloest to patch_bverts
        return min_dist.mean()


    def create_an_arc(self, new_arc, all_arcs):
        
        arc_exist = -1
        arc_reverse = False
        new_arc_vert_set = new_arc['vertices'].tolist()

        for i, arc in enumerate(all_arcs):
            arc_vert_set = arc['vertices'].tolist()
            if set(new_arc_vert_set).difference(set(arc_vert_set)) == set():
                # print("overlap", )
                arc_exist = i
                if new_arc['corner_ids'] == arc['corner_ids']:
                    # print('arc already exists')
                    arc_exist = i
                    break
                elif new_arc['corner_ids'][::-1] == arc['corner_ids']:
                    # print("overlap", set(new_arc_vert_set).difference(set(arc_vert_set)) == set())
                    # print('arc already exists, but reversed')
                    arc_exist = i
                    arc_reverse = True
                    break
            else:
                if set(new_arc['corner_ids']).difference(set(arc['corner_ids'])) == set():
                    # print('arc need to be split')
                    new_arc['split'] = True
                    arc['split'] = True
        return arc_exist, arc_reverse
    

    def split_arc(self, arc):
        corner_ids = arc['corner_ids']
        mid_idx = len(arc['vertices'])//2
        arc_vids = arc['vertices']
        split_arc_0 = {
            'corner_ids': [corner_ids[0], arc_vids[mid_idx]],
            'vertices': arc_vids[:mid_idx+1],
            'split': False,
        }
        split_arc_1 = {
            'corner_ids': [arc_vids[mid_idx], corner_ids[1]],
            'vertices': arc_vids[mid_idx:],
            'split': False,
        }
        return split_arc_0, split_arc_1
    

    def main_build(self):
        print("main build")
        patch_topology_graph = []
        all_arcs = []
        ## we find the closest points in each patch to dup_vert_coords_larger_than_2



        for patch_name, m in enumerate(self.mask):
            print("patch name", patch_name)
            patch_mesh = self.base.submesh([m], only_watertight=False)[0]

            ## get corner point indices
            patch_bedges = get_border_edges(patch_mesh)            
            ## get the first in bedge list
            patch_bvids = [edge[0] for edge in patch_bedges]
            # patch_bvids = np.unique(patch_bvids)
            patch_bverts = patch_mesh.vertices[patch_bvids]

            ## find the closest points in self.scalfold_vids
            scaffold_vertices = self.base.vertices[self.scaffold_vids]
            dist = np.linalg.norm(scaffold_vertices[None,:,:] - patch_bverts[:,None,:], axis=-1) 
            min_id_to_scaffold = np.argmin(dist, axis=-1) ## scaffold vids cloest to patch_bverts

            patch_scaffold_vids = self.scaffold_vids[min_id_to_scaffold]
            list_patch_scaffold_vids = patch_scaffold_vids.tolist()
            ## find corner ids in patch_scalfold_vids
            crn_ids = []
            for crn_id in self.scaffold_corner_ids: 
                if crn_id in list_patch_scaffold_vids:
                    crn_ids.append(list_patch_scaffold_vids.index(crn_id))
            crn_ids.sort() ## index of list_patch_scaffold_vids
            
            patch_topology = {
                'patch_name': patch_name,
                'arcs': [],
                'arc_reverse': [], ## whether the arc is reversed
            }

            ## loop over the arcs
            for idx in range(len(crn_ids)-1):
                corner_ids = [patch_scaffold_vids[crn_ids[idx]], patch_scaffold_vids[crn_ids[idx+1]]]
                arc = {
                    'corner_ids': corner_ids,
                    'vertices': patch_scaffold_vids[crn_ids[idx]:crn_ids[idx+1]+1],
                    'split': False,
                }
                arc_exist, arc_reverse = self.create_an_arc(arc, all_arcs)
                ## if arc does not exist, create a new arc; arc_reverse is False
                if arc_exist == -1:
                    all_arcs.append(arc)
                    arc_exist = len(all_arcs) - 1

                patch_topology['arcs'].append(arc_exist)
                patch_topology['arc_reverse'].append(arc_reverse)

            ## handling the final
            print(len(patch_scaffold_vids), len(crn_ids), len(self.scaffold_corner_ids))
            corner_ids = [patch_scaffold_vids[crn_ids[-1]], patch_scaffold_vids[crn_ids[0]]]
            vertices = patch_scaffold_vids[crn_ids[-1]:]
            vertices = np.concatenate([vertices, patch_scaffold_vids[:crn_ids[0]+1]])
            arc = {
                'corner_ids': corner_ids,
                'vertices': vertices,
                'split': False,
            }
            arc_exist, arc_reverse = self.create_an_arc(arc, all_arcs)
            if arc_exist == -1:
                all_arcs.append(arc)
                arc_exist = len(all_arcs) - 1
            patch_topology['arcs'].append(arc_exist)
            patch_topology['arc_reverse'].append(arc_reverse)

            ## append
            patch_topology_graph.append(patch_topology)

        return patch_topology_graph, all_arcs
        

    def build_complex_recursive(self):

        has_scaffold = self.init_scaffold_vertices()

        cnt = 0 
        split = True
        while split:
            print(cnt, len(self.scaffold_corner_ids), split)
            cnt += 1
            ## split arcs
            patch_topology_graph, all_arcs = self.main_build()
            split = False
            for arc_id, arc in enumerate(all_arcs):
                ## split the arc at middle
                if arc['split']:
                    arc_0, arc_1 = self.split_arc(arc)
                    self.scaffold_corner_ids.append(arc_0['corner_ids'][1])
                    split = True

        
        # ## visualization
        # for patch_topology in patch_topology_graph:
        #     print(patch_topology)
        #     for idx, arc_id in enumerate(patch_topology['arcs']):
        #         arc = all_arcs[arc_id]
        #         arc_reverse = patch_topology['arc_reverse'][idx]
        #         if arc_reverse:
        #             vids = arc['vertices'][::-1]
        #         else:
        #             vids = arc['vertices']
        #         draw_colored_points_to_obj(
        #             f"./{self.savefolder}/{patch_topology['patch_name']}_arc_{idx}.obj", 
        #             self.base.vertices[vids], 
        #             scalars_for_color=np.arange(len(arc['vertices'])))

        # for arc_id, arc in enumerate(all_arcs):
        #     print(arc_id, arc['corner_ids'], arc['split'])
        #     write_obj_file(
        #         f"./{self.savefolder}/arc_{arc_id}.obj", 
        #         self.base.vertices[arc['vertices']])

        ## unique the corner ids
        graph = {
            'cells': [],
            'node_ids': [],
        }
        all_corner_ids = set()            
        for arc in all_arcs:
            all_corner_ids.add(int(arc['corner_ids'][0]))
            all_corner_ids.add(int(arc['corner_ids'][1]))
        all_corner_ids = list(all_corner_ids)
        graph['node_ids'] = all_corner_ids

        ## make cells
        for patch_topology in patch_topology_graph:
            cell = []
            for idx, arc_id in enumerate(patch_topology['arcs']):
                arc = all_arcs[arc_id]
                arc_reverse = patch_topology['arc_reverse'][idx]
                if arc_reverse:
                    cell.append(all_corner_ids.index(arc['corner_ids'][1]))
                else:
                    cell.append(all_corner_ids.index(arc['corner_ids'][0]))
            graph['cells'].append(cell)
        
        return graph

def normalize_data(mesh):
    mean_verts = np.mean(mesh.vertices, axis=0)
    scale = np.max(np.abs(mesh.vertices - mean_verts))*1.0
    mesh_out = mesh.copy()
    mesh_out.apply_translation(-mean_verts)
    mesh_out.apply_scale(1/scale)
    return mesh_out


def parse_args():
    parser = argparse.ArgumentParser(description="Modeling 3D shapes with neural patches")
    parser.add_argument("--model_name",
                        required=True,
                        type=str,
                        help="path to config"
                        )
    
    args = parser.parse_args()
    return args


if __name__ == '__main__':

    args = parse_args()
    
    ## basemesh needs to be watertight
    root = f'./data/{args.model_name}/data'
    meshfile = os.path.join(root, 'single/mesh.obj')
    mesh = trimesh.load(meshfile, process=False, maintain_order=True)

    mesh = normalize_data(mesh)
    mesh.export(meshfile) ## overwrite
    print(mesh.vertices.shape, mesh.faces.shape)

    mask_array = np.zeros((len(mesh.faces)))-1
    mask = read_json(os.path.join(root, 'mask.json'))


    savefolder = './out'
    if not os.path.exists(savefolder):
        os.makedirs(savefolder)

    patches = {}
    for i in range(len(mask)):
        submesh = mesh.submesh([mask[i]], only_watertight=False)[0]
        patches[i] = submesh
        submesh.export(f'{savefolder}/submesh_{i}.obj')

    ## build complex from the base mesh and its patches (dict)
    complex_builder = Complex(mesh, mask, savefolder=savefolder)
    graph = complex_builder.build_complex_recursive()
    print(graph)
    write_json(graph, os.path.join(root, "topology_graph.json"))
