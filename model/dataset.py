import sys
import os
sys.path.append(os.getcwd())

import numpy as np
import os
import trimesh
import igl
from PIL import Image
from common_tools.io_tools import read_json, write_obj_file, read_textured_obj_file, write_line_file, to_numpy, write_xyz_file, draw_colored_points_to_obj
import pickle
from scipy.spatial import KDTree
import networkx as nx
import triangle as tr
import potpourri3d as pp3d
import glob 
from plyfile import PlyData, PlyElement
import open3d as o3d

import matplotlib.cm as cm
import matplotlib

def compute_vertex_area(mesh):
    area_faces = mesh.area_faces
    area_vertices = np.zeros((mesh.vertices.shape[0]))
    for i in range(len(mesh.faces)):
        area_vertices[mesh.faces[i]] += area_faces[i]
    return area_vertices

def sample_barycentrics_from_mesh(mesh: trimesh.Trimesh, num_samples:int, weight=None):
    if weight is None:
        samples, fids = trimesh.sample.sample_surface_even(mesh, count=num_samples)
    else:
        samples, fids = trimesh.sample.sample_surface(mesh, count=num_samples, face_weight=weight)
    triangles = mesh.faces[fids]
    tri_vertices = mesh.vertices[triangles]

    ## use cross instead of cramer for better numerical stability
    barycentrics = trimesh.triangles.points_to_barycentric(tri_vertices, samples, method='cross')
    if np.isnan(barycentrics).any():
        ## find nan pos in barycentrics
        nan_pos = np.argwhere(np.isnan(barycentrics))
        nan_pos = np.unique(nan_pos[:,0])
        print("nan_pos", nan_pos)
        print(tri_vertices[nan_pos])
        print("barycentrics has nan", np.isnan(barycentrics).any())
        # input("Press Enter to continue")
    
    return barycentrics, fids
    
def get_points_from_barycentric(mesh, barycentrics, triangles, return_uv=False):
    tri_vertices = mesh.vertices[triangles]
    samples = trimesh.triangles.barycentric_to_points(tri_vertices, barycentrics)
    return samples

def get_one_ring(mesh:trimesh.Trimesh):
    g = nx.from_edgelist(mesh.edges_unique)
    one_ring = []
    max_len = 0
    for i in range(len(mesh.vertices)):
        one_ring.append(list(g[i].keys()))
        max_len = max(max_len, len(g[i].keys()))

    sparse_mat = np.zeros((len(mesh.vertices), max_len), dtype=np.int32)
    for i in range(len(mesh.vertices)):
        # ## circulating the data for padding, like 0123012
        # for j in range(max_len): 
        #     sparse_mat[i,j] = one_ring[i][j%len(one_ring[i])]
        
        ## pad -1
        sparse_mat[i,:] = -1
        sparse_mat[i,0:len(one_ring[i])] = one_ring[i]
    return sparse_mat

def sample_near_boundary(mesh, num_samples, rad=0.1):
    ## boundary
    bvids = igl.boundary_loop(mesh.faces)
    b_verts = mesh.vertices[bvids]

    ## densify the boundary
    tangent = np.roll(b_verts, -1, axis=0) - b_verts
    # tangent_norm = np.linalg.norm(tangent, axis=-1, keepdims=True)
    tparam = np.linspace(0, 1, 10)[1:-1]
    tparam = tparam.reshape(-1,1)
    b_vert_dense = b_verts[:,None,...] + tparam[None,...]*tangent[:,None,...]
    b_vert_dense = b_vert_dense.reshape(-1,3)

    b_tree = KDTree(b_vert_dense)

    ## dense sample the mesh
    samples, face_ids = trimesh.sample.sample_surface_even(mesh, count=num_samples*5)
    d, _ = b_tree.query(samples)

    if False:
        mask = np.bitwise_and(d < rad, d > 1e-3)
        # mask = np.ones((len(samples),), dtype=bool)
        samples = samples[mask]
        fids = face_ids[mask]
    else:
        ## 
        p = 1.0/(d+rad)
        p = p/np.sum(p)
        idx = np.random.choice(np.arange(len(samples)), num_samples, replace=False, p = p)
        samples = samples[idx]
        fids = face_ids[idx]

    write_obj_file(f"flat_uv.obj", samples)
    print("sampled", len(samples), "samples near boundary")

    triangles = mesh.faces[fids]
    tri_vertices = mesh.vertices[triangles]

    ## use cross instead of cramer for better numerical stability
    barycentrics = trimesh.triangles.points_to_barycentric(tri_vertices, samples, method='cross')
    if np.isnan(barycentrics).any():
        ## find nan pos in barycentrics
        nan_pos = np.argwhere(np.isnan(barycentrics))
        nan_pos = np.unique(nan_pos[:,0])
        print("nan_pos", nan_pos)
        print(tri_vertices[nan_pos])
        print("barycentrics has nan", np.isnan(barycentrics).any())
        # input("Press Enter to continue")

    if len(barycentrics) > num_samples:
        barycentrics = barycentrics[:num_samples]
        fids = fids[:num_samples]

    return barycentrics, fids

def sample_each_patch(flat:trimesh.Trimesh, patch:trimesh.Trimesh, add_samples:int):

    face_weight = np.ones((len(patch.faces)))+1
    for zipped in zip(patch.face_adjacency_angles, patch.face_adjacency):
        a, fpair = zipped
        f0 = fpair[0]
        f1 = fpair[1]
        face_weight[f0] = min(1+np.cos(a), face_weight[f0])
        face_weight[f1] = min(1+np.cos(a), face_weight[f0])

    ## sample for each uv domain a number of points
    ## type 1 samples
    flat_barycentrics, flat_fids = sample_near_boundary(flat, add_samples//2, rad=0.05)
    ## type 2 samples
    weight_curvature = 1.0 / (face_weight/3)
    flat_barycentrics_curvature, flat_fids_curvature = sample_barycentrics_from_mesh(flat, add_samples//2, weight=weight_curvature)
    flat_barycentrics = np.concatenate([flat_barycentrics_curvature, flat_barycentrics], axis=0)
    flat_fids = np.concatenate([flat_fids_curvature, flat_fids], axis=0)

    # ## get the uv coordinates and the 3D coordinates
    # flat_triangles = flat.faces[flat_fids]
    # flat_uv = get_points_from_barycentric(flat, flat_barycentrics, flat_triangles)
    # flat_samples = get_points_from_barycentric(patch, flat_barycentrics, flat_triangles)
    # flat_sample_normals = patch.face_normals[flat_fids]

    # ## organize the data
    # samples = np.concatenate([samples, flat_samples], axis=0)                
    # sample_normals = np.concatenate([sample_normals, flat_sample_normals], axis=0)
    # samples_uv = np.concatenate([samples_uv, flat_uv[:,:2]], axis=0)

    return flat_barycentrics, flat_fids


class ParamDataset():
    def __init__(self, root, batchsize=1, mode='train') -> None:
        self.batchsize = batchsize
        self.dir = root
        self.mode = mode

        print(f"Dataset mode set to {self.mode}")


        ## load complex
        json_data = read_json(os.path.join(root, 'data/topology_graph.json'))
        self.graph_cells = json_data['cells']
        
        try:
            self.node_ids = json_data['node_ids']
            ## load each patch and store the xyz and uv
            single_mesh_path = os.path.join(root, 'data/single/mesh.obj')
            if not os.path.exists(single_mesh_path):
                single_mesh_path = os.path.join(root, 'data/single/mesh.ply')
            single_mesh = trimesh.load(single_mesh_path, process=False, maintain_order=True)
            self.corners = single_mesh.vertices[self.node_ids,:]
        except KeyError:
            self.corners = json_data['node_coords']


        self.degs_list = read_json(os.path.join(root, 'data/cell_arc_lengths.json'))

        if False:  # False
            include_patch_ids = [0, 1]
            # include_patch_ids = [0,1,2,3,4,5,6,11]
            self.include_patch_ids = include_patch_ids
            self.degs_list = [self.degs_list[i] for i in include_patch_ids]

            """
            #####################################################################
            new_graph_cells = []
            new_node_ids = {}
            cnt = 0
            self.degs_list = [self.degs_list[i] for i in include_patch_ids]

            for i in range(len(self.graph_cells)):
                if i not in include_patch_ids:
                    continue
                
                # vids = self.graph_cells[i]
                # for vid in vids:
                #     nid = self.node_ids[vid]
                #     if nid not in new_node_ids:
                #         new_node_ids[nid] = cnt
                #         cnt += 1

                new_cell = []
                for cvid in self.graph_cells[i]:
                    nid = self.node_ids[cvid]
                    new_cell.append(new_node_ids[nid])
                
                new_graph_cells.append(new_cell)
            # self.node_ids = np.array(list(new_node_ids.keys()))
            self.graph_cells = new_graph_cells
            #####################################################################
            """

            kept_graph_cells = []
            referred_corner_ids = set()
            for i in include_patch_ids:
                kept_graph_cells.append(self.graph_cells[i])
                for vid in self.graph_cells[i]:
                    referred_corner_ids.add(vid)

            referred_corner_ids = list(referred_corner_ids)
            new_graph_cells = []
            for i, cell in enumerate(kept_graph_cells):
                new_cell = []
                for vid in cell:
                    new_cell.append(referred_corner_ids.index(vid))
                new_graph_cells.append(new_cell)

            corners = [self.corners[i] for i in referred_corner_ids]
            self.corners = corners
            self.graph_cells = new_graph_cells

            print("referred_corner_ids", referred_corner_ids)
            print("new_graph_cells", new_graph_cells)

        else:
            include_patch_ids = np.arange(len(self.graph_cells))
            self.include_patch_ids = include_patch_ids

        print("include_patch_ids: ", self.include_patch_ids)

        self.num_patches = len(self.graph_cells)
        self.num_nodes = len(self.corners)

        # write_obj_file("corners.obj", self.corners)

        # print("mesh vertices", single_mesh.vertices.shape)
  
    def sample(self, add_samples):
        include_patch_ids = self.include_patch_ids

        self.patch_xyz = []
        self.patch_normals = []
        self.patch_grads = []
        self.patch_masks = []
        self.patch_uv = []
        self.patch_ids = []
        self.patch_textures = []
        patch_folder = os.path.join(self.dir, 'parameterization')
        patch_tex_folder = os.path.join(self.dir, 'patch_textures')
        # self.patch_faces = []

        self.patch_list = []
        self.boundary_samples = []
        self.boundary_samples_uv = []
        self.boundary_sample_pids = []
        self.faces = []

        total_area = 0
        if True and add_samples > 0:
            total_add_samples = add_samples*self.num_patches
            for i, pid in enumerate(include_patch_ids):
                patchfile = os.path.join(patch_folder, f'mesh_uv_{pid}.obj')
                patch = trimesh.load(patchfile, process=False, maintain_order=True)
                total_area += patch.area
                

        for i, pid in enumerate(include_patch_ids):
            patchfile = os.path.join(patch_folder, f'mesh_uv_{pid}.obj')
            # print("Loading patch", pid, "from", patchfile)

            vertices, uv, faces, _ = read_textured_obj_file(patchfile)
            patch = trimesh.Trimesh(vertices=vertices, faces=faces, process=False, maintain_order=True)
            self.patch_list.append(patchfile)

            ## trace boundary
            unique_edges = patch.edges[trimesh.grouping.group_rows(patch.edges_sorted, require_count=1)]
            unique_edges = unique_edges.tolist()
            curve = []
            curve.append(unique_edges[0][0])
            while len(unique_edges) > 0:
                for edge in unique_edges:
                    if curve[-1] == edge[0]:
                        unique_edges.remove(edge)
                        curve.append(edge[1])
                        break
                    elif curve[-1] == edge[1]:
                        unique_edges.remove(edge)
                        curve.append(edge[0])
                        break
            boundary_vertex_ids = np.array(curve[:-1], dtype=np.int32)
            self.boundary_samples.append(patch.vertices[boundary_vertex_ids])
            
            flat_file = os.path.join(os.path.join(self.dir, 'flat_parameterization'), f'flat_{pid}.obj')
            flat = trimesh.load(flat_file, process=False, maintain_order=True)
            self.faces.append(flat.faces)

            self.boundary_samples_uv.append(flat.vertices[boundary_vertex_ids][:,0:2])
            self.boundary_sample_pids.append(np.ones((len(boundary_vertex_ids)))*i)

            # ## get non-boundary samples = all samples - boundary samples
            # no_boundary_vertices = True
            # if no_boundary_vertices:
            #     internal_vertex_ids = np.setdiff1d(np.arange(len(vertices)), boundary_vertex_ids)
            #     # print(internal_vertex_ids.shape, len(vertices), len(boundary_vertex_ids))
            # else:
            #     input("Now it has boundary vertices; press Enter to continue")
            #     internal_vertex_ids = np.arange(len(vertices))

            # samples = vertices[internal_vertex_ids]
            # sample_normals = patch.vertex_normals[internal_vertex_ids]
            # samples_uv = uv[internal_vertex_ids][:,0:2]
            samples = vertices
            sample_normals = patch.vertex_normals
            samples_uv = uv

            if total_area > 0:
                add_samples = max(int(total_add_samples*patch.area/total_area), 1000)

            if add_samples > 0:
                ## curvature weight
                face_weight = np.ones((len(patch.faces)))+1
                for zipped in zip(patch.face_adjacency_angles, patch.face_adjacency):
                    a, fpair = zipped
                    f0 = fpair[0]
                    f1 = fpair[1]
                    face_weight[f0] = min(1+np.cos(a), face_weight[f0])
                    face_weight[f1] = min(1+np.cos(a), face_weight[f0])

                if self.mode == 'train':
                    ## 
                    flat_barycentrics, flat_fids = sample_near_boundary(flat, add_samples//2, rad=0.05)

                    weight_curvature = 1.0 / (face_weight/3)
                    flat_barycentrics_curvature, flat_fids_curvature = sample_barycentrics_from_mesh(flat, add_samples//2, weight=weight_curvature)
                    flat_barycentrics = np.concatenate([flat_barycentrics_curvature, flat_barycentrics], axis=0)
                    flat_fids = np.concatenate([flat_fids_curvature, flat_fids], axis=0)

                    # flat_barycentrics, flat_fids = sample_barycentrics_from_mesh(flat, add_samples)
                    flat_triangles = flat.faces[flat_fids]
                    flat_uv = get_points_from_barycentric(flat, flat_barycentrics, flat_triangles)
                    flat_samples = get_points_from_barycentric(patch, flat_barycentrics, flat_triangles)
                    flat_sample_normals = patch.face_normals[flat_fids]

                    samples = np.concatenate([samples, flat_samples], axis=0)                
                    sample_normals = np.concatenate([sample_normals, flat_sample_normals], axis=0)
                    samples_uv = np.concatenate([samples_uv, flat_uv[:,:2]], axis=0)
                else:
                    flat_barycentrics, flat_fids = sample_barycentrics_from_mesh(flat, 30000)
                    flat_triangles = flat.faces[flat_fids]

                    ## not mesh vertices for evaluation
                    samples_uv = get_points_from_barycentric(flat, flat_barycentrics, flat_triangles)[:,:2]
                    samples = get_points_from_barycentric(patch, flat_barycentrics, flat_triangles)
                    sample_normals = patch.face_normals[flat_fids]

            ## sample dxdu dxdv
            def compute_sample_gradients(dim):
                offset = np.zeros_like(samples_uv)
                offset[:,dim] = 1e-3
                offset = samples_uv + offset
                offset = np.concatenate([offset, np.zeros((offset.shape[0], 1))], axis=-1)

                #
                # print(flat.vertices.shape, offset.shape)
                # write_obj_file(f"flat.obj", flat.vertices)
                # write_obj_file(f"offset.obj", offset)
                closest, dist, fids = trimesh.proximity.closest_point(flat, offset)
                mask = dist < 1e-10
                vertex_ids = flat.faces[fids]
                tri_vertices = flat.vertices[vertex_ids]
                
                ## compute uv's barycentric coordinates 
                barycentrics = trimesh.triangles.points_to_barycentric(tri_vertices, offset, method='cross')
                # barycentrics = trimesh.triangles.points_to_barycentric(tri_vertices, offset, method='cramer')
                xyz_offset = get_points_from_barycentric(patch, barycentrics, vertex_ids)

                ## compute the gradients
                # print("xyz_offset", xyz_offset.shape, samples.shape)
                grad = (xyz_offset - samples)
                denominator = np.linalg.norm(grad, axis=-1, keepdims=True)
                denominator = np.nan_to_num(denominator, nan=1.0, posinf=1.0, neginf=1.0)
                grad = grad/denominator
                return grad, mask

            grad_u, mask_u = compute_sample_gradients(dim=0)
            grad_v, mask_v = compute_sample_gradients(dim=1)

            mask = mask_u*mask_v
            grad = np.stack([grad_u, grad_v], axis=-1)

            self.patch_xyz.append(samples)
            self.patch_uv.append(samples_uv)
            self.patch_normals.append(sample_normals)
            self.patch_grads.append(grad)
            self.patch_masks.append(mask)
            self.patch_ids.append(np.ones((self.patch_xyz[-1].shape[0],1))*i)

            # print("Loaded patch", pid, "with", len(samples), "samples")

            # patch_tex_file = os.path.join(patch_tex_folder, f'{i}.png')
            # if os.path.exists(patch_tex_file):
            #     patch_tex_img = Image.open(patch_tex_file)
            #     self.patch_textures.append(patch_tex_img)
        
        self.patch_xyz = np.concatenate(self.patch_xyz, axis=0)
        self.patch_uv = np.concatenate(self.patch_uv, axis=0)
        self.patch_normals = np.concatenate(self.patch_normals, axis=0)
        self.patch_grads = np.concatenate(self.patch_grads, axis=0)
        self.patch_masks = np.concatenate(self.patch_masks, axis=0, dtype=bool)
        self.patch_ids = np.concatenate(self.patch_ids, axis=0)
        
        self.boundary_samples = np.concatenate(self.boundary_samples, axis=0)
        self.boundary_samples_uv = np.concatenate(self.boundary_samples_uv, axis=0)
        self.boundary_sample_pids = np.concatenate(self.boundary_sample_pids, axis=0)




        self.data_kdtree = KDTree(self.patch_xyz)
        self.num_points = self.patch_xyz.shape[0]
        self.num_batches = self.num_points // self.batchsize

        print("self.num_points", self.num_points, "self.num_batches", self.num_batches)

        self.data = {
            'points': self.patch_xyz,
            'uv': self.patch_uv,
            'normals': self.patch_normals,
            'grad': self.patch_grads,
            'mask': self.patch_masks,
            'pid': self.patch_ids,
            'corners': self.corners,
            "graph_cells": self.graph_cells,
            "num_nodes": self.num_nodes,
            "degs_list": self.degs_list,
            "boundary_samples": self.boundary_samples,
            "boundary_samples_uv": self.boundary_samples_uv,
            "boundary_sample_pids": self.boundary_sample_pids,
        }

        # # write_obj_file("gt_samples.obj", np.concatenate([self.patch_uv, np.zeros((self.patch_uv.shape[0], 1))], axis=-1))
        # write_obj_file("gt_samples.obj", self.patch_xyz)
        # write_obj_file("boundary_samples.obj", self.boundary_samples)


    def sample_for_point_cloud(self, num_samples:int):
        
        include_patch_ids = self.include_patch_ids

        self.patch_xyz = []
        self.patch_normals = []
        self.patch_grads = []
        self.patch_masks = []
        self.patch_uv = []
        self.patch_ids = []
        self.patch_textures = []
        patch_folder = os.path.join(self.dir, 'parameterization')
        patch_tex_folder = os.path.join(self.dir, 'patch_textures')
        # self.patch_faces = []

        self.patch_list = []
        self.boundary_samples = []
        self.boundary_samples_uv = []
        self.boundary_sample_pids = []

        for i, pid in enumerate(include_patch_ids):
            patchfile = os.path.join(patch_folder, f'mesh_uv_{pid}.obj')
            print("Loading patch", pid, "from", patchfile)

            print(patchfile)
            # vertices, uv, faces, _ = read_textured_obj_file(patchfile)
            # patch = trimesh.Trimesh(vertices=vertices, faces=faces, process=False, maintain_order=True)
            patch = trimesh.load(patchfile, process=False, maintain_order=True)
            self.patch_list.append(patchfile)

            ## trace boundary
            unique_edges = patch.edges[trimesh.grouping.group_rows(patch.edges_sorted, require_count=1)]
            unique_edges = unique_edges.tolist()
            curve = []
            curve.append(unique_edges[0][0])
            while len(unique_edges) > 0:
                for edge in unique_edges:
                    if curve[-1] == edge[0]:
                        unique_edges.remove(edge)
                        curve.append(edge[1])
                        break
                    elif curve[-1] == edge[1]:
                        unique_edges.remove(edge)
                        curve.append(edge[0])
                        break
            boundary_vertex_ids = np.array(curve[:-1], dtype=np.int32)
            self.boundary_samples.append(patch.vertices[boundary_vertex_ids])
            
            flat_file = os.path.join(os.path.join(self.dir, 'flat_parameterization'), f'flat_{pid}.obj')
            flat = trimesh.load(flat_file, process=False, maintain_order=True)

            self.boundary_samples_uv.append(flat.vertices[boundary_vertex_ids][:,0:2])
            self.boundary_sample_pids.append(np.ones((len(boundary_vertex_ids)))*i)

            flat_barycentrics, flat_fids = sample_barycentrics_from_mesh(flat, num_samples)
            flat_triangles = flat.faces[flat_fids]

            ## not mesh vertices for evaluation
            samples_uv = get_points_from_barycentric(flat, flat_barycentrics, flat_triangles)[:,:2]
            samples = get_points_from_barycentric(patch, flat_barycentrics, flat_triangles)
            sample_normals = patch.face_normals[flat_fids]

            solver = pp3d.PointCloudHeatSolver(samples)
            basisX, basisY, basisN = solver.get_tangent_frames()
            grad = np.stack([basisX, basisY], axis=-1)
            mask = np.ones((len(samples)), dtype=bool)

            self.patch_xyz.append(samples)
            self.patch_uv.append(samples_uv)
            self.patch_normals.append(sample_normals)
            self.patch_grads.append(grad)
            self.patch_masks.append(mask)
            self.patch_ids.append(np.ones((self.patch_xyz[-1].shape[0],1))*i)

            print("Loaded patch", pid, "with", len(samples), "samples")
        
        self.patch_xyz = np.concatenate(self.patch_xyz, axis=0)
        self.patch_uv = np.concatenate(self.patch_uv, axis=0)
        self.patch_normals = np.concatenate(self.patch_normals, axis=0)
        self.patch_grads = np.concatenate(self.patch_grads, axis=0)
        self.patch_masks = np.concatenate(self.patch_masks, axis=0, dtype=bool)
        self.patch_ids = np.concatenate(self.patch_ids, axis=0)
        
        self.boundary_samples = np.concatenate(self.boundary_samples, axis=0)
        self.boundary_samples_uv = np.concatenate(self.boundary_samples_uv, axis=0)
        self.boundary_sample_pids = np.concatenate(self.boundary_sample_pids, axis=0)

        self.num_points = self.patch_xyz.shape[0]
        self.num_batches = self.num_points // self.batchsize

        print("self.num_points", self.num_points, "self.num_batches", self.num_batches)

        self.data = {
            'points': self.patch_xyz,
            'uv': self.patch_uv,
            'normals': self.patch_normals,
            'grad': self.patch_grads,
            'mask': self.patch_masks,
            'pid': self.patch_ids,
            'corners': self.corners,
            "graph_cells": self.graph_cells,
            "num_nodes": self.num_nodes,
            "degs_list": self.degs_list,
            "boundary_samples": self.boundary_samples,
            "boundary_samples_uv": self.boundary_samples_uv,
            "boundary_sample_pids": self.boundary_sample_pids,
        }

        # # print(self.patch_uv.shape)
        # write_obj_file("gt_uvs.obj", np.concatenate([self.patch_uv, np.zeros((self.patch_uv.shape[0], 1))], axis=-1))
        # write_obj_file("gt_samples.obj", self.patch_xyz)
        # write_obj_file("boundary_samples.obj", self.boundary_samples)

class PointDataset(ParamDataset):
    def __init__(self, root, batchsize=1, mode='train') -> None:
        # super().__init__(root, batchsize, mode)
        self.batchsize = batchsize
        self.dir = root
        self.mode = mode

        print(f"Dataset mode set to {self.mode}")

        ## load complex
        json_data = read_json(os.path.join(root, 'topology_graph_normalized.json'))
        self.graph_cells = json_data['cells']
        
        self.pointcloud = trimesh.load(os.path.join(root, 'pc_normalized.ply'), process=False, maintain_order=True)
        self.total_area = self.pointcloud.vertices.shape[0]
        # plydata = PlyData.read(os.path.join(root, 'pc_normalized.ply'))
        # self.vertices = np.stack([plydata['vertex']['x'], plydata['vertex']['y'], plydata['vertex']['z']], axis=-1)
        # self.normals = np.stack([plydata['vertex']['nx'], plydata['vertex']['ny'], plydata['vertex']['nz']], axis=-1)
        self.corners = json_data['nodes']
        
        # include_patch_ids = np.arange(len(self.graph_cells))            
        # self.include_patch_ids = include_patch_ids

        if False:
            self.include_patch_ids = [3,4,5]
        else:
            self.include_patch_ids = []
            patch_folder = os.path.join(self.dir, 'normalized')
            for pid, cell in enumerate(self.graph_cells):
                patchfile = os.path.join(patch_folder, f'out_{pid}.ply') 
                if not os.path.exists(patchfile):
                    print(f"Patch {pid} not found, removed from the list")
                else:
                    self.include_patch_ids.append(pid)
        
        print("include_patch_ids: ", self.include_patch_ids)
        self.num_patches = len(self.include_patch_ids)
        self.num_nodes = len(self.corners)

        self.degs_list = []
        for i in range(len(self.graph_cells)):
            degs = [1/len(self.graph_cells[i]) for _ in range(len(self.graph_cells[i]))]
            self.degs_list.append(degs)
        
        ## cell corners; start from 0 degree
        self.cell_corners = []
        for i, degs in enumerate(self.degs_list):
            # degs.insert(0, 0) ## add cyclic start
            degs = 2*np.pi*np.array(np.cumsum(degs))
            degs = degs[:len(self.graph_cells[i])] ## remove cyclic last one
            # degs = torch.from_numpy(degs).to(torch.float32)
            x = np.cos(degs)
            y = np.sin(degs)
            corners = np.stack([x,y], axis=-1)
            self.cell_corners.append(corners)

    """
    curve points: [num_points, 3]
    """
    def resample_polyline(self, curve_points:np.array, num_samples:int):
        ## compute arc length parameters of each curve point
        arc_lengths = np.linalg.norm(np.diff(curve_points, axis=0), axis=-1)
        arc_lengths = np.concatenate([np.array([0]), np.cumsum(arc_lengths)])
        arc_lengths = arc_lengths / arc_lengths[-1]

        ## resample
        t = np.linspace(0, 1, num_samples)
        samples = np.zeros((num_samples, 3))
        for i in range(num_samples):
            idx = np.searchsorted(arc_lengths, t[i])
            if idx == 0:
                samples[i] = curve_points[0]
            elif idx == len(curve_points):
                samples[i] = curve_points[-1]
            else:
                alpha = (t[i] - arc_lengths[idx-1]) / (arc_lengths[idx] - arc_lengths[idx-1])
                samples[i] = (1-alpha)*curve_points[idx-1] + alpha*curve_points[idx]
        return samples


    def sample_for_point_cloud(self, num_samples: int):
        include_patch_ids = self.include_patch_ids

        self.patch_xyz = []
        self.patch_normals = []
        self.patch_gt_ids = []
        # self.patch_grads = []
        self.patch_masks = []
        self.patch_uv = []
        self.patch_uv_ids = []
        self.patch_textures = []
        patch_folder = os.path.join(self.dir, 'normalized')
        boundary_folder = os.path.join(self.dir, 'normalized/boundary')

        self.patch_list = []
        self.boundary_samples = []
        self.boundary_samples_uv = []
        self.boundary_sample_pids = []

        total_samples = num_samples*self.num_patches
        print(f"Total number of samples: {total_samples}")

        ## use point size to approximate the area
        patch_boundary_samples = []
        for i, pid in enumerate(include_patch_ids):

            # patchfile = os.path.join(patch_folder, f'out_{pid}.ply')
            # print("Loading patch", pid, "from", patchfile)
            # ## samples
            # pcd = o3d.io.read_point_cloud(patchfile)
            # print(f"Number of points in patch {pid}: {len(pcd.points)}")
            # if len(pcd.points) > 20000:
            #     downpcd = pcd.farthest_point_down_sample(20000)
            # else:
            #     ## repeated sampling
            #     points = np.array(pcd.points)
            #     idx = np.arange(len(points))
            #     # print(f"Number of points in patch {pid}: {len(points)}")
            #     rand_idx = np.random.choice(idx, 20000, replace=True, p=None)
            #     points = points[rand_idx]
            #     downpcd = o3d.geometry.PointCloud()
            #     downpcd.points = o3d.utility.Vector3dVector(points)
            # downpcd.estimate_normals()
            # downpcd.orient_normals_towards_camera_location()
            # downpcd.normalize_normals()
            ## save downsampled point cloud
            # o3d.io.write_point_cloud(f"downsampled_out_{pid}.ply", downpcd)

            patchfile = os.path.join(patch_folder, f'out_{pid}.ply')
            self.patch_list.append(patchfile)
            plydata = PlyData.read(patchfile)
            vertices = np.stack([plydata['vertex']['x'], plydata['vertex']['y'], plydata['vertex']['z']], axis=-1)
            
            normals = np.stack([plydata['vertex']['nx'], plydata['vertex']['ny'], plydata['vertex']['nz']], axis=-1)
            ## to make tensor shape, we resample the points
            # if len(vertices) > 20000:
            #     downsampled_idx = np.random.choice(len(vertices), 20000, replace=False)
            # else:
            #     downsampled_idx = np.random.choice(len(vertices), 20000, replace=True)
            if len(vertices) < 1000:
                downsampled_idx = np.arange(len(vertices))
            else:
                num_patch_samples = int(total_samples*float(len(vertices))/self.total_area)
                downsampled_idx = np.random.choice(len(vertices), num_patch_samples, replace=False)

            vertices = vertices[downsampled_idx]
            normals = normals[downsampled_idx]
            self.patch_xyz.append(vertices)
            self.patch_normals.append(normals)
            self.patch_gt_ids.append(np.ones(len(vertices))*i)

            ## boundary
            boundary_files = os.path.join(boundary_folder, f'b_out_{pid}_*')
            num_bfiles = len(glob.glob(boundary_files))
            # boundary_files = sorted(glob.glob(boundary_files))
            # for j, bf in enumerate(boundary_files):
            for j in range(num_bfiles):
                bf = os.path.join(boundary_folder, f'b_out_{pid}_{j}.ply')
                try:
                    boundary_patch = trimesh.load(bf, process=False, maintain_order=True)
                except:
                    boundary_patch = trimesh.load(bf.replace('.ply', '.obj'), process=False, maintain_order=True)
                # self.boundary_sample_pids.append(np.ones((len(boundary_patch.vertices)))*i)
                ## resampled
                sample_rate = 50 ## same as the boundary sample rate in complex object
                boundary_samples = self.resample_polyline(boundary_patch.vertices, sample_rate)
                # cid = np.arange(sample_rate)*1.0/sample_rate
                # draw_colored_points_to_obj(f"boundary_patch_{pid}_{j}.obj", boundary_samples, cid)
                patch_boundary_samples.append(boundary_samples)

            ## sample uv
            ratio = float(len(vertices)) / self.total_area
            print(f"Ratio of points in patch {pid}: {ratio}")
            num_samples = int(ratio*total_samples)
            num_samples = max(num_samples, 4000)
            print(f"Number of samples in patch {pid}: {num_samples}")
            sqrt_num_samples = int(np.sqrt(num_samples))
            # uvs = np.meshgrid(np.linspace(0, 1, sqrt_num_samples), np.linspace(0, 1, sqrt_num_samples))
            # uvs = np.stack(uvs, axis=-1).reshape(-1,2)
            ## point inside the patch
            
            def point_is_inside(p, crns):
                angle_sum = 0
                L = len(crns)
                for i in range(L):
                    a = crns[i]
                    b = crns[(i + 1) % L]
                    # cross = np.cross(a - p, b - p)
                    ap = (a - p)
                    bp = (b - p)
                    cross = (ap[:,0]*bp[:,1] - ap[:,1]*bp[:,0])
                    inner = np.sum((a-p)*(b-p), axis=-1)
                    angle_sum += np.arctan2(cross, inner)
                mask = np.abs(angle_sum) > 1
                return mask
            
            def pre_compute_uv_grid(crns, sample_size: int):
                x = np.linspace(-1, 1, sample_size)
                y = np.linspace(-1, 1, sample_size)
                xy = np.stack(np.meshgrid(x, y), axis=-1).reshape(-1, 2)
                mask = point_is_inside(xy, crns)
                xy = xy[mask]
                return xy

            uvs = pre_compute_uv_grid(self.cell_corners[pid], sqrt_num_samples)
            self.patch_uv.append(uvs)
            
            ## masks
            mask = np.ones((len(uvs)), dtype=bool)
            self.patch_masks.append(mask)
            self.patch_uv_ids.append(np.ones((mask.shape[0],1))*i)
        
            print("Loaded patch", patchfile, "with", len(self.patch_xyz[-1]), "points (GT) and", len(uvs), "samples (UV)")
        
        ## normalization
        # self.patch_xyz = np.stack(self.patch_xyz, axis=0)
        # self.patch_normals = np.stack(self.patch_normals, axis=0)
        self.patch_xyz = np.concatenate(self.patch_xyz, axis=0)
        self.patch_normals = np.concatenate(self.patch_normals, axis=0)
        self.patch_gt_ids = np.concatenate(self.patch_gt_ids, axis=0)
        self.boundary_samples = np.stack(patch_boundary_samples, axis=0)
        # self.boundary_sample_pids = np.concatenate(self.boundary_sample_pids, axis=0)

        # ## normalize
        # xyz = self.patch_xyz.reshape(-1,3)
        # xyz = xyz - np.mean(xyz, axis=0)
        # xyz = xyz / np.max(np.linalg.norm(xyz, axis=-1))
        # self.patch_xyz = xyz.reshape(self.patch_xyz.shape)        

        ## concatenate
        self.patch_uv = np.concatenate(self.patch_uv, axis=0)
        self.patch_uv_ids = np.concatenate(self.patch_uv_ids, axis=0)
        self.patch_masks = np.concatenate(self.patch_masks, axis=0, dtype=bool)

        self.num_points = self.patch_xyz.shape[0]
        self.num_batches = self.num_points // self.batchsize

        print("self.num_points", self.num_points, "self.num_batches", self.num_batches)

        self.data = {
            'points': self.patch_xyz,
            'normals': self.patch_normals,
            'uv': self.patch_uv,
            'mask': self.patch_masks,
            'gt_pid': self.patch_gt_ids,
            'uv_pid': self.patch_uv_ids,
            'corners': self.corners,
            "graph_cells": self.graph_cells,
            "num_nodes": self.num_nodes,
            "degs_list": self.degs_list,
            "boundary_samples": self.boundary_samples,
            "boundary_sample_pids": self.boundary_sample_pids,
        }

        # # print(self.patch_uv.shape)
        # write_obj_file("gt_uvs.obj", np.concatenate([self.patch_uv, np.zeros((self.patch_uv.shape[0], 1))], axis=-1))
        # write_obj_file("gt_samples.obj", self.patch_xyz.reshape(-1,3))
        write_xyz_file("gt_samples.xyz", self.patch_xyz.reshape(-1,3), self.patch_normals.reshape(-1,3))




if __name__ == "__main__":
    # root = "./data/bimba_yanglei_20240205_BPE_harmonic"
    # root = "./data/auto_seg_data2/auto_388_20240320_2144"
    # root = "./data/auto_seg_data2/auto_389_20240317_2048"
    # root = "./data/dress_20240407_1655_cut_66"
    # texture_img = Image.open(f'./asset/cm_tab20_v.png')
    # dataset = ParamDataset(root, batchsize=10000)
    # dataset.sample(add_samples=10000)
    # xyz = dataset.data['points']
    
    root = "./data/old_data/may24_dp3_skirt_529"
    dataset = PointDataset(root, batchsize=10000)
    dataset.sample_for_point_cloud(num_samples=10000)

    # ## color
    # norm = matplotlib.colors.Normalize(0,20, clip=True)
    # mapper = cm.ScalarMappable(norm=norm, cmap=cm.tab20)

    # corners = dataset.data['corners']
    # cells = dataset.data['graph_cells']
    # print(corners.shape, cells)

    # # for i, cell in enumerate(cells):
    # #     r,g,b,a = mapper.to_rgba(i%20)
    # #     write_obj_file(f"cell_{i}.obj", corners[cell], C=np.array([[r,g,b]]*len(cell))*255)


    # rgb = np.ones_like(xyz)
    # for i in range(len(dataset.patch_list)):
    #     print(i, i%20, mapper.to_rgba(i%20))
    #     r,g,b,a = mapper.to_rgba(i%20)
    #     rgb[pid==i] = np.array([[r,g,b]])*255
    # rgb = rgb.astype(np.uint8)
    # write_obj_file(f"gt_samples.obj", xyz, C=rgb)

    # meshfile = os.path.join(os.path.join(root, 'parameterization'), f'mesh_uv_{pid}.obj')
    # vertices, uvcoords, faces, _ = read_textured_obj_file(meshfile)
    # xyz_mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False, maintain_order=True)
    # xyz_mesh.visual = trimesh.visual.TextureVisuals(uv=uvcoords, material=None, image=texture_img)
    # xyz_mesh.export(f'xyz_mesh.obj')



    # pid = 0
    # gu, masku = dataset.compute_sample_gradients(pid=0, dim=0)
    # xyz = dataset.data['points'][masku]
    # gu = gu[masku]
    # paired_xyz = gu*0.001 + xyz
    # all_pts = np.stack([paired_xyz, xyz], axis=1)
    # all_pts = all_pts.reshape(-1,3)
    # links = [np.arange(0, len(all_pts), 2), np.arange(1, len(all_pts), 2)]
    # links = np.stack(links, axis=-1)
    # write_line_file(f"dxdu_gt.obj", all_pts, links)

    # gv, maskv = dataset.compute_sample_gradients(pid=0, dim=1)
    # xyz = dataset.data['points'][maskv]
    # gv = gv[maskv]
    # paired_xyz = gv*0.001 + xyz
    # all_pts = np.stack([paired_xyz, xyz], axis=1)
    # all_pts = all_pts.reshape(-1,3)
    # links = [np.arange(0, len(all_pts), 2), np.arange(1, len(all_pts), 2)]
    # links = np.stack(links, axis=-1)
    # write_line_file(f"dxdv_gt.obj", all_pts, links)

    # """
    # I need to add this part to the initialization of the dataset
    # Then, for each iteration, I will use the grads to supervise the network training
    # """