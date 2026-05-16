import numpy as np
from common_tools.io_tools import *
import os

import PIL.Image as Image
import triangle as tr
# from scipy.spatial import Delaunay as tr
import trimesh 
import cv2
import numpy as np
import igl
from tqdm import trange

import torch
from torch.utils.tensorboard import SummaryWriter
from torch import optim
import torch.nn.functional as F

from model.model import MultiNetGraph, Complex
from model.dataset import PointDataset

from vedo import mesh
from model.para_trainer import BaseBatchTrainer, vec_dist_norm, vec_cosine_signagnostic, vec_tdm, vec_dist

class OptimizerWrapper():
    def __init__(self, optimizer_list, scheduler_list=None):
        self.optimizers = optimizer_list
        self.schedulers = scheduler_list
    
    def opt_zero_grad(self):
        for opt in self.optimizers:
            opt.zero_grad()
    
    def opt_step(self):
        for opt in self.optimizers:
            opt.step()
    
    def sch_step(self):
        if self.schedulers is None:
            return
        
        for sch in self.schedulers:
            sch.step()

    def print_last_lr(self):
        lrs = []
        for sch in self.schedulers:
            lrs.append(sch.get_last_lr())
        return lrs

class SingleTrainerPC(BaseBatchTrainer):
    def __init__(self, cfg):
        super(SingleTrainerPC).__init__()

        self.max_epoch = cfg['max_epoch'] + 1
        self.eval_step = cfg['eval_step']

        ############################
        resume_path = cfg['resume']
        checkpoint = cfg['checkpoint']
        self.save_dir = cfg['save_dir']
        
        if cfg['mode'] == 'train':
            self.ckpt_dir = cfg['ckpt_dir']
        else:
            assert resume_path is not None

        self.train_method = cfg['train_method']
        self.mode = cfg['mode'] == 'train'
        
        ############################
        if cfg['cuda_id'] > -1:
            self.device = torch.device(f"cuda:{cfg['cuda_id']}")
        else:
            self.device = torch.device(f"cpu")
        
        if cfg['dtype'] == 'double':
            self.dtype = torch.double
        else:
            self.dtype = torch.float

        self.dataset = PointDataset(cfg['train_data'], batchsize=cfg['batchsize'], mode=cfg['mode'])
        if cfg['train_method'] == 'no_param':
            self.dataset.sample_for_point_cloud(cfg['add_samples'])
        else:
            raise NotImplementedError

        self.num_patches = self.dataset.num_patches
        # self.num_nodes = len(self.dataset.node_ids)
        self.num_nodes = self.dataset.num_nodes
        
        print(self.dataset.dir)


        self.complex = Complex(
            # node_ids = self.dataset.node_ids,
            num_nodes = self.num_nodes,
            cells = self.dataset.graph_cells,
            uvs = self.dataset.patch_uv,
            uv_to_cid = self.dataset.patch_uv_ids,
            degs_list = self.dataset.degs_list,
            device = self.device, 
            dtype = self.dtype,
            require_grad = self.mode)
        
        self.use_duv = cfg['net_params']['use_duv']
        self.net = MultiNetGraph(
            complex=self.complex,
            device=self.device,
            dtype=self.dtype,
            **cfg['net_params']
        )
        init_embed_data = torch.tensor(self.dataset.corners, dtype=self.dtype, device=self.device)
        self.net.init_embedding_weights_with_data(init_embed_data)
        self.net = self.net.to(self.device)
        
        
        trainable_parameters = []
        embedding = []
        for n, p in self.net.named_parameters():
            if n != 'nodal_embeddings.weight':
                trainable_parameters.append(p)
            else:
                embedding.append(p)


        """
        do not change this
        """
        # emb_optimizer = optim.Adam(embedding, lr=0.0005, amsgrad=False)
        # nn_optimizer = optim.Adam(trainable_parameters, lr=0.001, amsgrad=False)

        # ## dirichlet energy
        emb_optimizer = optim.Adam(embedding, lr=0.000, amsgrad=False)
        nn_optimizer = optim.Adam(trainable_parameters, lr=0.001, amsgrad=False)

        if True:
            emb_scheduler = optim.lr_scheduler.CosineAnnealingLR(emb_optimizer, T_max=self.max_epoch, eta_min=0.00001)
            nn_scheduler = optim.lr_scheduler.CosineAnnealingLR(nn_optimizer, T_max=self.max_epoch, eta_min=0.00001)
            self.opt = OptimizerWrapper([emb_optimizer, nn_optimizer], [emb_scheduler, nn_scheduler])
        else:
            emb_scheduler1 = optim.lr_scheduler.LinearLR(emb_optimizer, start_factor=0.01, end_factor=1, total_iters=1000)
            emb_scheduler2 = optim.lr_scheduler.CosineAnnealingLR(emb_optimizer, T_max=self.max_epoch, eta_min=0.00001)
            emb_scheduler = optim.lr_scheduler.SequentialLR(emb_optimizer, schedulers=[emb_scheduler1, emb_scheduler2], milestones=[1000])

            # emb_scheduler3 = optim.lr_scheduler.LinearLR(emb_optimizer, start_factor=0.01, end_factor=0.01, total_iters=self.max_epoch//2)
            # emb_scheduler = optim.lr_scheduler.SequentialLR(emb_optimizer, schedulers=[emb_scheduler1, emb_scheduler2, emb_scheduler3], milestones=[1000, 6000])

            nn_scheduler1 = optim.lr_scheduler.LinearLR(nn_optimizer, start_factor=0.01, end_factor=1, total_iters=1000)
            nn_scheduler2 = optim.lr_scheduler.CosineAnnealingLR(nn_optimizer, T_max=self.max_epoch, eta_min=0.00001)
            nn_scheduler = optim.lr_scheduler.SequentialLR(nn_optimizer, schedulers=[nn_scheduler1, nn_scheduler2], milestones=[1000])

            # nn_scheduler3 = optim.lr_scheduler.LinearLR(nn_optimizer, start_factor=0.01, end_factor=0.01, total_iters=self.max_epoch//2)
            # nn_scheduler = optim.lr_scheduler.SequentialLR(nn_optimizer, schedulers=[nn_scheduler1, nn_scheduler2, nn_scheduler3], milestones=[1000, 6000])
                        
            self.opt = OptimizerWrapper([emb_optimizer, nn_optimizer], [emb_scheduler, nn_scheduler])
            # self.opt = OptimizerWrapper([nn_optimizer], [nn_scheduler])
        
        self.start_epoch = 0
        if resume_path is not None:
            print(resume_path)
            ckpt_path = os.path.join(resume_path, f'ckpt/{checkpoint}.pth')
            print(ckpt_path)
            state_dict = torch.load(ckpt_path, map_location=self.device)
            print(state_dict['model_state_dict'].keys())
            self.net.load_state_dict(state_dict['model_state_dict'], strict=False)
            self.start_epoch = state_dict['epoch']
            print("resume")

        self.ckpt_dir = cfg['ckpt_dir']

        self.loss_params = cfg['loss_params']
        self.surf_metric = cfg['loss_params']['surf_metric'] if 'surf_metric' in cfg['loss_params'] else 'pdm'

        if os.path.exists(f'{self.dataset.dir}/data/smooth_boundaries.json'):
            self.smooth_b_constraint = read_json(
                f'{self.dataset.dir}/data/smooth_boundaries.json')
            print("has smooth_boundaries")
        else:
            self.smooth_b_constraint = None

        print(cfg)
        print("Trainer init")


    ## 
    def compute_boundary_and_normal(self):
        bxx, _, _ = self.net.forward_boundary()
        # boundary_nxx = boundary_nxx.reshape(-1, self.complex.boundary_sample_rate, 3)
        bxx = bxx.reshape(-1, self.complex.boundary_sample_rate, 3)
        return bxx, None


    def evaluate(self, epoch, sample_idx=None):
        return {}
    
        assert sample_idx is not None
        corners = self.dataset.corners

        patch_samples = self.dataset.patch_xyz
        patch_sample_normals = self.dataset.patch_normals

        corners = torch.tensor(corners, device=self.device, dtype=self.dtype)
        patch_samples = torch.tensor(patch_samples, device=self.device, dtype=self.dtype)
        patch_sample_normals = torch.tensor(patch_sample_normals, device=self.device, dtype=self.dtype)
        
        x, uv, uv_to_cid = self.net.evaluate_patch()
        if sample_idx is not None:
            x = x[sample_idx]
            patch_samples = patch_samples[sample_idx]
            uv_to_cid = uv_to_cid[sample_idx]
            patch_sample_normals = patch_sample_normals[sample_idx]

        surf_loss = F.l1_loss(x, patch_samples)
        max_surf_loss = torch.max(torch.norm(x-patch_samples, dim=-1))
        surf_tqm_loss = vec_tdm(x, patch_samples, patch_sample_normals)
        x_crns = self.net.forward_corners()
        corner_loss = F.l1_loss(x_crns, corners)

        loss_dict = {
            "corner_loss": corner_loss,
            "surf_loss": surf_loss,
            "max_surf_loss": max_surf_loss,
            "tdm_loss": surf_tqm_loss,
            'epoch': epoch,
        }
        return loss_dict
    
    

    def train_iteration(self, epoch, sample_idx=None):

        self.opt.opt_zero_grad()

        t0 = time.time()
        loss, loss_dict = self.train_iteration_match(epoch, sample_idx)
        t1 = time.time()

        
        ############################################### 
        t0 = time.time()        
        loss.backward()
        t1 = time.time()
        # print("backward: total time: ", t1-t0)
        param_grad, emb_grad = self.check_grad_norm()
        self.opt.opt_step()
        self.opt.sch_step()
        t2 = time.time()
        return {"loss": loss.item(), **loss_dict, "param_grad": param_grad, "emb_grad": emb_grad}
    
    

    def train_iteration_match(self,  epoch:int,  sample_idx:list=None):
        # self.opt.opt_zero_grad()
        ###############################################
        
        # corners = data['corners']
        corners = self.dataset.corners
        corners = torch.tensor(corners, device=self.device, dtype=self.dtype)

        boundary_samples = self.dataset.data['boundary_samples']
        boundary_samples = torch.tensor(boundary_samples, device=self.device, dtype=self.dtype)
        # for i in range(len(boundary_samples)):
        #     boundary_samples[i] = torch.tensor(boundary_samples[i], device=self.device, dtype=self.dtype)

        mask = self.dataset.data['mask']
        mask = torch.tensor(mask, device=self.device, dtype=torch.bool)

        ## gt
        patch_samples = self.dataset.patch_xyz
        patch_sample_normals = self.dataset.patch_normals
        patch_gt_ids = self.dataset.patch_gt_ids.squeeze()
        patch_uv_ids = self.dataset.patch_uv_ids.squeeze()

        # ## out of memory if patch_samples are too large
        # random_idx = np.random.choice(len(patch_samples), 10000, replace=False)
        # patch_samples = torch.tensor(patch_samples[random_idx,:], device=self.device, dtype=self.dtype)
        # patch_sample_normals = torch.tensor(patch_sample_normals[random_idx,:], device=self.device, dtype=self.dtype)
        patch_samples = torch.tensor(patch_samples, device=self.device, dtype=self.dtype)
        patch_sample_normals = torch.tensor(patch_sample_normals, device=self.device, dtype=self.dtype)
        patch_gt_ids = torch.tensor(patch_gt_ids, device=self.device, dtype=torch.long)
        patch_uv_ids = torch.tensor(patch_uv_ids, device=self.device, dtype=torch.long)
        
        boundary_loss = 0.0
        boundary_normal_smooth = 0.0
        boundary_chamfer_loss = 0.0
        ## another way to compute boundary loss
        patch_bxx_list = None
        if True:
            bxx, bnxx, bxx_to_cid = self.net.forward_boundary(return_normals=True)
            bxx_reshaped = bxx.reshape(-1, self.complex.boundary_sample_rate, 3)
            bnxx_reshaped = bnxx.reshape(-1, self.complex.boundary_sample_rate, 3)
            diff_bxx = bxx_reshaped[:,1:-1,:] - 0.5*(bxx_reshaped[:,:-2,:] + bxx_reshaped[:,2:,:])
            boundary_loss = torch.norm(diff_bxx, dim=-1).mean()

            # if epoch % 500 == 0:
            #     cid = np.arange(self.complex.boundary_sample_rate)*1.0/self.complex.boundary_sample_rate
            #     for ii, bxx_i in enumerate(bxx_reshaped):
            #         draw_colored_points_to_obj(f"pred_b_patch_{ii}.obj", to_numpy(bxx_i), cid)
            boundary_chamfer_loss = vec_dist_norm(bxx_reshaped, boundary_samples)
            boundary_chamfer_loss = boundary_chamfer_loss.mean()

            if not hasattr(self, 'sparse_adjacency'):
                # bxx_reshaped ## (K, N, 3)
                bxx_reshaped = bxx_reshaped.detach()
                sparse_adjacency = []
                for i, bxx_i in enumerate(bxx_reshaped):
                    for j, bxx_j in enumerate(bxx_reshaped):
                        if i == j or i > j:
                            continue
                        dist_matrix = torch.cdist(bxx_i, bxx_j, p=2)
                        dist1, idx1 = torch.min(dist_matrix, axis=1)
                        dist2, idx2 = torch.min(dist_matrix, axis=0)
                        dist = torch.mean(dist1)+torch.mean(dist2)
                        # print(i, j, dist)
                        if dist < 0.00005:
                            sparse_adjacency.append([i, j])
                print("sparse_adjacency:\n", sparse_adjacency)
                self.sparse_adjacency = sparse_adjacency

            for i, j in self.sparse_adjacency:
                bnxx_reshaped_i = bnxx_reshaped[i][1::, :] ## skip the first point
                bnxx_reshaped_j = bnxx_reshaped[j][torch.arange(self.complex.boundary_sample_rate-1, 0, -1),:] ## reverse and skip the first point
                bnerr_ij = vec_dist_norm(bnxx_reshaped_i, bnxx_reshaped_j)
                boundary_normal_smooth += bnerr_ij

            boundary_normal_smooth /= len(self.sparse_adjacency)

            # patch_bxx_list = []
            # for pid in range(self.num_patches):
            #     patch_bxx_list.append(bxx[bxx_to_cid == pid])
           
        surf_loss = 0.0
        normal_loss = 0.0
        surf_tqm_loss = 0.0
        varia_energy = 0.0
        conformal_loss = 0.0
        cycle_loss = 0.0
        reg_loss = 0.0
        for pid in range(self.num_patches):
            pts_idx = [patch_uv_ids == pid]
            xx, nxx, uv_to_cid, varia_energy_i, conformal_loss_i, cycle_loss_i, reg_loss_i = self.net.forward_dgp(pts_idx=pts_idx)

            patch_samples_i = patch_samples[patch_gt_ids == pid]
            patch_sample_normals_i = patch_sample_normals[patch_gt_ids == pid]

            varia_energy += varia_energy_i
            conformal_loss += conformal_loss_i
            cycle_loss += cycle_loss_i
            reg_loss += reg_loss_i
            
            dist_matrix = torch.cdist(xx.detach(), patch_samples_i, p=2)
            idx = torch.argmin(dist_matrix, axis=1)
            idx2 = torch.argmin(dist_matrix, axis=0)

            dist = vec_dist(xx, patch_samples_i[idx]) 
            ndist = vec_dist(nxx, patch_sample_normals_i[idx])
            # ndist = vec_cosine_signagnostic(nxx, patch_sample_normals[pid][idx])
            ## normal vectors are not consitently oriented
            
            dist2 = vec_dist(xx[idx2], patch_samples_i) 
            ndist2 = vec_dist(nxx[idx2], patch_sample_normals_i) 
            # ndist2 = vec_cosine_signagnostic(nxx[idx2], patch_sample_normals[pid])

            surf_loss += 0.1*dist + dist2
            normal_loss += 0.1*ndist + ndist2
            # surf_tqm_loss += 0.1*tdm + tdm2

        surf_loss /= self.num_patches
        normal_loss /= self.num_patches
        surf_tqm_loss /= self.num_patches
        varia_energy /= self.num_patches
        conformal_loss /= self.num_patches
        cycle_loss /= self.num_patches
        reg_loss /= self.num_patches
        boundary_loss /= self.num_patches
        boundary_normal_smooth /= self.num_patches

        x_crns = self.net.forward_corners()
        # corner_loss = vec_dist(x_crns, corners)
        corner_loss = vec_dist(x_crns, corners)

        loss_dict = {
            "corner_loss": corner_loss,
            "surf_loss": surf_loss,
            "nomral_loss": normal_loss,
            "boundary_loss": boundary_loss,
            "boundary_normal_smooth": boundary_normal_smooth,
            "boundary_chamfer_loss": boundary_chamfer_loss,
            "tdm_loss": surf_tqm_loss,
            "conformal_loss": conformal_loss,
            "varia_energy": varia_energy,
            "cycle_loss": cycle_loss,
            "reg_loss": reg_loss,
            "epoch": epoch,
        }
        
        
        # loss = self.loss_fn(loss_dict)

        corner_loss = loss_dict['corner_loss']
        surf_loss = loss_dict['surf_loss']
        surf_tqm_loss = loss_dict['tdm_loss']
        boundary_loss = loss_dict['boundary_loss']
        varia_energy = loss_dict['varia_energy']

        surf_loss_weight = self.loss_params['alpha']
        normal_loss_weight = self.loss_params['normal']
        uniform_loss_weight = self.loss_params['uniform']
        conformal_loss_weight = self.loss_params['conformal']
        cycle_loss_weight = self.loss_params['cycle']
        boundary_weight = self.loss_params['boundary']

        ######################################################
        epoch = loss_dict['epoch']
        # scale = min(epoch/3000.0, 1.0)

        loss = 0.0
        loss += 0.5*reg_loss
        loss += 0.1*surf_loss_weight*corner_loss
        loss += 0.1*boundary_weight*boundary_loss ## 0.1 for the skirt
        loss += 0.1*uniform_loss_weight*varia_energy
        loss += cycle_loss_weight*cycle_loss

        if epoch > 1000:
            loss += surf_loss_weight*surf_loss
            loss += 0.1*surf_loss_weight*boundary_chamfer_loss
        if epoch > 5000:
            # loss += cycle_loss_weight*cycle_loss
            loss += 0.01*boundary_normal_smooth ## for raw point cloud; 0.001 is good
            loss += normal_loss_weight*normal_loss

        return loss, loss_dict


    def extract_mesh(self, epoch, mesh_res=50):
        patchmeshes = self.show_patches(epoch=epoch, name="out", mesh_res=mesh_res)
        return patchmeshes
        

    def extract_mesh_with_uv(self, epoch):
        self.patch_list = self.dataset.patch_list
        # texture_img = Image.open(f'./asset/cm_tab20_v.png')
        texture_img = Image.open(f'./asset/checkerboard.png')
        
        patchmeshes = []
        for cid, patchfile in enumerate(self.patch_list):
            # print(patchfile)
            _, uv, faces, _ = read_textured_obj_file(patchfile)
            uv = torch.tensor(uv, device=self.device, dtype=self.dtype)
            xyz, _ = self.net.forward_uv(uv, cid=cid)
            xyz = to_numpy(xyz)
            patchmesh = trimesh.Trimesh(vertices=xyz, faces=faces, process=False, maintain_order=True)

            cm_uv = to_numpy(uv)
            # ## texture
            # cm_uv = np.ones_like(to_numpy(uv))
            # cm_uv[:,1] = (cid%20+0.5)/20

            patchmesh.visual = trimesh.visual.TextureVisuals(uv=cm_uv, material=None, image=texture_img)
            patchmeshes.append(patchmesh)
            
            # if not os.path.exists(f"{self.save_dir}/{cid}"):
            #     os.makedirs(f"{self.save_dir}/{cid}")
            # patchmesh.export(f"{self.save_dir}/out_{epoch}_{cid}.obj")

        scene = trimesh.Scene(patchmeshes)
        scene.export(f"{self.save_dir}/out_{epoch}.obj")
        return patchmeshes

    def extract_mesh_with_uv2(self, epoch, uv_mesh:trimesh.Trimesh):

        uv = torch.tensor(uv_mesh.vertices[:,:2], device=self.device, dtype=self.dtype)
        xyz, dx_du, mask = self.net.forward_uv_w_grad(uv, cid=0)

        xyz = to_numpy(xyz)
        uv_mesh.vertices = xyz
        uv_mesh.export(f"{self.save_dir}/xyz_{epoch}.obj")

        ## normal
        dx_du = to_numpy(dx_du)
        normals = np.cross(dx_du[:,:,0], dx_du[:,:,1], axis=-1)
        denominator = np.linalg.norm(normals, axis=-1, keepdims=True)
        denominator = np.nan_to_num(denominator, nan=1.0, posinf=1.0, neginf=1.0)
        denominator = np.clip(denominator, a_min=1e-6, a_max=None)
        normals = normals/denominator ## normalize

        paired_xyz = normals*0.0001 + xyz
        all_pts = np.stack([paired_xyz, xyz], axis=1)
        all_pts = all_pts.reshape(-1,3)
        links = [np.arange(0, len(all_pts), 2), np.arange(1, len(all_pts), 2)]
        links = np.stack(links, axis=-1)
        write_line_file(f"{self.save_dir}/normal_vec_{epoch}.obj", all_pts, links)

        ## tangent u
        dxdu = dx_du[:,:,0]
        denominator = np.linalg.norm(dxdu, axis=-1, keepdims=True)
        denominator = np.nan_to_num(denominator, nan=1.0, posinf=1.0, neginf=1.0)
        denominator = np.clip(denominator, a_min=1e-6, a_max=None)
        dxdu = dxdu/denominator
        paired_xyz = dxdu*0.00005 + xyz
        all_pts = np.stack([paired_xyz, xyz], axis=1)
        all_pts = all_pts.reshape(-1,3)
        links = [np.arange(0, len(all_pts), 2), np.arange(1, len(all_pts), 2)]
        links = np.stack(links, axis=-1)
        write_line_file(f"{self.save_dir}/dxdu_{epoch}.obj", all_pts, links)
        
        ## tangent v
        dxdv = dx_du[:,:,1]
        denominator = np.linalg.norm(dxdv, axis=-1, keepdims=True)
        denominator = np.nan_to_num(denominator, nan=1.0, posinf=1.0, neginf=1.0)
        denominator = np.clip(denominator, a_min=1e-6, a_max=None)
        dxdv = dxdv/denominator
        paired_xyz = dxdv*0.00005 + xyz
        all_pts = np.stack([paired_xyz, xyz], axis=1)
        all_pts = all_pts.reshape(-1,3)
        links = [np.arange(0, len(all_pts), 2), np.arange(1, len(all_pts), 2)]
        links = np.stack(links, axis=-1)
        write_line_file(f"{self.save_dir}/dxdv_{epoch}.obj", all_pts, links)

        return mask


    def extract_points(self, epoch):
        x = self.net()
        x = to_numpy(x)
        print(f"{self.save_dir}/out_{epoch}.obj")
        write_obj_file(f"{self.save_dir}/out_{epoch}.obj", x)


    def extract_mesh_remeshed(self, epoch):
        # texture_img = Image.open(f'./asset/checkerboard.png')
        
        texture_imgs = []
        for i in range(1, 10):
            texture_imgs.append(Image.open(f'./asset/checkboard/Slide{i}.jpg'))
        
        buv_list = self.net.complex.boundary_sampling(100)
        uv_list = self.net.complex.interior_sampling(200)
        patchmeshes = []
        for pid, _ in enumerate(self.dataset.patch_list):        
            buv = buv_list[pid]
            uv = uv_list[pid]
            bxyz, _ = self.net.forward_uv(buv, cid=pid)
            xyz, _ = self.net.forward_uv(uv, cid=pid)

            buv = to_numpy(buv)
            uv = to_numpy(uv)
            bxyz = to_numpy(bxyz)
            xyz = to_numpy(xyz)

            cdt_flag = True
            if cdt_flag:
                xyz = np.concatenate([bxyz, xyz], axis=0)
            _, faces = self.triangulate(buv, uv, cdt=cdt_flag)
            uv = np.concatenate([buv, uv], axis=0)

            mesh = trimesh.Trimesh(vertices=xyz, faces=faces)
            mesh.visual = trimesh.visual.TextureVisuals(uv=uv, material=None, image=texture_imgs[pid%9])
            mesh.export(f"{self.save_dir}/out_{epoch}_{pid}.obj")
            patchmeshes.append(mesh)

        scene = trimesh.Scene(patchmeshes)
        scene.export(f"{self.save_dir}/out_{epoch}.obj")


    def evaluate_param(self, epoch, sample_idx=None):
        assert sample_idx is not None
        self.optimizer.zero_grad()
        ###############################################
        
        patch_samples = self.dataset.patch_xyz
        patch_sample_normals = self.dataset.patch_normals
        patch_samples = torch.tensor(patch_samples[sample_idx,:], device=self.device, dtype=self.dtype)
        patch_sample_normals = torch.tensor(patch_sample_normals[sample_idx,:], device=self.device, dtype=self.dtype)
        x, nx, uniform_dgp, conformal_dgp, cycle_loss, uv_to_cid = self.net(sample_idx)

        cos_sim = 1 - torch.sum(nx*patch_sample_normals, dim=-1)
        return x.detach(), cos_sim.detach()
    
    ## TODO:
    # evaluate random normals