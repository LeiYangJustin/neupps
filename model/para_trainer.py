import numpy as np
from common_tools.io_tools import *
import os

import PIL.Image as Image
import triangle as tr
import trimesh 
import cv2
import numpy as np

import torch
from torch.utils.tensorboard import SummaryWriter
from torch import optim
import torch.nn.functional as F

from model.model import MultiNetGraph, Complex, QuadComplex
from model.dataset import ParamDataset, PointDataset

from common_tools.eig_decomposition import fft1d

def np_chamfer_distance_matching(x, y):
    x = to_numpy(x)
    y = to_numpy(y)
    dist = np.sum((x[:,None,...]-y[None,...])**2, axis=-1) ## N x M
    _, idx1 = np.min(dist, axis=0) ## M
    _, idx2 = np.min(dist, axis=1) ## N
    return idx1, idx2

def chamfer_distance_matching(x, y):
    N, _ = x.shape
    M, _ = y.shape
    with torch.no_grad():
        dist = torch.sum((x[:,None,...]-y[None,...])**2, dim=-1) ## N x M
        _, idx1 = torch.min(dist, dim=0) ## M
        _, idx2 = torch.min(dist, dim=1) ## N
    return idx1, idx2

def compute_distance(x, idx, y, idy):
    dist1 = vec_dist(x, y[idy])
    dist2 = vec_dist(x[idx], y)
    return (dist1 + dist2)/2

def vec_cosine(x, y, mean=True):
    if mean:
        return torch.mean(1 - torch.sum(x*y, dim=-1))
    else:
        return 1 - torch.sum(x*y, dim=-1)
    
def vec_cosine_signagnostic(x, y, mean=True):
    if mean:
        return torch.mean(1 - torch.abs(torch.sum(x*y, dim=-1)))
    else:
        return 1 - torch.sum(x*y, dim=-1)

def vec_dist_norm(x, y):
    return torch.mean(torch.norm(x-y, dim=-1))

def vec_dist(x, y):
    return F.l1_loss(x, y)

def vec_dist_per_vertex(x, y):
    return torch.norm(x-y, dim=-1)

def vec_tdm(x, y, ny, threshold=None):
    ## project dist between x and y to ny
    vec = x-y
    if threshold is None:
        return torch.mean(torch.abs(torch.sum(vec*ny, dim=-1)))
    else:
        return torch.mean(torch.abs(torch.sum(vec*ny, dim=-1)) < threshold).float()


class BaseBatchTrainer():
    def __init__(self) -> None:
        pass

    def save_checkpoint(self, epoch, state_dict=None):
        if state_dict is None:
            state_dict = {
                "epoch": epoch, 
                "model_state_dict": self.net.state_dict(),
                # "optimizer_state_dict": self.optimizer.state_dict()
                }
        else:
            state_dict = {
                **state_dict,
                "model_state_dict": self.net.state_dict(),
            }

        if hasattr(self, "latent_gen"):
            state_dict = {
                **state_dict, 
                "latent_gen_state_dict": self.latent_gen.state_dict()}
                          
        torch.save(state_dict, os.path.join(self.ckpt_dir, f'{epoch}.pth'))
        torch.save(state_dict, os.path.join(self.ckpt_dir, f'latest.pth')) 
    
    def show_boundary(self, epoch, name=None):
        bxx, _, _ = self.net.evaluate_boundary()
        bxx = to_numpy(bxx)

        ## visualize normals
        bxx, bnxx, _, _, _ = self.net.forward_boundary()
        bxx = to_numpy(bxx)
        bnxx = to_numpy(bnxx)
        offset_points = bxx + bnxx*0.01
        all_pts = np.stack([offset_points, bxx], axis=1)
        all_pts = all_pts.reshape(-1,3)
        links = [np.arange(0, len(all_pts), 2), np.arange(1, len(all_pts), 2)]
        links = np.stack(links, axis=-1)

        # print(bxx.shape)
        if name is None:
            savefile = f"{self.save_dir}/out_boundary_{epoch}.obj"
        else:
            savefile = f"{self.save_dir}/{name}_out_boundary_{epoch}.obj"
        # write_obj_file(savefile, bxx)
        write_line_file(savefile, all_pts, links)

    
    def check_grad_norm(self):
        trainable_parameters = []
        embedding = []
        for n, p in self.net.named_parameters():
            if n != 'nodal_embeddings.weight' and p.grad is not None and p.requires_grad:
                trainable_parameters.append(p)
            elif p.grad is not None and p.requires_grad:
                embedding.append(p)
        
        param_grad = 0.0
        if len(trainable_parameters) == 0:
            param_grad = 0.0
        else:
            device = trainable_parameters[0].grad.device
            param_grad = torch.mean(torch.stack([torch.norm(p.grad.detach()).to(device) for p in trainable_parameters]))
        
        if len(embedding) == 0:
            emb_grad = 0.0
        else:
            device = embedding[0].grad.device
            emb_grad = torch.mean(torch.stack([torch.norm(p.grad.detach()).to(device) for p in embedding]))
        return param_grad, emb_grad


    """
    buv: boundary uv
    uv: interior uv
    """
    def triangulate(self, buv, uv, cdt=True):
        if cdt:
            ## use constrained delaunay triangulation
            seg = np.stack([np.arange(buv.shape[0]), np.arange(buv.shape[0]) + 1], axis=1) % buv.shape[0]
            uv = np.concatenate([buv, uv], axis=0)
            uv_input = dict(vertices=uv, segments=seg)
            mesh = tr.triangulate(uv_input, 'p')
        else:
            uv_input = dict(vertices=uv,)
            mesh = tr.triangulate(uv_input)
        return mesh['vertices'], mesh['triangles']

    def show_patches(self, epoch, name=None, mesh_res=50):

        # texture_img = Image.open(f'./asset/checkerboard.png')
        texture_imgs = []
        for i in range(1, 10):
            texture_imgs.append(Image.open(f'./asset/checkboard/Slide{i}.jpg'))
        
        meshes = []
        
        xyz, uv, uv_to_cid = self.net.evaluate_patch()
        xyz = to_numpy(xyz)
        uv = to_numpy(uv)
        uv_to_cid = to_numpy(uv_to_cid)

        bxyz, buv, buv_to_cid = self.net.evaluate_boundary()
        bxyz = to_numpy(bxyz)
        buv = to_numpy(buv)
        buv_to_cid = to_numpy(buv_to_cid.squeeze())

        for cid in range(self.num_patches):
            mask = uv_to_cid == cid
            uv_cid = uv[mask]
            xyz_cid = xyz[mask]

            bmask = buv_to_cid == cid
            buv_cid = buv[bmask]
            bxyz_cid = bxyz[bmask]

            cdt_flag = True
            if cdt_flag:
                xyz_cid = np.concatenate([bxyz_cid, xyz_cid], axis=0)
                uv_all = np.concatenate([buv_cid, uv_cid], axis=0)
            else:
                uv_all = uv_cid
                buv_cid = None

            _, faces = self.triangulate(buv_cid, uv_cid, cdt=cdt_flag)
            # flat = trimesh.Trimesh(vertices=np.concatenate([uv_all, np.zeros((uv_all.shape[0], 1))], axis=-1), faces=faces, process=False, maintain_order=True)
            # flat.visual = trimesh.visual.TextureVisuals(uv=uv_all, material=None, image=texture_img)
            # flat.export(f"{self.save_dir}/flat_{epoch}_{cid}.obj")

            # ## stripes using tab20
            # cm_uv = np.ones_like(uv)
            # cm_uv[:,0] = (cid%20+0.5)/20
            # os.makedirs(f"{self.save_dir}/{cid}", exist_ok=True)
            if len(self.dataset.patch_textures) > 0:
                
                uv_all = (uv_all + 1) / 2
                uv_visuals = trimesh.visual.texture.TextureVisuals(
                    uv=uv_all, 
                    image=self.dataset.patch_textures[cid]
                )

                mesh = trimesh.Trimesh(vertices=xyz_cid, faces=faces, visual=uv_visuals)

                h, w = self.dataset.patch_textures[cid].size
                img = np.ones((h, w, 3), dtype=np.uint8) * 255
                for pid in range(len(uv_all)):
                    cv2.circle(img, tuple((uv_all[pid] * [w, h]).astype(np.int32)), 5, (0,0,255), -1)
                cv2.imwrite(os.path.join(self.save_dir, str(cid), f"debug_uv.png"), img)
            else:
                mesh = trimesh.Trimesh(vertices=xyz_cid, faces=faces, process=False, maintain_order=True)
                mesh.visual = trimesh.visual.TextureVisuals(uv=uv_all, material=None, image=texture_imgs[cid%9])
            
            """ use color map instead of texture"""
            # norm = matplotlib.colors.Normalize(0, 20, clip=True)
            # mapper = cm.ScalarMappable(norm=norm, cmap=cm.tab20)
            # mesh.visual.face_colors = mapper.to_rgba(cid%20)
            meshes.append(mesh)
            mesh.export(f"{self.save_dir}/out_{epoch}_{cid}.obj")

        scene = trimesh.Scene(meshes)
        if name is None:
            scene.export(f"{self.save_dir}/out_{epoch}.obj")
        else:
            scene.export(f"{self.save_dir}/{name}_out_{epoch}.obj")


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

class SingleTrainerParam(BaseBatchTrainer):
    def __init__(self, cfg):
        super(SingleTrainerParam).__init__()

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

        self.dataset = ParamDataset(cfg['train_data'], batchsize=cfg['batchsize'], mode=cfg['mode'])
        if cfg['train_method'] == 'no_param':
            self.dataset.sample_for_point_cloud(cfg['add_samples'])
        else:
            self.dataset.sample(add_samples=cfg['add_samples'])

        self.num_patches = len(self.dataset.graph_cells)
        self.num_nodes = self.dataset.num_nodes
        
        """
        It should be Polygonal Complex
        """
        if not "quadmesh" in cfg.keys() or not cfg["quadmesh"]:
            print("Using Complex")
            self.complex = Complex(
                # node_ids = self.dataset.node_ids,
                num_nodes = self.num_nodes,
                cells = self.dataset.graph_cells,
                uvs = self.dataset.patch_uv,
                uv_to_cid = self.dataset.patch_ids,
                degs_list = self.dataset.degs_list,
                device = self.device, 
                dtype = self.dtype,
                require_grad = self.mode)
        else:
            print("Using QuadComplex")
            self.complex = QuadComplex(
                num_nodes= self.num_nodes,
                cells = self.dataset.graph_cells,
                uvs = self.dataset.patch_uv,
                uv_to_cid = self.dataset.patch_ids,
                device = self.device, 
                dtype = self.dtype,
                require_grad = self.mode)

        
        self.complex.add_boundary_uvs(self.dataset.boundary_samples_uv, self.dataset.boundary_sample_pids)

        self.use_duv = cfg['net_params']['use_duv']
        self.net = MultiNetGraph(
            complex=self.complex,
            device=self.device,
            dtype=self.dtype,
            **cfg['net_params']
        )
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
        emb_optimizer = optim.Adam(embedding, lr=0.008, amsgrad=False)
        emb_scheduler1 = optim.lr_scheduler.LinearLR(emb_optimizer, start_factor=0.01, end_factor=1, total_iters=1000)
        emb_scheduler2 = optim.lr_scheduler.CosineAnnealingLR(emb_optimizer, T_max=self.max_epoch, eta_min=0.00001)
        emb_scheduler = optim.lr_scheduler.SequentialLR(emb_optimizer, schedulers=[emb_scheduler1, emb_scheduler2], milestones=[1000])

        nn_optimizer = optim.Adam(trainable_parameters, lr=0.008, amsgrad=False) ## need to be a bit larger than emb_optimizer lr
        nn_scheduler1 = optim.lr_scheduler.LinearLR(nn_optimizer, start_factor=0.01, end_factor=1, total_iters=1000)
        nn_scheduler2 = optim.lr_scheduler.CosineAnnealingLR(nn_optimizer, T_max=self.max_epoch, eta_min=0.00001)
        nn_scheduler = optim.lr_scheduler.SequentialLR(nn_optimizer, schedulers=[nn_scheduler1, nn_scheduler2], milestones=[1000])                    
        self.opt = OptimizerWrapper([emb_optimizer, nn_optimizer], [emb_scheduler, nn_scheduler])
            

        self.start_epoch = 0
        if resume_path is not None:
            ckpt_path = os.path.join(resume_path, f'ckpt/{checkpoint}.pth')
            state_dict = torch.load(ckpt_path, map_location=self.device)
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

        print("Trainer init")


    ## 
    def compute_boundary_and_normal(self):
        bxx, _, _ = self.net.forward_boundary()
        bxx = bxx.reshape(-1, self.complex.boundary_sample_rate, 3)
        return bxx, None

    def loss_fn(self, loss_dict):
        
        corner_loss = loss_dict['corner_loss']
        surf_loss = loss_dict['surf_loss']
        surf_tqm_loss = loss_dict['tdm_loss']
        normal_loss = loss_dict['normal_loss']
        boundary_loss = loss_dict['boundary_loss']
        u_loss = loss_dict['u_loss']
        v_loss = loss_dict['v_loss']
        iso_loss = loss_dict['iso_loss']

        # smooth_boundary_loss = loss_dict['smooth_boundary_loss']
        # b_normal_constraint = loss_dict['smooth_boundary_normal']

        surf_loss_weight = self.loss_params['alpha']
        normal_loss_weight = self.loss_params['normal']
        boundary_weight = self.loss_params['boundary']
        # smooth_boundary_loss_weight = self.loss_params['smooth_boundary']

        ######################################################
        epoch = loss_dict['epoch']

        loss = 0.0
        loss += surf_loss_weight*surf_loss

        if epoch > 5000:
            loss += normal_loss_weight*normal_loss

        loss += boundary_weight*boundary_loss
        loss += corner_loss

        if self.use_duv:        
            loss += normal_loss_weight*u_loss
            loss += normal_loss_weight*v_loss

        return loss
    


    def evaluate(self, epoch, sample_idx=None):
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
        if self.train_method == 'param':
            loss, loss_dict = self.train_iteration_param(epoch, sample_idx)
        elif self.train_method == 'hybrid':
            if epoch > 5000:
                loss, loss_dict = self.train_iteration_match(epoch, sample_idx)
            else:
                loss, loss_dict = self.train_iteration_param(epoch, sample_idx)       
        else:
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

        ## if v is nan exit()
        if loss.item() != loss.item():
            print(f"loss is nan")
            exit()

        return {"loss": loss.item(), **loss_dict, "param_grad": param_grad, "emb_grad": emb_grad}
    

    def train_iteration_param(self, epoch:int, sample_idx=None, added_data=None):
        assert sample_idx is not None
        ###############################################
        
        # corners = data['corners']
        corners = self.dataset.corners
        corners = torch.tensor(corners, device=self.device, dtype=self.dtype)

        boundary_samples_uv = self.dataset.data['boundary_samples_uv']
        boundary_samples = self.dataset.data['boundary_samples']
        boundary_samples = torch.tensor(boundary_samples, device=self.device, dtype=self.dtype)

        mask = self.dataset.data['mask'][sample_idx]
        mask = torch.tensor(mask, device=self.device, dtype=torch.bool)

        dx_du_gt = self.dataset.data['grad'][sample_idx]
        dx_du_gt = torch.tensor(dx_du_gt, device=self.device, dtype=self.dtype)
        dx_du_gt = dx_du_gt[mask] ## select valid

        patch_samples = self.dataset.patch_xyz
        patch_samples_uv = self.dataset.patch_uv
        patch_sample_normals = self.dataset.patch_normals
        
        patch_samples = torch.tensor(patch_samples[sample_idx,:], device=self.device, dtype=self.dtype)
        patch_sample_normals = torch.tensor(patch_sample_normals[sample_idx,:], device=self.device, dtype=self.dtype)
        patch_samples_uv = torch.tensor(patch_samples_uv[sample_idx,:], device=self.device, dtype=self.dtype)
        
        x, nx, uv_to_cid, reg_loss = self.net(sample_idx)
        pids = self.dataset.patch_ids.squeeze()
        pids = pids[sample_idx]

        per_vert_dist = vec_dist_per_vertex(x, patch_samples)

        K = int(0.01*len(patch_samples))
        topk_err = torch.topk(per_vert_dist, k=K, largest=True)
        max_surf_loss = torch.mean(topk_err.values)
        
        surf_loss = per_vert_dist.mean()
        surf_tqm_loss = vec_tdm(x, patch_samples, patch_sample_normals)

        normal_loss = vec_dist(nx, patch_sample_normals)
        dx_du = self.net.get_data()
        dx_du = dx_du[mask]
        u_loss = vec_cosine(dx_du[:,:,0], dx_du_gt[:,:,0])
        v_loss = vec_cosine(dx_du[:,:,1], dx_du_gt[:,:,1])

        x_crns = self.net.forward_corners()
        corner_loss = vec_dist(x_crns, corners)

        boundary_loss = 0.0
        ## another way to compute boundary loss
        bxx, _, buv = self.net.forward_boundary()
        boundary_loss = vec_dist(bxx, boundary_samples)

        """
        smooth boundary loss applies laplacian smoothing on the boundary points
        """

        loss_dict = {
            "corner_loss": corner_loss,
            "surf_loss": surf_loss,
            "normal_loss": normal_loss,
            "boundary_loss": boundary_loss,
            "tdm_loss": surf_tqm_loss,
            "max_surf_loss": max_surf_loss,
            "u_loss": u_loss,
            "v_loss": v_loss,
            "conformal_loss": reg_loss,
            "reg_loss": reg_loss,
            "iso_loss": reg_loss,
            'epoch': epoch,
        }
        loss = self.loss_fn(loss_dict)

        return loss, loss_dict
    

    def train_iteration_match(self,  epoch:int,  sample_idx:list=None):
        
        corners = self.dataset.corners
        corners = torch.tensor(corners, device=self.device, dtype=self.dtype)

        boundary_samples_uv = self.dataset.data['boundary_samples_uv']
        boundary_samples = self.dataset.data['boundary_samples']
        boundary_samples = torch.tensor(boundary_samples, device=self.device, dtype=self.dtype)

        mask = self.dataset.data['mask'][sample_idx]
        mask = torch.tensor(mask, device=self.device, dtype=torch.bool)

        dx_du_gt = self.dataset.data['grad'][sample_idx]
        dx_du_gt = torch.tensor(dx_du_gt, device=self.device, dtype=self.dtype)
        dx_du_gt = dx_du_gt[mask] ## select valid

        patch_samples = self.dataset.patch_xyz
        patch_samples_uv = self.dataset.patch_uv
        patch_sample_normals = self.dataset.patch_normals
        
        random_idx = np.random.choice(len(patch_samples), int(35000), replace=False)
        patch_samples = torch.tensor(patch_samples[random_idx,:], device=self.device, dtype=self.dtype)
        patch_sample_normals = torch.tensor(patch_sample_normals[random_idx,:], device=self.device, dtype=self.dtype)
        patch_samples_uv = torch.tensor(patch_samples_uv[random_idx,:], device=self.device, dtype=self.dtype)
        
        """
        get the differential goemetric regularization losses
        """
        x, nx, uv_to_cid, iso_loss, conformal_loss, cycle_loss, reg_loss = self.net.forward_dgp(sample_idx)

        surf_loss = 0.0
        normal_loss = 0.0
        surf_tqm_loss = 0.0

        dist_matrix = torch.cdist(x.detach(), patch_samples, p=2)
        idx = torch.argmin(dist_matrix, axis=1)
        idx2 = torch.argmin(dist_matrix, axis=0)

        dist = vec_dist(x, patch_samples[idx])
        ndist = vec_dist(nx, patch_sample_normals[idx])
        tdm = vec_tdm(x, patch_samples[idx], patch_sample_normals[idx])

        dist2 = vec_dist(x[idx2], patch_samples)
        ndist2 = vec_dist(nx[idx2], patch_sample_normals)
        tdm2 = vec_tdm(x[idx2], patch_samples, patch_sample_normals)

        """
        This is not robust enough
        You need to tune the weights between dist and dist2 carefully
        This is because the number of samples in the data and that in the neural surface are different
        Different ratio will lead to different balance between dist and dist2
        """
        surf_loss += dist + 10*dist2
        # surf_loss += dist + dist2
        normal_loss += ndist + ndist2
        surf_tqm_loss += tdm + tdm2

        surf_loss /= self.num_patches
        normal_loss /= self.num_patches
        surf_tqm_loss /= self.num_patches

        x_crns = self.net.forward_corners()
        corner_loss = vec_dist(x_crns, corners)

        boundary_loss = 0.0
        ## another way to compute boundary loss
        if True:
            bxx, _, _ = self.net.forward_boundary()
            boundary_loss = vec_dist(bxx, boundary_samples)

        loss_dict = {
            "corner_loss": corner_loss,
            "surf_loss": surf_loss,
            "normal_loss": normal_loss,
            "boundary_loss": boundary_loss,
            "tdm_loss": surf_tqm_loss,
            "conformal_loss": conformal_loss,
            "iso_loss": iso_loss,
            "cycle_loss": cycle_loss,
            "reg_loss": reg_loss,
            "epoch": epoch,
        }
        
        corner_loss = loss_dict['corner_loss']
        surf_loss = loss_dict['surf_loss']
        surf_tqm_loss = loss_dict['tdm_loss']
        boundary_loss = loss_dict['boundary_loss']
        iso_loss = loss_dict['iso_loss']

        # smooth_boundary_loss = loss_dict['smooth_boundary_loss']
        # b_normal_constraint = loss_dict['smooth_boundary_normal']

        surf_loss_weight = self.loss_params['alpha']
        normal_loss_weight = self.loss_params['normal']
        conformal_loss_weight = self.loss_params['conformal']
        boundary_weight = self.loss_params['boundary']
        # smooth_boundary_loss_weight = self.loss_params['smooth_boundary']

        ######################################################
        epoch = loss_dict['epoch']
        # scale = min(epoch/3000.0, 1.0)

        loss = 0.0
        loss += boundary_weight*boundary_loss
        loss += surf_loss_weight*corner_loss

        if epoch > 0:
            loss += conformal_loss_weight*conformal_loss
            # loss += uniform_loss_weight*uniform_loss
            # loss += cycle_loss_weight*cycle_loss

        ## when working with slender shape, we need to first fit the boundary
        if epoch > 1000:
            loss += surf_loss_weight*surf_loss
            loss += normal_loss_weight*normal_loss

        if epoch > 0:
            loss += 0.3*iso_loss
        elif epoch > 5000:
            loss += 0.1*iso_loss
        elif epoch > 8000:
            loss += 0.001*iso_loss

        return loss, loss_dict


    def extract_mesh(self, epoch, mesh_res=50):
        patchmeshes = self.show_patches(epoch=epoch, name="out", mesh_res=mesh_res)
        return patchmeshes
        

    def extract_mesh_with_uv(self, epoch):
        self.patch_list = self.dataset.patch_list
        # texture_img = Image.open(f'./asset/cm_tab20_v.png')
        # texture_img = Image.open(f'./asset/checkerboard.png')

        texture_imgs = []
        for i in range(1, 10):
            # texture_imgs.append(Image.open(f'./asset/checkboard/Slide{i}.jpg'))
            texture_imgs.append(Image.open(f'./asset/checkerboard.png'))
        
        patchmeshes = []
        for cid, patchfile in enumerate(self.patch_list):
            _, uv, faces, _ = read_textured_obj_file(patchfile)
            uv = torch.tensor(uv, device=self.device, dtype=self.dtype)
            xyz, _ = self.net.forward_uv(uv, cid=cid)
            xyz = to_numpy(xyz)
            patchmesh = trimesh.Trimesh(vertices=xyz, faces=faces, process=False, maintain_order=True)

            cm_uv = to_numpy(uv)
            ## texture
            # cm_uv = np.ones_like(to_numpy(uv))
            # cm_uv[:,1] = (cid%20+0.5)/20
            # patchmesh.visual = trimesh.visual.TextureVisuals(uv=cm_uv, material=None, image=texture_img)
            patchmesh.visual = trimesh.visual.TextureVisuals(uv=cm_uv, material=None, image=texture_imgs[cid%9])
            patchmeshes.append(patchmesh)
            
        scene = trimesh.Scene(patchmeshes)
        scene.export(f"{self.save_dir}/out_{epoch}.obj")

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

    def extract_mesh_remeshed(self, epoch):
        # texture_img = Image.open(f'./asset/checkerboard.png')

        texture_imgs = []
        for i in range(1, 10):
            texture_imgs.append(Image.open(f'./asset/checkboard/Slide{i}.jpg'))

        patchmeshes = []
        uv_list = self.complex.interior_sampling()
        buv_list = self.complex.boundary_sampling()
        for pid, _ in enumerate(self.dataset.patch_list):
            
            ## sample buv
            buv = buv_list[pid]
            bxyz, _ = self.net.forward_uv(buv, cid=pid)

            ## sample uv
            uv = uv_list[pid]
            xyz, _ = self.net.forward_uv(uv, cid=pid)

            buv = to_numpy(buv)
            bxyz = to_numpy(bxyz)
            uv = to_numpy(uv)
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