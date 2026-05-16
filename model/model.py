
import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import grad

import torch.nn as nn   
from common_tools.io_tools import *
from copy import deepcopy
from torch_batch_svd import svd
from tqdm import tqdm

def grad_compute(inputs, outputs, dtype=torch.float):
        d_points = torch.ones_like(outputs, requires_grad=False, device=inputs.device, dtype=dtype)
        ori_grad = grad(
            outputs=outputs,
            inputs=inputs,
            grad_outputs=d_points,  
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
            allow_unused=False
        )
        points_grad = ori_grad[0]
        return points_grad

class PositionalEncoding(nn.Module):
    def __init__(self, dim, feat_dim_in):
        super().__init__()
        self.dim = dim
        self.pi = np.pi
        self.feat_dim_in = feat_dim_in
        self.feat_dim_out = self.feat_dim_in*self.dim*2 ## *2 == sin, cos
        self.scale = 0.1

    def forward(self, x, scale=None, on=True):
        if self.dim == 0:
            return x

        if scale is None:
            scale = self.scale

        B = x.shape[0]
        x1 = x[:, :-self.feat_dim_in]
        x2 = x[:, -self.feat_dim_in:]
        xpe = []

        ## use PE or not
        if on:
            for i in range(1, self.dim+1):
                # xpe.append(scale*torch.sin(self.pi*2*float(i)*x1)*torch.cos(self.pi*0.5*x1))
                # xpe.append(scale*torch.cos(self.pi*2*float(i)*x1)*torch.cos(self.pi*0.5*x1))
                xpe.append(scale*torch.sin(self.pi*2*float(i)*x2))
                xpe.append(scale*torch.cos(self.pi*2*float(i)*x2))
        else:
            for i in range(1, self.dim+1):
                xpe.append(torch.zeros_like(x2))
                xpe.append(torch.zeros_like(x2))

        xpe = torch.stack(xpe, dim=-1)
        xpe = xpe.reshape(B, -1) ## *2 == sin, cos
        xpe = torch.cat([x1, xpe], dim=-1)
        return xpe

class Complex():
    def __init__(self, num_nodes, cells, uvs, uv_to_cid, degs_list, device, require_grad=False, dtype=torch.float32):
        self.device = device
        self.eps = 1e-6
        self.require_grad=require_grad
        self.boundary_sample_rate = 50

        self.dtype = dtype

        ## corner indices
        self.num_nodes = num_nodes
        self.cells = cells

        print("Number of nodes: ", self.num_nodes)
        print("Number of cells: ", len(self.cells))
        print("degs_list: ", len(degs_list))

        ## cell corners; start from 0 degree
        self.cell_corners = []
        for i, degs in enumerate(degs_list):
            degs = 2*np.pi*np.array(np.cumsum(degs))
            degs = torch.from_numpy(degs).to(self.dtype)[:len(self.cells[i])] ## remove cyclic last one
            x = torch.cos(degs)
            y = torch.sin(degs)
            x = x.to(dtype=self.dtype, device=self.device)
            y = y.to(dtype=self.dtype, device=self.device)
            corners = torch.stack([x,y], dim=-1)
            self.cell_corners.append(corners)

        ## uv
        self.uvs = torch.tensor(uvs, device=self.device, dtype=self.dtype)
        self.uv_to_cid = torch.LongTensor(uv_to_cid).to(self.device).squeeze()

        ###############################################
        ## mv_coords and g of mesh uv
        self.all_mv_coords = []
        self.all_g = []
        ## eval
        self.eval_mv_coords = []
        self.eval_uv = []
        self.eval_uv_to_cid = []
        ## boundary
        self.boundary_mv_coords = []
        self.boundary_uv = []
        self.boundary_uvpe = []
        self.boundary_uv_to_cid = []
        self.boundary_g = []
        ## mv_coords and g of random sampling
        self.patch_random_sample_size = 300
        
        self.rand_mv_coords = []
        self.rand_uv = []
        self.rand_g = []
        self.rand_uv_to_cid = []
        self.rand_all_uvpe = []
        
        for cell_idx, crns in tqdm(enumerate(self.cell_corners)):
            nids = self.cells[cell_idx] ## node ids
            if True:
                uvmask = self.uv_to_cid == cell_idx
                # print(self.uv_to_cid.shape, cell_idx, uvmask.sum(), nids)
                # input()
                uv = self.uvs[uvmask]
                uv_split = torch.split(uv, 20000, dim=0)
                mv_coords = []
                g = []
                for uv_ in tqdm(uv_split):
                    mvc_, grad_ = self.query_robust(uv_, crns, nids, store_grad=require_grad)
                    mv_coords.append(mvc_)
                    if require_grad:
                        g.append(grad_)
                # mv_coords, g = self.query_robust(uv, crns, nids, store_grad=require_grad)
                # mv_coords, g = self.query(uv, crns, nids, store_grad=True)
                mv_coords = torch.cat(mv_coords, dim=0)
                self.all_mv_coords.append(mv_coords.detach())
                if g is not None and len(g) > 0:
                    g = torch.cat(g, dim=0)
                    self.all_g.append(g.detach())
            else:
                uv = self.uvs[cell_idx]
                mv_coords, g = self.query(uv, crns, nids, store_grad=True)
                self.all_mv_coords.append(mv_coords.detach())
                self.all_g.append(g.detach())
                self.uv_to_cid.append(torch.ones(uv.shape[0], dtype=torch.long, device=self.device)*cell_idx)

            # ## boundary
            # b_mvc, b_g, b_uv = self.query_robust_boundary(crns, nids, store_grad=require_grad)
            b_mvc, b_uv = self.query_boundary(cell_idx)
            b_g = self.query_boundary_normals(b_uv, crns, nids)
            self.boundary_mv_coords.append(b_mvc.detach())
            self.boundary_uv.append(b_uv.detach())
            self.boundary_uv_to_cid.append(torch.ones(b_uv.shape[0], dtype=torch.long, device=self.device)*cell_idx)
            if b_g is not None:
                self.boundary_g.append(b_g.detach())
    
        self.all_mv_coords = torch.cat(self.all_mv_coords, dim=0)
        if len(self.all_g) > 0:
            self.all_g = torch.cat(self.all_g, dim=0)
        if isinstance(self.uv_to_cid, list):
            self.uv_to_cid = torch.cat(self.uv_to_cid, dim=0)

        self.boundary_mv_coords = torch.cat(self.boundary_mv_coords, dim=0).to(device='cpu')
        self.boundary_uv = torch.cat(self.boundary_uv, dim=0).to(device='cpu')
        self.boundary_uv_to_cid = torch.cat(self.boundary_uv_to_cid, dim=0).to(device='cpu')
        if len(self.boundary_g) > 0:
            self.boundary_g = torch.cat(self.boundary_g, dim=0).to(device='cpu')

    def interior_sampling(self, sample_size=None):
        if sample_size is None:
            sample_size = self.patch_random_sample_size
        rand_uv_list = []
        for cell_idx, crns in tqdm(enumerate(self.cell_corners)):
            rand_uv = self.pre_compute_uv_grid(crns, sample_size)
            rand_uv_list.append(rand_uv)    
        return rand_uv_list
    

    def boundary_sampling(self, sample_size=100):
        boundary_uv_list = []
        for cell_idx, crns in tqdm(enumerate(self.cell_corners)):
            b_mvc, b_uv = self.query_boundary(cell_idx, boundary_sample_rate=sample_size)
            # b_g = self.query_boundary_normals(b_uv, crns, nids)
            # self.boundary_mv_coords.append(b_mvc.detach())
            boundary_uv_list.append(b_uv)
        return boundary_uv_list
    

    def _release(self):
        self.all_g = None
        self.all_mv_coords = None
        self.uv_to_cid = None


    def get_ridx(self, iteration=0, sample_size=None):
        if self.ridx is None:
            self.ridx = torch.randperm(self.rand_sample_size)
            self.ridx_chunks = torch.chunk(self.ridx, self.rand_sample_size//sample_size)

        section = iteration % len(self.ridx_chunks)
        return self.ridx_chunks[section]


    def add_uvs(self, uvs, uv_to_cid):
        all_mv_coords = []
        all_g = []
        for cell_idx, crns in enumerate(self.cell_corners):
            nids = self.cells[cell_idx] ## node ids
            uvmask = uv_to_cid == cell_idx
            uv = uvs[uvmask]
            mv_coords, g = self.query_robust(uv, crns, nids, store_grad=self.require_grad)
            all_mv_coords.append(mv_coords.detach())
            if g is not None:
                all_g.append(g.detach())

        all_mv_coords = torch.cat(all_mv_coords, dim=0).to(device='cpu')
        self.all_mv_coords = torch.cat([self.all_mv_coords, all_mv_coords], dim=0)
        self.uv_to_cid = torch.cat([self.uv_to_cid, uv_to_cid], dim=0)

        if len(all_g) > 0:
            all_g = torch.cat(all_g, dim=0).to(device='cpu')
            self.all_g = torch.cat([self.all_g, all_g], dim=0)

    def add_boundary_uvs(self, buvs, buv_to_cid):
        b_mv_coords = []
        boundary_uvs = []
        boundary_uv_to_cid = []

        print("buv_to_cid: ", buv_to_cid.shape)
        print("buv: ", buvs.shape)

        for cell_idx, crns in enumerate(self.cell_corners):
            nids = self.cells[cell_idx]
            uvmask = buv_to_cid == cell_idx
            buv = buvs[uvmask]
            boundary_uvs.append(buv)
            boundary_uv_to_cid.append(buv_to_cid[uvmask])

            b_mvc = self.query_boundary2(cell_idx, buv)
            b_mv_coords.append(b_mvc.detach())

        b_mv_coords = torch.cat(b_mv_coords, dim=0)
        self.boundary_mv_coords = b_mv_coords

        boundary_uvs = np.concatenate(boundary_uvs, axis=0)
        self.boundary_uv = torch.tensor(boundary_uvs, device=self.device, dtype=self.dtype)

        boundary_uv_to_cid = np.concatenate(boundary_uv_to_cid, axis=0)        
        self.boundary_uv_to_cid = torch.tensor(boundary_uv_to_cid, device=self.device, dtype=torch.long)
        


    ##
    def query(self, uv, crns, nids, store_grad=False):
        if store_grad:
            uv.requires_grad = True

        vec = uv[:,None,:] - crns[None,:,:]
        d = torch.linalg.norm(vec, dim=-1)+self.eps ## distance
        degree = crns.shape[0]

        ## cos a = (va^T*vb) / (|va|*|vb|)
        shifted_indices = torch.arange(-1, degree-1)
        cos_alpha = (vec[:,shifted_indices]*vec).sum(dim=-1) 
        cos_alpha = cos_alpha / (d[:,shifted_indices]*d) ## normalized
        find_nan(cos_alpha, "cos_alpha")

        ## tan(a/2) = sqrt((1-cos(a)) / (1+cos(a)))
        tan_half_alpha = torch.sqrt((1-cos_alpha)/(1+cos_alpha))
        find_nan(tan_half_alpha, "tan_half_alpha")

        ## MVC
        shifted_indices = torch.arange(1, degree+1) % degree
        weight = (tan_half_alpha[:,shifted_indices] + tan_half_alpha) / d
        weight = weight/torch.sum(weight, dim=-1, keepdims=True)

        out_weight = torch.zeros(uv.shape[0], self.num_nodes).to(self.device)

        ## scatter weight into out_weight according to the one hot vector of corners
        out_weight[:, nids] = weight
        find_nan(out_weight, "out_weight")

        g = None
        if store_grad:
            g = [grad_compute(uv, out_weight[:,j], dtype=self.dtype) for j in range(self.num_nodes)]
            g = torch.stack(g, dim=1)
            find_nan(g, "g")

        return out_weight, g
    
    # """
    # make it differentiable
    # """
    def query_boundary(self, cid=0, boundary_sample_rate=None):
        crns = self.cell_corners[cid]
        nids = self.cells[cid]
        if boundary_sample_rate is None:
            boundary_sample_rate = self.boundary_sample_rate

        t = torch.linspace(0.0, 1.0, boundary_sample_rate+1).reshape(-1) ## include the end
        t = t[:-1]

        degree = crns.shape[0]

        b_mvcs = []
        for i in range(degree):
            b_mvc = torch.zeros(boundary_sample_rate, degree)
            b_mvc[:, i] = 1-t
            b_mvc[:, (i + 1) % degree] = t
            b_mvcs.append(b_mvc)
        b_mvcs = torch.cat(b_mvcs, dim=0).to(self.device)
        buv = b_mvcs @ crns
        out_weight = torch.zeros(b_mvcs.shape[0], self.num_nodes).to(self.device)
        ## scatter weight into out_weight according to the one hot vector of corners
        out_weight[:, nids] = b_mvcs
        return out_weight, buv
    
    def query_boundary2(self, cid, buv):
        crns = self.cell_corners[cid]
        nids = self.cells[cid]
        boundary_sample_rate = 50000
        t = torch.linspace(0.0, 1.0, boundary_sample_rate+1).reshape(-1) ## include the end
        t = t[:-1]
    
        degree = crns.shape[0]
        b_mvcs = []
        for i in range(degree):
            b_mvc = torch.zeros(boundary_sample_rate, degree, dtype=self.dtype)
            b_mvc[:, i] = 1-t
            b_mvc[:, (i + 1) % degree] = t
            b_mvcs.append(b_mvc)
        b_mvcs = torch.cat(b_mvcs, dim=0)
        base_buv = b_mvcs @ crns.to(device='cpu')

        ## make a knn tree
        from scipy.spatial import KDTree
        tree = KDTree(to_numpy(base_buv))
        dist, idx = tree.query(buv, k=1)
        # print(f"query dist stats -- max: {dist.max()}, mean: {dist.mean()}")

        b_mvcs = b_mvcs[idx].to(self.device)
        out_weight = torch.zeros(b_mvcs.shape[0], self.num_nodes, dtype=self.dtype).to(self.device)
        ## scatter weight into out_weight according to the one hot vector of corners
        out_weight[:, nids] = b_mvcs
        return out_weight
    
    
    ## d1,d2 vectors
    def compute_weight(self, d1, d2, ops):
        assert ops in ['add', 'sub']
        norm_d1 = torch.linalg.norm(d1, dim=-1)
        norm_d2 = torch.linalg.norm(d2, dim=-1)
        inner_prod = torch.sum(d1*d2, dim=-1)

        if ops == 'add':
            out_Sq = norm_d1*norm_d2+inner_prod
            out_Sq[out_Sq<0] = out_Sq[out_Sq<0] - out_Sq[out_Sq<0].detach() ## reparameterization to avoid sqrt(negative)
            out = torch.sqrt(out_Sq)
            find_nan(out, "add")
            return out
            # return torch.nan_to_num(torch.sqrt(norm_d1*norm_d2 + inner_prod), nan=0.0)
        
        out_Sq = norm_d1*norm_d2-inner_prod
        out_Sq[out_Sq<0] = out_Sq[out_Sq<0] - out_Sq[out_Sq<0].detach() ## reparameterization
        out = torch.sqrt(out_Sq)
        find_nan(out, "sub")
        return out

    def query_robust(self, uv, crns, nids, store_grad=False):
        degree = len(nids)
        
        ## compute gradients
        if store_grad:
            uv = uv.requires_grad_(True)

        ## B, N, 2
        d = crns[None,...] - uv[:,None,:]
        mvc = torch.zeros_like(d[:,:,0], dtype=self.dtype)

        """
        Wachspress and mean value coordinates
        Global form, Section 4.2
        """
        ## for each component
        for k in range(degree):
            w = torch.zeros_like(d[:,:,0])
            for i in range(degree):
                w_i = 1
                for j in range(degree):
                    if j == i:
                        out = self.compute_weight(
                           d[:,(j-1)%degree,:], 
                           d[:,(j+1)%degree,:], 
                           'sub') 
                        # find_nan(out, "out_sub")
                        # out = torch.nan_to_num(out, nan=0.0)
                        w_i *= out
                    elif j == (i-1)%degree:
                        continue
                    else:
                        out = self.compute_weight(
                            d[:,j,:],
                            d[:,(j+1)%degree,:],
                            'add')
                        # find_nan(out, "out_add")
                        # out = torch.nan_to_num(out, nan=0.0)
                        w_i *= out
                w[:,i] = w_i
            sum_w = torch.sum(w, dim=-1)
            sum_w = torch.clamp(sum_w, min=1e-8) ## clip
            ww = w[:,k] / sum_w
            mvc[:,k] = ww
        
        ## scatter weight into out_weight according to the one hot vector of corners
        out_weight = torch.zeros(mvc.shape[0], self.num_nodes, dtype=self.dtype, device=self.device)
        out_weight[:, nids] = mvc

        ## compute gradients
        out_g = None
        if store_grad:
            g = [grad_compute(uv, mvc[:,j], dtype=self.dtype).detach() for j in range(degree)]
            g = torch.stack(g, dim=1)
            g = torch.nan_to_num(g, nan=0.0)
            ## scatter g into out_g according to the one hot vector of corners
            out_g = torch.zeros(g.shape[0], self.num_nodes, 2, dtype=self.dtype, device=self.device)
            out_g[:, nids, :] = g

        return out_weight, out_g


    def query_robust_boundary(self, crns, nids, store_grad=False):
        degree = len(nids)
        ## make boundary parmaeters
        t = torch.linspace(0, 1, self.boundary_sample_rate+1).to(self.device)
        t = t[:-1]
        tinv = 1-t
        t_matrix = torch.zeros((self.boundary_sample_rate*degree, degree)).to(self.device)
        for i in range(degree):
            starting = i*self.boundary_sample_rate
            ending = (i+1)*self.boundary_sample_rate
            t_matrix[starting:ending, i%degree] = tinv
            t_matrix[starting:ending, (i+1)%degree] = t

        ## x: shape (sample_rate, 2)
        x = (t_matrix[...,None]*crns[None,:]).sum(dim=-2)
        ## compute gradients
        if store_grad:
            x = x.requires_grad_(True)
        return *self.query_robust(x, crns, nids, store_grad=store_grad), x


    def query_boundary_normals(self, uv, crns, nids):
        _, b_g = self.query(uv, crns, nids, store_grad=True)
        return b_g


    def pre_compute_uv_grid(self, crns, sample_size: int):
        x = torch.linspace(-1, 1, sample_size)
        y = torch.linspace(-1, 1, sample_size)
        xy = torch.stack(torch.meshgrid(x, y), dim=-1).reshape(-1, 2)
        xy = xy.to(crns.device)
        xy = xy + torch.randn_like(xy)*0.0001
        mask = self.point_is_inside(xy, crns)
        xy = xy[mask]
        return xy
    

    def point_is_inside(self, p, crns):
        angle_sum = 0
        L = len( crns)
        for i in range(L):
            a = crns[i]
            b = crns[(i + 1) % L]
            # cross = np.cross(a - p, b - p)
            ap = (a - p)
            bp = (b - p)
            cross = (ap[:,0]*bp[:,1] - ap[:,1]*bp[:,0])
            inner = torch.sum((a-p)*(b-p), dim=-1)
            angle_sum += torch.atan2(cross, inner)
        mask = torch.abs(angle_sum) > 1
        return mask


    def query_uv(self, uv, cid=0, masking=False, store_grad=False):
        crns = self.cell_corners[cid]
        nids = self.cells[cid]
        mask = None
        if masking:
            mask = self.point_is_inside(uv, crns) ## valid coordinates
            # uv = uv[mask]
        
        uv_split = torch.split(uv, 5000, dim=0)
        mv_coords = []
        gradients = []
        for uv_ in uv_split:
            mvc_, grad_ = self.query_robust(uv_, crns, nids, store_grad=store_grad)
            mv_coords.append(mvc_)
            if store_grad:
                gradients.append(grad_)
        if store_grad:
            gradients = torch.cat(gradients, dim=0)
            return mv_coords, gradients, mask
        mv_coords = torch.cat(mv_coords, dim=0)
        return mv_coords, mask

class QuadComplex(Complex):
    def __init__(self, num_nodes, cells, uvs, uv_to_cid, device, require_grad=False, dtype=torch.float32):
        self.device = device
        self.eps = 1e-6
        self.require_grad=require_grad
        self.boundary_sample_rate = 50

        self.dtype = dtype

        ## corner indices
        self.num_nodes = num_nodes
        self.cells = cells

        print("Number of nodes: ", self.num_nodes)
        print("Number of cells: ", len(self.cells))

        cell_corners = torch.Tensor([[1.0, 1.0],
                             [0.0, 1.0],
                             [0.0, 0.0],
                             [1.0, 0.0]]).to(self.device).to(self.dtype)
        ## make cell_corners list
        self.cell_corners = [cell_corners for _ in range(len(self.cells))]

        ## uv
        self.uvs = torch.tensor(uvs, device=self.device, dtype=self.dtype)
        self.uv_to_cid = torch.LongTensor(uv_to_cid).to(self.device).squeeze()

        ###############################################
        ## mv_coords and g of mesh uv
        self.all_mv_coords = []
        self.all_g = []
        ## eval
        self.eval_mv_coords = []
        self.eval_uv = []
        self.eval_uv_to_cid = []
        ## boundary
        self.boundary_mv_coords = []
        self.boundary_uv = []
        self.boundary_uvpe = []
        self.boundary_uv_to_cid = []
        self.boundary_g = []
        self.patch_random_sample_size = 300
        
        self.rand_mv_coords = []
        self.rand_uv = []
        self.rand_g = []
        self.rand_uv_to_cid = []
        self.rand_all_uvpe = []
        
        for cell_idx, crns in enumerate(self.cell_corners):
            nids = self.cells[cell_idx] ## node ids
            uvmask = self.uv_to_cid == cell_idx
            uv = self.uvs[uvmask]
            uv_split = torch.split(uv, 20000, dim=0)
            mv_coords = []
            g = []
            for uv_ in uv_split:
                mvc_, grad_ = self.query_robust(uv_, crns, nids, store_grad=require_grad)
                mv_coords.append(mvc_)
                if require_grad:
                    g.append(grad_)
            mv_coords = torch.cat(mv_coords, dim=0)
            self.all_mv_coords.append(mv_coords.detach())
            if g is not None and len(g) > 0:
                g = torch.cat(g, dim=0)
                self.all_g.append(g.detach())
            
            # ## boundary
            b_mvc, b_uv = self.query_boundary(cell_idx)
            b_g = None
            if require_grad:
                b_g = self.query_boundary_normals(b_uv, crns, nids)
            self.boundary_mv_coords.append(b_mvc.detach())
            self.boundary_uv.append(b_uv.detach())
            self.boundary_uv_to_cid.append(torch.ones(b_uv.shape[0], dtype=torch.long, device=self.device)*cell_idx)
            if b_g is not None:
                self.boundary_g.append(b_g.detach())

    
        self.all_mv_coords = torch.cat(self.all_mv_coords, dim=0)
        if len(self.all_g) > 0:
            self.all_g = torch.cat(self.all_g, dim=0)
        if isinstance(self.uv_to_cid, list):
            self.uv_to_cid = torch.cat(self.uv_to_cid, dim=0)

        self.boundary_mv_coords = torch.cat(self.boundary_mv_coords, dim=0).to(device='cpu')
        self.boundary_uv = torch.cat(self.boundary_uv, dim=0).to(device='cpu')
        self.boundary_uv_to_cid = torch.cat(self.boundary_uv_to_cid, dim=0).to(device='cpu')
        if len(self.boundary_g) > 0:
            self.boundary_g = torch.cat(self.boundary_g, dim=0).to(device='cpu')



class feedForwardNet(nn.Module):
    def __init__(self, dim_in, dim_out, num_neurons=256, num_layers=5, layer_norm=False, bias=True, dtype=torch.float32):
        super().__init__()
        
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.bias = bias
        # self.bias = False

        actv = torch.nn.Softplus(beta=100)
        dec_layers = []
        for i in range(num_layers-1):
            ## linear
            if i == 0:
                dec_layers.append(torch.nn.Linear(dim_in, num_neurons, bias=self.bias, dtype=dtype))
            else:
                dec_layers.append(torch.nn.Linear(num_neurons, num_neurons, bias=self.bias, dtype=dtype))

            ## layer norm
            if layer_norm:
                dec_layers.append(nn.GroupNorm(4, num_neurons, dtype=dtype))

            dec_layers.append(actv)

        self.ff = nn.Sequential(*dec_layers)
        self.physc_head = torch.nn.Linear(num_neurons, self.dim_out, bias=self.bias, dtype=dtype)

        ## init
        """
        turn on this to model details
        """
        for l in self.ff:
            if isinstance(l, torch.nn.Linear):
                torch.nn.init.xavier_normal_(l.weight)

    def forward(self, x):
        x = self.ff(x)
        return self.physc_head(x)
    

class MultiNetGraph(nn.Module):

    debug = False

    def __init__(
        self, 
        complex:Complex, 
        device, 
        dim_in, 
        num_de_layers=5,
        num_neurons=256, 
        emb_deg=128, 
        pe_dim=0, 
        pe_feature_dim_in=0,
        layer_norm=False,
        sn_init = False,
        dtype=torch.float32,
        **kwargs
        ):
        super().__init__()

        self.device = device
        dim0 = emb_deg
        self.curriculum_weight = 0.0
        self.sample_size = 10000
        self.physc_dim = dim_in
        self.pe_dim = pe_dim
        self.sn_init = sn_init

        self.complex = complex
        self.dtype = dtype

        self.emb_deg = emb_deg
        self.nodal_embeddings = nn.Embedding(self.complex.num_nodes, emb_deg, dtype=self.dtype) ## primal
        if not self.sn_init:
            self.__init_embedding_weights()

        """
        we do not use positional encoding at the end
        """
        self.pe_enc = PositionalEncoding(dim=self.pe_dim, feat_dim_in=pe_feature_dim_in)
        if self.pe_dim > 0:
            print("use pe", self.pe_dim)
            dim0 = self.pe_enc.feat_dim_out + (emb_deg - self.pe_enc.feat_dim_in)
            self.pe_on = True
        else:
            self.pe_on = False

        self.decoding = feedForwardNet(dim0, dim_in, num_neurons=num_neurons, num_layers=num_de_layers, layer_norm=layer_norm, dtype=self.dtype)        
        self.encoding = feedForwardNet(dim_in, dim0, num_neurons=num_neurons, num_layers=num_de_layers, layer_norm=layer_norm, dtype=self.dtype)

        self.register_buffer('eps',  torch.tensor(0.01))
        self.register_buffer('eye', torch.eye(2))

        print(self)
    
    def set_pe_on(self, pe_on:bool):
        self.pe_on = pe_on

    def set_pe_scale(self, scale:float = 0.1):
        self.pe_enc.scale = scale

    def get_data(self):
        return self.data

    def __init_embedding_weights(self):
        torch.nn.init.xavier_uniform_(self.nodal_embeddings.weight)
    
    def init_embedding_weights_with_data(self, data):
        with torch.no_grad():
            self.nodal_embeddings.weight.data[:,-3:] = deepcopy(data)

    def set_debug(self, debug:bool=True):
        self.debug = debug

    """
    sample the cell for reconstruction
    """
    def reconstruct(self, mv_latents):
        out = self.decoding(mv_latents)
        return out
    
    def cycle(self, x):
        out = self.encoding(x)
        return out
    
    def interpolate_latents(self, mv_coords:torch.Tensor, nodal_embeddings:torch.Tensor):
        return torch.matmul(mv_coords, nodal_embeddings)

    def forward_nograd(self, pts_idx=None):

        ## compute the mv_coords for each cell
        mv_coords = self.complex.all_mv_coords
        uv_to_cid = self.complex.uv_to_cid

        if pts_idx is not None:
            mv_coords = mv_coords[pts_idx]  # (N, 2)
            uv_to_cid = uv_to_cid[pts_idx]  # (N,)
        
        mv_latents = self.interpolate_latents(mv_coords, self.nodal_embeddings.weight)  # weight: 32x16
        mv_latents = self.pe_enc(mv_latents, on=self.pe_on)  # (N,)
        x = self.reconstruct(mv_latents)
        
        return x, uv_to_cid
    

    def forward(self, pts_idx=None):        
        ## compute the mv_coords for each cell
        mv_coords = self.complex.all_mv_coords
        dc_du = self.complex.all_g
        uv_to_cid = self.complex.uv_to_cid

        if pts_idx is not None:
            mv_coords = mv_coords[pts_idx]  # (N, 2)
            dc_du = dc_du[pts_idx]  # (N, 32, 2)
            uv_to_cid = uv_to_cid[pts_idx]  # (N,)

        ## compute the latent points from mv_coords
        mv_coords = mv_coords.requires_grad_()
        mv_latents = self.interpolate_latents(mv_coords, self.nodal_embeddings.weight)  # weight: 32x16
        mv_latents = self.pe_enc(mv_latents, on=self.pe_on)  # (N,)

        ## compute the physical coordinates from latent points
        x = self.reconstruct(mv_latents)
        normals = torch.zeros_like(x)
     
        ## jacobian with respect to uv
        dx_dc = [grad_compute(mv_coords, x[:,j], dtype=self.dtype) for j in range(self.physc_dim)]
        dx_dc = torch.stack(dx_dc, dim=1)
        dx_du = torch.matmul(dx_dc, dc_du)

        ## normalize
        denominator = torch.linalg.norm(dx_du, dim=1, keepdim=True)
        denominator = torch.nan_to_num(denominator, nan=1.0, posinf=1.0, neginf=1.0)
        denominator = torch.clamp(denominator, min=1e-6)
        self.data = dx_du/denominator ## normalize

        ## normal
        normals = torch.cross(dx_du[:,:,0], dx_du[:,:,1], dim=-1)
        denominator = torch.linalg.norm(normals, dim=-1, keepdim=True)
        denominator = torch.nan_to_num(denominator, nan=1.0, posinf=1.0, neginf=1.0)
        denominator = torch.clamp(denominator, min=1e-6)
        normals = normals/denominator ## normalize

        ## regularization on the codes
        reg_loss = torch.mean(self.nodal_embeddings.weight**2)

        return x, normals, uv_to_cid, reg_loss


    def forward_dgp(self, pts_idx=None):
        ## compute the mv_coords for each cell
        mv_coords = self.complex.all_mv_coords
        dc_du = self.complex.all_g
        uv_to_cid = self.complex.uv_to_cid
        # print("forward: access data: ", t01-t0)

        if pts_idx is not None:
            mv_coords = mv_coords[pts_idx]  # (N, 2)
            dc_du = dc_du[pts_idx]  # (N, 32, 2)
            uv_to_cid = uv_to_cid[pts_idx]  # (N,)

        ## compute the latent points from mv_coords
        mv_coords = mv_coords.requires_grad_()
        mv_latents = self.interpolate_latents(mv_coords, self.nodal_embeddings.weight)  # weight: 32x16
        mv_latents = self.pe_enc(mv_latents, on=self.pe_on)  # (N,)

        ## compute the physical coordinates from latent points
        x = self.reconstruct(mv_latents)
        mv_latents_cycle = self.cycle(x)
        cycle_loss = torch.mean((mv_latents - mv_latents_cycle)**2)

        ## jacobian with respect to uv
        dx_dc = [grad_compute(mv_coords, x[:,j], dtype=self.dtype) for j in range(self.physc_dim)]
        dx_dc = torch.stack(dx_dc, dim=1)
        dx_du = torch.matmul(dx_dc, dc_du)


        ## normalize normal
        normals = torch.cross(dx_du[:,:,0], dx_du[:,:,1], dim=-1)
        area = torch.linalg.norm(normals, dim=-1, keepdim=True)
        area = torch.nan_to_num(area, nan=1.0, posinf=1.0, neginf=1.0)
        area = torch.clamp(area, min=1e-6)

        normals = normals/area ## normalize
        # mean_area = torch.mean(area).detach()
        # area_loss = torch.mean((area - mean_area)**2)

        ## regularization on the codes
        reg_loss = torch.mean(self.nodal_embeddings.weight**2)

        ## first fundamental form (N, 2, 2)
        ## E, G, F = FFF[:,0,0], FFF[:,1,1], FFF[:,0,1]
        FFF = torch.matmul(dx_du.transpose(1,2), dx_du)

        """
        you may use ARAP or Dirichlet energy here
        """

        # U, S, V = torch.svd(dx_du)
        # ones = torch.ones_like(S)
        # e_arap = torch.mean(torch.sum((S - ones)**2, dim=-1))

        ## dirichlet
        ### \int_S(||df/du||^2 + ||df/dv||^2)
        e_dirichlet = 1.0*torch.mean(0.5*torch.sum(dx_du*dx_du, dim=1))

        conformal_loss = torch.mean(FFF[:,1,0]**2)
        return x, normals, uv_to_cid, e_dirichlet, conformal_loss, cycle_loss, reg_loss
    
    
    def forward_corners(self):
        mv_corners = torch.eye(self.complex.num_nodes, dtype=self.dtype, device=self.device)
        mv_latents = self.interpolate_latents(mv_corners, self.nodal_embeddings.weight)
        mv_latents = self.pe_enc(mv_latents, on=self.pe_on)
        x = self.reconstruct(mv_latents)
        return x

    def forward_boundary(self, return_normals=False):
        mv_coords = self.complex.boundary_mv_coords.to(self.device)
        uv = self.complex.boundary_uv.to(self.device)
        mv_coords = mv_coords.requires_grad_()
        uv_to_cid = self.complex.boundary_uv_to_cid.to(self.device)
        b_mv_latents = self.interpolate_latents(mv_coords, self.nodal_embeddings.weight)
        b_mv_latents = self.pe_enc(b_mv_latents, on=self.pe_on)  # (N,)
        boundary_x = self.reconstruct(b_mv_latents)
        if not return_normals:
            return boundary_x, None, uv

        # ## get dc_du for boundary
        dc_du = self.complex.boundary_g.to(self.device)
        ## jacobian with respect to uv
        dx_dc = [grad_compute(mv_coords, boundary_x[:,j], dtype=self.dtype) for j in range(self.physc_dim)]
        dx_dc = torch.stack(dx_dc, dim=1)
        dx_du = torch.matmul(dx_dc, dc_du)
        t4 = time.time()

        ## normalize normal
        normals = torch.cross(dx_du[:,:,0], dx_du[:,:,1], dim=-1)
        area = torch.linalg.norm(normals, dim=-1, keepdim=True)
        area = torch.nan_to_num(area, nan=1.0, posinf=1.0, neginf=1.0)
        area = torch.clamp(area, min=1e-6)
        normals = normals/area ## normalize

        return boundary_x, normals, uv

    def forward_uv(self, uv:torch.Tensor, cid=0):
        mv_coords, mask = self.complex.query_uv(uv, cid=cid, masking=False)
        interp_latents = self.interpolate_latents(mv_coords, self.nodal_embeddings.weight)
        interp_latents = self.pe_enc(interp_latents, on=self.pe_on)
        x = self.reconstruct(interp_latents)
        return x, mask
    
    def forward_uv_w_grad(self, uv, cid=0):
        mask = None
        mv_coords, dc_du, _ = self.complex.query_uv(uv, cid=cid, masking=False, store_grad=True)
        interp_latents = self.interpolate_latents(mv_coords, self.nodal_embeddings.weight)
        interp_latents = self.pe_enc(interp_latents, on=self.pe_on)
        x = self.reconstruct(interp_latents)

        ## jacobian with respect to uv
        dx_dc = [grad_compute(mv_coords, x[:,j], dtype=self.dtype) for j in range(self.physc_dim)]
        dx_dc = torch.stack(dx_dc, dim=1)
        dx_du = torch.matmul(dx_dc, dc_du)

        return x, dx_du, mask
    
    def forward_boundary2(self, cid):
        mv_coords, buv = self.complex.query_boundary(cid)
        interp_latents = self.interpolate_latents(mv_coords, self.nodal_embeddings.weight)
        interp_latents = self.pe_enc(interp_latents, on=self.pe_on)
        x = self.reconstruct(interp_latents)
        return x, buv

    def evaluate_uv(self, uv, cid=0):
        with torch.no_grad():
            mv_coords, _ = self.complex.query_uv(uv, cid=cid)
            split_mv_coords = torch.tensor_split(mv_coords, 100, dim=0) 
            split_x = []
            split_latents = []
            split_latents_enc = []
            for i, mv_coords in enumerate(split_mv_coords):
                interp_latents = self.interpolate_latents(mv_coords, self.nodal_embeddings.weight)
                split_latents.append(interp_latents)
                interp_latents = self.pe_enc(interp_latents, on=self.pe_on)
                split_latents_enc.append(interp_latents)
                x = self.reconstruct(interp_latents)
                split_x.append(x)
            x = torch.cat(split_x, dim=0)
            latents = torch.cat(split_latents, dim=0)
            latents_enc = torch.cat(split_latents_enc, dim=0)
        return x, latents, latents_enc

    def evaluate_patch(self):
        with torch.no_grad():
            mv_coords = self.complex.all_mv_coords.to(self.device)
            uv = self.complex.uvs.to(self.device)
            uv_to_cid = self.complex.uv_to_cid.to(self.device)
            
            ## chunk mv_coords
            split_mv_coords = torch.tensor_split(mv_coords, 100, dim=0) 
            split_x = []
            for i, mv_coords in enumerate(split_mv_coords):
                interp_latents = self.interpolate_latents(mv_coords, self.nodal_embeddings.weight)
                interp_latents = self.pe_enc(interp_latents, on=self.pe_on)
                x = self.reconstruct(interp_latents)
                split_x.append(x)
            x = torch.cat(split_x, dim=0)
        return x, uv, uv_to_cid
    

    def evaluate_boundary(self):
        with torch.no_grad():
            mv_coords = self.complex.boundary_mv_coords.to(self.device)
            print(len(self.complex.boundary_uv), self.complex.boundary_uv[0].shape)
            uv = self.complex.boundary_uv.to(self.device)
            uv_to_cid = self.complex.boundary_uv_to_cid.to(self.device)
            interp_latents = self.interpolate_latents(mv_coords, self.nodal_embeddings.weight)
            interp_latents = self.pe_enc(interp_latents, on=self.pe_on)
            boundary_x = self.reconstruct(interp_latents)
        return boundary_x, uv, uv_to_cid
    

    def evaluate_inv_uv(self, x):
        with torch.no_grad():
            h = self.encoding(x)
            logits = self.inverse_cls(h)
            pred_uv = self.inverse_uv(h)
        return logits, pred_uv