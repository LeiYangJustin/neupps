import sys
import os
sys.path.append(os.getcwd())

import numpy as np
from common_tools.io_tools import *
from tqdm import tqdm
import argparse
import shutil


import torch
from torch.utils.tensorboard import SummaryWriter
from model.para_trainer import SingleTrainerParam as Trainer
# from planar import SingleTrainerParam as Trainer

import trimesh

def parse_args():
    parser = argparse.ArgumentParser(description="Modeling 3D shapes with neural patches")
    parser.add_argument("--config", "-c",
                        required=True,
                        type=str,
                        help="path to config"
                        )
    parser.add_argument("--seed", "-s",
                        default=None,
                        type=int,
                        help="random seed"
                        )
    parser.add_argument("--model_name", 
                        required=True,
                        type=str,
                        help="which shape to train on"
                        )
    parser.add_argument("--eval", 
                        action='store_true',
                        help="evaluation mode"
                        )
    parser.add_argument("--expname", "-n",
                        default="default",
                        type=str,
                        help="experiment name"
                        )
    parser.add_argument("--checkpoint",
                        default='latest', 
                        type=str,
                        help="which checkpoint to load"
                        )
    parser.add_argument("--resume",
                        default=None, 
                        type=str,
                        help="resume from the checkpoint"
                        )
    parser.add_argument("--cuda_id", 
                        default=None, 
                        type=int,
                        help="which cuda device to use"
                        )
    parser.add_argument("--screen_log", '-sl',
                action='store_true')
    
    ## hyper-parameters for the network
    parser.add_argument("--pe", 
                        default=None,
                        type=int,
                        help="positional encoding dimension"
                        )
    parser.add_argument("--num_layers",
                        default=None,
                        type=int,
                        help="number of layers in the network"
                        )
    parser.add_argument("--num_neurons",
                        default=None,
                        type=int,
                        help="number of neurons in each layer"
                        )
    parser.add_argument("--latent_dim",
                        default=None,
                        type=int,
                        help="latent dimension for the feature complex"
                        )
    ## use extra samples for training
    parser.add_argument("--extra_samples",
                        action='store_true',
                        help="use extra samples for training"
                        )
    
    ## hyper-parameters for the loss
    parser.add_argument("--p2p",
                        action='store_true',
                        help="use point-to-point loss"
                        )
    parser.add_argument("--duv",
                        action='store_true',
                        help="use uv regularization"
                        )
    parser.add_argument("--uniform", 
                        default=None, 
                        type=float,
                        help="weight for uniform loss"
                        )
    parser.add_argument("--conformal", 
                        default=None, 
                        type=float,
                        help="weight for conformal loss"
                        )

    args = parser.parse_args()
    return args



if __name__ == "__main__":


    args = parse_args()
    cfg = read_json(args.config)

    if args.seed is not None:
        cfg['seed'] = args.seed

    seed = cfg['seed']
    np.random.seed(seed)
    torch.manual_seed(seed)

    cfg['expname'] = args.expname + '_' + str(seed)
    cfg['resume'] = args.resume
    cfg['checkpoint'] = args.checkpoint

    if args.cuda_id is not None:
        cfg['cuda_id'] = args.cuda_id
    if args.model_name is not None:
        cfg['model_name'] = args.model_name
        splits = cfg['train_data'].split('/')
        splits[-1] = args.model_name
        cfg['train_data'] = '/'.join(splits)
        print(cfg['train_data'])

        ## expname
        cfg['expname'] = args.model_name + '_' + cfg['expname']
    else:
        cfg['expname'] = args.expname

    if args.num_layers is not None:
        cfg['net_params']['num_de_layers'] = args.num_layers
    if args.num_neurons is not None:
        cfg['net_params']['num_neurons'] = args.num_neurons
    if args.latent_dim is not None:
        cfg['net_params']['emb_deg'] = args.latent_dim
    if args.pe is not None:
        cfg['net_params']['pe_dim'] = args.pe
    if args.duv:
        cfg['net_params']['use_duv'] = True
        cfg['expname'] += '_duv'
    else:
        cfg['net_params']['use_duv'] = False

    if args.conformal is not None:
        cfg['loss_params']['conformal'] = args.conformal
        cfg['expname'] += '_c' + str(args.conformal)    
    if args.uniform is not None:
        cfg['loss_params']['uniform'] = args.uniform
        cfg['expname'] += '_u' + str(args.uniform)

    """
    Evaluation mode
    1) Tesselate the surface
    2) Compute metrics
    """
    if args.eval:
        if not os.path.exists("eval"):
            os.makedirs("eval")
        config_path = os.path.join(args.resume, 'config.json')
        cfg = read_json(config_path)
        cfg['expname'] = args.expname
        cfg['resume'] = args.resume
        cfg['checkpoint'] = args.checkpoint
        cfg['mode'] = 'eval'
        cfg['save_dir'] = cfg['resume'].replace('res_param', 'eval')
        
        ## compute metrics
        try:
            gt_mesh = trimesh.load(
                os.path.join(cfg["train_data"], "data/single/mesh.obj"),
                process=False, maintain_order=True)
        except ValueError:
            gt_mesh = trimesh.load(
                os.path.join(cfg["train_data"], "data/single/mesh.ply"),
                process=False, maintain_order=True)
        gt_pq = trimesh.proximity.ProximityQuery(gt_mesh)
        
        ## tesselate the neural surface
        if not os.path.exists(cfg['save_dir']):
            os.makedirs(cfg['save_dir'])
        cfg['add_samples'] = 0
        trainer = Trainer(cfg)
        patches = trainer.extract_mesh_with_uv(epoch=0)        
        exit()

    """
    setting up the save directory
    1) res/: save the training results
    2) ckpt/: save the checkpoints
    3) model/: save the code files
    4) config.json: save the config file
    5) tensorboard log files    
    """
    SAVE_DIR = os.path.join(cfg['save_dir'], cfg['expname']+'_'+str(get_timestamp()))
    cfg['save_dir'] = os.path.join(SAVE_DIR, 'res')
    cfg['ckpt_dir'] = os.path.join(SAVE_DIR, 'ckpt')
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
        os.makedirs(cfg['save_dir'])
        os.makedirs(cfg['ckpt_dir'])
        os.makedirs(os.path.join(SAVE_DIR, 'model'))

    print(cfg)
    shutil.copyfile(args.config, os.path.join(SAVE_DIR, 'config.json'))
    shutil.copyfile("./model/para_trainer.py", os.path.join(SAVE_DIR, 'model/para_trainer.py'))
    shutil.copyfile("./model/model.py", os.path.join(SAVE_DIR, 'model/model.py'))
    shutil.copyfile("./model/dataset.py", os.path.join(SAVE_DIR, 'model/dataset.py'))
    write_json(cfg, os.path.join(SAVE_DIR, 'config.json'))

    ## tensorboard writer
    writer = SummaryWriter(log_dir=SAVE_DIR)


    """
    setting up trainer

    """
    trainer = Trainer(cfg)
    start_epoch = trainer.start_epoch
    max_epoch = trainer.max_epoch
    dataset_size = len(trainer.dataset.patch_xyz)

    ## split the dataset into batches
    if cfg['batchsize'] > 0:
        num_batches = int(np.ceil(dataset_size / cfg['batchsize']))
        print("dataset size", dataset_size)
        print("batchsize", cfg['batchsize'])
        print(f"Num batches: {num_batches}")
        all_sample_idx = np.arange(dataset_size)
        all_sample_idx = np.random.permutation(all_sample_idx)
        batch_of_sample_idx = np.array_split(all_sample_idx, num_batches)

    ## training loop
    for epoch in tqdm(range(start_epoch, max_epoch), desc="training...", total=max_epoch, ncols=80):
        
        """ checkpointing and mesh extraction """
        if epoch % cfg['eval_step'] == 0:
            all_sample_idx = np.arange(dataset_size)
            all_sample_idx = np.random.permutation(all_sample_idx)
            batch_of_sample_idx = np.array_split(all_sample_idx, num_batches)
            trainer.save_checkpoint(epoch)            
            if trainer.train_method == 'no_param':
                print("Extracting mesh")
                trainer.extract_mesh(epoch=epoch)
            else:
                trainer.extract_mesh_with_uv(epoch=epoch)
            
            trainer.net.set_debug()
        else:
            trainer.net.set_debug(False)

        sample_idx = batch_of_sample_idx[epoch % num_batches]
        # all_sample_idx = np.arange(dataset_size)
        # sample_idx = all_sample_idx

        """ evaluation """
        eval_dict = trainer.evaluate(epoch, sample_idx = sample_idx)
        for k, v in eval_dict.items():
            writer.add_scalar(f"eval/{k}", v, epoch)

        """ training iteration """
        loss_dict = trainer.train_iteration(epoch, sample_idx)
        if epoch % 10 == 0:
            for k, v in loss_dict.items():                    
                if k == 'sample_idx':
                    continue
                try:
                    writer.add_scalar(f"loss/{k}", v, epoch)
                except NotImplementedError:
                    print(f"loss/{k}", v, epoch)
                    raise NotImplementedError        
        writer.flush()

    trainer.save_checkpoint(epoch)
    print("Done")