# NeuPPS: Neural Piecewise Parametric Surfaces

**This is the readme file for the following paper:**
Lei Yang*, Yongqing Liang*, Xin Li, Congyi Zhang, Guying Lin, Cheng Lin, Alla Sheffer, Scott Schaefer, John Keyser, and Wenping Wang. 2025. **NeuPPS: Neural Piecewise Parametric Surfaces** . ACM Trans. Graph. 45, 2, Article 13, 18 pages. https://doi.org/10.1145/3771546

An earlier version of this paper (2023) was published in Arxiv: 
https://arxiv.org/abs/2309.09911


## A. Environment setup
**Training NeuPPS**
This repository mainly requires
`trimesh` for processing mesh data
`pytorch` for neural network implementation
`triangle` for planar triangulation in the parametric domain
See `environment.yml` for the detailed environment setup.

**For data processing and visualization**
For parameterization, here we provide the SLIM parameterization offered in `libigl` [[Jacobson et al. 2018](https://libigl.github.io)]
For computing geodesics, we use `potpourri3d` [[Sharp 2021](https://github.com/nmwsharp/potpourri3d)]
For visualization, we use `vedo` [[Musy et al. 2021](https://github.com/marcomusy/vedo)].

## B. Training code
```
python main.py --config Path/to/config --model_name shapename --duv
```
For example, you may specify a config file (e.g., `./configs/config.json`) and a model name, e.g., bimba (assuming the folder `./data/bimba` exists and contain the same data structure as described above).

**1) Example for a genus-1 shape with polygonal segmentation, Bimba**
```
python main.py --config configs/config.json --model_name bimba
```

**2) Example for point cloud data, pointcloud_skirt_529** 
```
python main_pc.py --config configs/config_noparam.json --model_name pointcloud_skirt_529 --duv
```

**Data structure**
Testing data: [OneDrive Link](
https://1drv.ms/f/c/072d8a240b13b9af/IgC-xE36InipQbco5HHG4vYYAUkjpdzb3nDMuJvq6FJH3o8?e=Nfygp3)


```
Folder
|-- data
    |-- single
        |-- mesh.obj (the surface mesh of the target shape)
    |-- cell_arc_lengths.json (arc length of each edge in each cell)
    |-- mask.json (??)
    |-- topology_graph.json
|-- flat_parameterization (planar parameterized meshes for checking)
|-- parameterization (segmented surface patches with parameterization)

```
`topology_graph.json` is the polygonal complex containing the cell topology and the node coordinates/ids referring back to the `./single/mesh.obj`

**How to prepare this data folder for training?** Please refer to [Data Preparation](data_preparation/DATA_PREPARATION.md) for detail.

## C. Evaluation code
```
python main.py --config Path/to/config --resume Path/to/resume/folder --eval
```

## D. Code structure

```
NeuralSurfaceRelease/
|-- main.py                  # Entry point for mesh-based training and evaluation
|-- main_pc.py               # Entry point for point cloud input
|-- run_all_exp.sh           # Script to run all experiments in batch
|-- environment.yml          # Conda environment specification
|
|-- configs/                 # JSON configuration files for experiments
|   |-- config.json          # Default config (mesh with parameterization)
|   |-- config_noparam.json  # Config for point cloud input (no parameterization)
|
|-- model/                   # Core model and training code
|   |-- model.py             # Neural network definitions
|   |   # - PositionalEncoding: Fourier positional encoding for UV inputs
|   |   # - Complex: polygonal complex structure storing cell topology and UV coords
|   |   # - QuadComplex: extension of Complex for quadrilateral patches
|   |   # - MultiNetGraph: main neural network; node embeddings + per-patch MLPs
|   |-- dataset.py           # Dataset classes for mesh-based and point cloud inputs
|   |-- para_trainer.py      # Trainer for parameterized mesh inputs (SingleTrainerParam)
|   |-- pc_trainer.py        # Trainer for point cloud inputs
|
|-- common_tools/            # Shared utility modules
|   |-- io_tools.py          # File I/O (JSON, OBJ, XYZ, etc.)
|   |-- build_complex.py     # Constructs the polygonal complex from topology data
|   |-- eig_decomposition.py # Eigenvalue / FFT utilities for boundary curves
|   |-- halfedge_mesh.py     # Half-edge mesh data structure
|   |-- get_mesh_boundary.py # Extracts mesh boundary loops
|   |-- get_border_edges.py  # Identifies border edges of patches
|   |-- normalize_data.py    # Data normalization helpers
|   |-- folder_tools.py      # Directory management utilities
|   |-- logger.py            # Logging helpers
|
|-- data/                    # Input data (one sub-folder per shape)
|   |-- <shape_name>/
|       |-- data/
|       |   |-- single/mesh.obj          # Input surface mesh
|       |   |-- topology_graph.json      # Polygonal complex topology
|       |   |-- cell_arc_lengths.json    # Arc lengths along patch boundaries
|       |   |-- mask.json                # Face mask for each patch
|       |-- parameterization/            # Segmented surface patches with UV maps
|       |-- flat_parameterization/       # Flattened (planar) parameterization for checking
|
|-- data_preparation/        # Scripts and instructions for preparing new shapes
|   |-- DATA_PREPARATION.md
|
|-- release_test/            # Example experiment output folders
    |-- <exp_name>/
        |-- config.json      # Config snapshot for this run
        |-- ckpt/            # Saved model checkpoints (every 1000 iters + latest)
        |-- model/           # Code snapshot saved alongside the checkpoint
        |-- res/             # Output meshes / results
```

**Key design concepts:**
- Each shape is decomposed into a **polygonal complex** of surface patches.  
- Each patch is assigned a set of **nodal embeddings** (learnable per-vertex latent codes) interpolated via mean-value coordinates to produce per-point features.  
- A shared **MLP decoder** (`MultiNetGraph`) maps the interpolated features + positional encoding to 3D surface positions.  
- Boundary consistency between adjacent patches is enforced during training via arc-length–parameterized boundary losses.

## Citation of Our Paper
If you find this paper and our codebase useful, please cite
```
@article{yang2025neupss,
author = {Yang, Lei and Liang, Yongqing and Li, Xin and Zhang, Congyi and Lin, Guying and Lin, Cheng and Sheffer, Alla and Schaefer, Scott and Keyser, John and Wang, Wenping},
title = {NeuPPS: Neural Piecewise Parametric Surfaces},
year = {2025},
issue_date = {April 2026},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
volume = {45},
number = {2},
issn = {0730-0301},
url = {https://doi.org/10.1145/3771546},
doi = {10.1145/3771546},
journal = {ACM Trans. Graph.},
month = dec,
articleno = {13},
numpages = {18},
keywords = {Deep learning, Surface reconstruction}
}
```

## Used open-sourced codes
We thank the authors of the following open-sourced codes that make this project possible
- [Libigl](https://libigl.github.io) — Alec Jacobson, Daniele Panozzo, et al. *libigl: A simple C++ geometry processing library.* 2018.
- [Vedo](https://github.com/marcomusy/vedo) — Marco Musy et al. *vedo, a python module for scientific analysis and visualization of 3D objects and point clouds.* 2021.
- [Triangle](https://www.cs.cmu.edu/~quake/triangle.html) — Jonathan R. Shewchuk. *Triangle: Engineering a 2D Quality Mesh Generator and Delaunay Triangulator.* 1996.
- [Trimesh](https://github.com/mikedh/trimesh) — Michael Dawson-Haggerty et al. *trimesh.* 2019.
- [PyTorch](https://pytorch.org) — Adam Paszke et al. *PyTorch: An Imperative Style, High-Performance Deep Learning Library.* NeurIPS 2019.
- [potpourri3d](https://github.com/nmwsharp/potpourri3d) — Nicholas Sharp. *potpourri3d.* 2021.
- [torch_batch_svd](https://github.com/KinglittleQ/torch-batch-svd) — for batched singular value decomposition.