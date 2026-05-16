import trimesh
import numpy as np
import argparse
import os


parser = argparse.ArgumentParser('Normalize the mesh.')
parser.add_argument('--input', type=str, default='data/bimba_nf1M/data', help='Input mesh path.')
parser.add_argument('--mesh', type=str, default='single/bimba_nf1M.obj', help='Input mesh path.')
args = parser.parse_args()

folder = args.input

mesh = trimesh.load(os.path.join(args.input, args.mesh), process=False, maintain_order=True)

print(len(mesh.vertices), len(mesh.faces))


## normalize the mesh
mesh.vertices -= mesh.centroid
mesh.vertices /= np.max(np.abs(mesh.vertices))
mesh.vertices *= 0.5

## save the normalized mesh
mesh.export(f'{folder}/single/mesh_normalized.obj')
