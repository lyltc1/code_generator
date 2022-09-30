import argparse
import pymeshlab as ml
import trimesh
from tqdm import tqdm

from utils.config import config, root

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='tless')
parser.add_argument('--face-quality-threshold', type=float, default=1e-3)
parser.add_argument('--remesh-percentage', type=float, default=1)
args = parser.parse_args()

mesh_folder = root / 'data/bop' / args.dataset / config[args.dataset].model_folder
remesh_folder = root / 'data/remesh_visible' / args.dataset
remesh_folder.mkdir(exist_ok=True, parents=True)

for mesh_fp in tqdm(list(mesh_folder.glob('*.ply'))):
    remesh_fp = remesh_folder / mesh_fp.name

    print()
    print(mesh_fp)
    print()

    ms = ml.MeshSet()
    ms.load_new_mesh(str(mesh_fp.absolute()))

    ms.meshing_repair_non_manifold_edges()
    ms.meshing_surface_subdivision_midpoint(iterations=10, threshold=ml.Percentage(args.remesh_percentage))
    ms.compute_scalar_ambient_occlusion(occmode='per-Face (deprecated)', reqviews=256)
    face_quality_array = ms.current_mesh().face_scalar_array()
    minq = face_quality_array.min()
    if minq < args.face_quality_threshold:
        assert face_quality_array.max() > args.face_quality_threshold
        ms.compute_selection_by_scalar_per_face(minq=minq, maxq=args.face_quality_threshold)
        ms.meshing_remove_selected_faces()
        ms.meshing_remove_unreferenced_vertices()
    ms.save_current_mesh(str(remesh_fp.absolute()), save_textures=False, binary=False)

    area_reduction = trimesh.load_mesh(remesh_fp).area / trimesh.load_mesh(mesh_fp).area
    print()
    print(mesh_fp)
    print(f'area reduction {area_reduction}')
    print()
