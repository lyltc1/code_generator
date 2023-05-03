""" This script generate labels for BOP datasets

    both models_GT_color directory and labels directory defined in cfg
"""

import os
import argparse
import numpy as np
from tqdm import tqdm
import cv2
import open3d as o3d
from utils import bop_io
from utils.config import config, root
from utils.obj import load_objs
from utils.renderer import ObjCoordRenderer

def generate_GT_images(bop_path, data_folder, cfg, obj_ids=None):
    print(f"start generate labels for {cfg.dataset}/{data_folder}")
    dataset_dir, _, _, _, model_ids, rgb_files, _, _, _, gts, _, cam_param_global, scene_cam = bop_io.get_dataset(
        bop_path, cfg.dataset, incl_param=True, data_folder=data_folder)

    target_dir = str(cfg.binary_code_folder / cfg.dataset / (data_folder + '_GT'))
    im_width, im_height = cam_param_global['im_size']
    objs, obj_ids = load_objs(cfg, obj_ids=obj_ids)
    renderer = ObjCoordRenderer(objs, obj_ids, im_width, im_height)

    for render_id in obj_ids:
        assert render_id in model_ids
        ply_path = cfg.models_GT_color_folder / cfg.dataset / "obj_{:06d}.ply".format(render_id)
        pcd = o3d.io.read_point_cloud(str(ply_path))
        color = np.array(pcd.colors)
        pcd_tree = o3d.geometry.KDTreeFlann(pcd)

        print("rgb_files:", len(rgb_files))
        print("gts:", len(gts))
        print("scene_cam:", len(scene_cam))

        ## for each image render the labels
        for img_id in tqdm(range(len(rgb_files))):
            rgb_path = rgb_files[img_id]
            rgb_path = rgb_path.split("/")
            scene_id = rgb_path[-3]
            image_name = rgb_path[-1][:-4]

            GT_img_dir = os.path.join(target_dir, scene_id)

            if not (os.path.exists(GT_img_dir)):
                os.makedirs(GT_img_dir)

            cam_K_local = np.array(scene_cam[img_id]["cam_K"]).reshape(3, 3)


            camera_parameters_local = np.array(
                [im_width, im_height, cam_K_local[0, 0], cam_K_local[1, 1], cam_K_local[0, 2], cam_K_local[1, 2]])

            # visible side
            for count, gt in enumerate(gts[img_id]):
                if gt['obj_id'] != render_id:
                    continue
                GT_img_fn = os.path.join(GT_img_dir, "{}_{:06d}.png".format(image_name, count))
                if os.path.exists(GT_img_fn):
                    print(f"Warning: {GT_img_fn} exists!")

                tra_pose = np.array(gt['cam_t_m2c'])
                rot_pose = np.array(gt['cam_R_m2c'])

                obj_coord = renderer.render(render_id, cam_K_local, rot_pose, tra_pose).copy()
                UV = np.nonzero(obj_coord[..., 3])
                obj_coord = obj_coord[..., :3]
                obj_coord = renderer.denormalize(obj_coord, render_id)
                coordinates = obj_coord[UV]
                labels = np.zeros_like(obj_coord)
                for coordi, u, v in zip(coordinates, UV[0], UV[1]):
                    k, idx, _ = pcd_tree.search_knn_vector_3d(coordi, 1)
                    labels[u, v] = color[idx] * 255.

                cv2.imwrite(GT_img_fn, labels[..., ::-1])



if __name__ == "__main__":
    # ---- parse arg ----
    parser = argparse.ArgumentParser(description='generate image labels for dataset')
    parser.add_argument('--dataset', help='the folder name of the dataset in the bop folder', required=True)
    args = parser.parse_args()
    # ---- generate binary code to cfg.binary_code_folder ----
    bop_path = root / 'data' / 'bop'
    cfg = config[args.dataset]
    cfg.dataset = args.dataset
    obj_ids = [1, 6, 9]
    for data_folder in [cfg.test_folder, cfg.train_folder, 'train_pbr']:
        generate_GT_images(bop_path, data_folder, cfg, obj_ids=obj_ids)
