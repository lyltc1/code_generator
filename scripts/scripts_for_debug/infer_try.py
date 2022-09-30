""" This script trys to recover pose based on ground truth code """
import argparse
from pathlib import Path
import numpy as np
import cv2

from utils.config import config, root
from utils.instance import BopInstanceDataset
from utils.std_auxs import GTLoader, RgbLoader, MaskLoader, RandomRotatedMaskCrop, ObjCoordAux
from utils.class_id_encoder_decoder import RGB_to_class_id, class_id_to_class_code_images, load_decoders
from utils.obj import load_objs
from utils.renderer import project
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='tless')
parser.add_argument('--pbr', default=False)
parser.add_argument('--test', default=False)
args = parser.parse_args()

res_crop = 224
dataset = args.dataset
dataset_path = root / 'data/bop' / dataset

cfg = config[dataset]
cfg.dataset = dataset
objs, obj_ids = load_objs(cfg)
decoders = load_decoders(cfg.models_GT_color_folder / dataset)

auxs = [RgbLoader(), MaskLoader(mask_type='mask'), GTLoader(),
        RandomRotatedMaskCrop(res_crop, mask_key='mask', crop_keys=('rgb', 'mask', 'code'), rgb_interpolation=cv2.INTER_LINEAR),
        ObjCoordAux(objs, res_crop)]

dataset_args = dict(dataset_root=dataset_path, auxs=auxs, cfg=cfg, obj_ids=obj_ids)
data = BopInstanceDataset(**dataset_args, pbr=args.pbr, test=args.test)

window_names = ['rgb_crop', 'mask_crop', 'code_crop', 'dist', 'xy', 'yz', 'zx']
for j, name in enumerate(window_names):
    cv2.namedWindow(name, cv2.WINDOW_KEEPRATIO)
    cv2.resizeWindow(name, res_crop, res_crop)
    cv2.moveWindow(name, 1 + 250 * j, 1 + 250 *0)
for j in range(16):
    row, col = j // 6, j % 6
    cv2.namedWindow(str(j), cv2.WINDOW_KEEPRATIO)
    cv2.resizeWindow(str(j), res_crop, res_crop)
    cv2.moveWindow(str(j), 1 + 250 * col, 300 + 250 * row)


print()
print('With an opencv window active:')
print("press 'a', 'd' and 'x'(random) to get a new input image,")
print("press 'q' to quit.")
data_i = 0
while True:
    print()
    print('------------ new input -------------')
    inst = data[data_i]
    obj_id = inst['obj_id']
    corresponding = decoders[inst['obj_id']]['corresponding']
    print(f'i: {data_i}, obj_id: {obj_id}')
    rgb_crop = inst['rgb_crop']
    mask_crop = inst['mask_crop']
    RGB_code_crop = inst['code_crop']
    if not RGB_code_crop.any():
        data_i = np.random.randint(len(data))
        continue
    class_code_img = RGB_to_class_id(RGB_code_crop)
    binary_code_img = class_id_to_class_code_images(class_code_img)  # [h, w, bit]
    cv2.imshow('rgb_crop', rgb_crop[..., ::-1])
    cv2.imshow('mask_crop', mask_crop)
    cv2.imshow('code_crop', RGB_code_crop[..., ::-1])
    cv2.imshow('dist', inst['obj_coord'][..., 2::-1])
    for j in range(16):
        cv2.imshow(str(j), binary_code_img[..., j])

    # corr vis
    uv_names = 'xy', 'xz', 'yz'
    uv_slices = slice(1, None, -1), slice(2, None, -2), slice(2, 0, -1)
    uv_uniques = []
    uv_all = ((objs[obj_id].mesh_norm.vertices + 1) * (res_crop / 2 - .5)).round().astype(int)
    for uv_name, uv_slice in zip(uv_names, uv_slices):
        view_uvs_unique, view_uvs_unique_inv = np.unique(uv_all[:, uv_slice], axis=0, return_inverse=True)
        uv_uniques.append((view_uvs_unique, view_uvs_unique_inv))

    last_mouse_pos = 0, 0
    def mouse_cb(event, x, y, flags=0, *param):
        global last_mouse_pos
        if flags & cv2.EVENT_FLAG_CTRLKEY:
            return
        bit = param[0][0]
        class_base = param[0][1]
        last_mouse_pos = x, y
        class_code = binary_code_img[y, x]
        if not np.any(class_code):
            return
        code_length = len(class_code)
        class_id = 0
        for i in range(bit + 1):
            class_id = class_id + class_code[i] * (class_base ** (code_length - 1 - i))
        class_id_max = class_id
        for i in range(bit + 1, code_length):
            class_id_max += 1 * (class_base ** (code_length - 1 - i))
        class_id, class_id_max = int(class_id), int(class_id_max)
        possible_points = []
        for i in range(class_id, class_id_max + 1):
            for point in corresponding[str(i)]:
                possible_points.append(point)
        possible_points = np.array(possible_points)
        xy = project(possible_points, inst['cam_R_obj'], inst['cam_t_obj'], inst['K_crop'])
        xy = xy.round().astype(int)
        prob_img = np.zeros((res_crop, res_crop, 3))
        for x, y in xy:
            prob_img[y, x, 2] = 1.
        cv2.imshow('dist', prob_img)

    for name in range(16):
        param = [name, 2]
        cv2.setMouseCallback(str(name), mouse_cb, param)

    while True:
        print()
        key = cv2.waitKey()
        if key == ord('q'):
            cv2.destroyAllWindows()
            quit()
        elif key == ord('a'):
            data_i = (data_i - 1) % len(data)
            break
        elif key == ord('d'):
            data_i = (data_i + 1) % len(data)
            break
        elif key == ord('x'):
            data_i = np.random.randint(len(data))
            break
