""" This script trys to recover pose based on ground truth code """
import argparse
import numpy as np
import cv2

from utils.config import config, root
from utils.instance import BopInstanceDataset
from utils.std_auxs import GTLoader, RgbLoader, MaskLoader, RandomRotatedMaskCrop, ObjCoordAux, PosePresentationAux
from utils.class_id_encoder_decoder import RGB_to_class_id, class_id_to_class_code_images, load_decoders
from utils.obj import load_objs
from utils.renderer import project, ObjCoordRenderer
from utils.utils import timer
from utils import pose_est
from utils.allo_pose_utils import ortho6d_to_mat, allocentric_to_egocentric

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='tless')
parser.add_argument('--pbr', default=False)
parser.add_argument('--test', default=True)
args = parser.parse_args()

res_crop = 224
dataset = args.dataset
dataset_path = root / 'data/bop' / dataset

cfg = config[dataset]
cfg.dataset = dataset
objs, obj_ids = load_objs(cfg)
renderer = ObjCoordRenderer(objs, [k for k in objs.keys()], res_crop)
decoders = load_decoders(cfg.models_GT_color_folder / dataset)

auxs = [RgbLoader(), MaskLoader(mask_type='mask'), GTLoader(),
        RandomRotatedMaskCrop(res_crop, mask_key='mask', crop_keys=('rgb', 'mask', 'code'), rgb_interpolation=cv2.INTER_LINEAR),
        PosePresentationAux(res_crop),
        ObjCoordAux(objs, res_crop)]

dataset_args = dict(dataset_root=dataset_path, auxs=auxs, cfg=cfg, obj_ids=obj_ids)
data = BopInstanceDataset(**dataset_args, pbr=args.pbr, test=args.test)

window_names = ['rgb_crop', 'mask_crop', 'code_crop', 'correspondence', 'gt_pose', 'allo_SITE_pose']
for j, name in enumerate(window_names):
    cv2.namedWindow(name, cv2.WINDOW_KEEPRATIO)
    cv2.resizeWindow(name, res_crop, res_crop)
    cv2.moveWindow(name, 1 + 250 * j, 1 + 250 * 0)
for j in range(16):
    row, col = j // 6, j % 6
    cv2.namedWindow(str(j), cv2.WINDOW_KEEPRATIO)
    cv2.resizeWindow(str(j), res_crop, res_crop)
    cv2.moveWindow(str(j), 1 + 250 * col, 300 + 250 * row)

def mouse_cb(event, x, y, flags=0, *param):
    global last_mouse_pos
    if flags & cv2.EVENT_FLAG_CTRLKEY:
        return
    bit = param[0][0]
    last_mouse_pos = x, y
    class_code = class_code_img[y, x]
    if not np.any(class_code):
        return
    code_length = len(class_code)
    class_id = 0
    for i in range(bit + 1):
        class_id = class_id + class_code[i] * (2 ** (code_length - 1 - i))
    class_id_max = class_id
    for i in range(bit + 1, code_length):
        class_id_max += 1 * (2 ** (code_length - 1 - i))
    class_id, class_id_max = int(class_id), int(class_id_max)
    possible_points = []
    for i in range(class_id, class_id_max + 1):
        possible_points.extend(corresponding[i])
    possible_points = np.array(possible_points)
    xy = project(possible_points, inst['cam_R_obj'], inst['cam_t_obj'], inst['K_crop'])
    xy = xy.round().astype(int)
    prob_img = np.zeros((res_crop, res_crop, 3))
    for x, y in xy:
        if 0 <= x < res_crop and 0 <= y < res_crop:
            # only draw points in camera view
            prob_img[y, x, 0] = 1.
    cv2.imshow('correspondence', prob_img[..., ::-1])

def visualize_pose(R, t, K, window_name):
    render = renderer.render(obj_id, K, R, t)
    render_mask = render[..., 3] == 1.
    pose_img = rgb_crop.copy()
    pose_img[render_mask] = pose_img[render_mask] * 0.5 + render[..., :3][render_mask] * 0.25 * 255 + 0.25 * 255
    cv2.imshow(window_name, pose_img[..., ::-1])

def estimate_pose(bit=15):
    with timer('pnp ransac'):
        R, t = pose_est.compare_different_pnp(mask_crop, class_code_img, corresponding, inst['K_crop'], bit=bit,
                                              visualize=False, gt_R=inst['cam_R_obj'], gt_t=inst['cam_t_obj'],
                                              rgb_crop=rgb_crop, renderer=renderer, obj_id=obj_id)
    # visualize_pose(R, t, inst['K_crop'], 'est_pose')
    visualize_pose(inst['cam_R_obj'], inst['cam_t_obj'], inst['K_crop'], 'gt_pose')

print()
print('With an opencv window active:')
print("press 'a', 'd' and 'x'(random) to get a new input image,")
print("press 'q' to quit.")
data_i = 155  # 740
while True:
    print('------------ new input -------------')
    inst = data[data_i]
    if not inst['code_crop'].any():
        # if not generated class code, skip it
        data_i = np.random.randint(len(data))
        continue
    obj_id = inst['obj_id']
    print(f'i: {data_i}, obj_id: {obj_id}')
    RGB_code_crop = inst['code_crop']
    cv2.imshow('code_crop', RGB_code_crop[..., ::-1])
    rgb_crop = inst['rgb_crop']
    cv2.imshow('rgb_crop', rgb_crop[..., ::-1])
    mask_crop = inst['mask_crop']
    cv2.imshow('mask_crop', mask_crop)
    # ---- vis object coordinate ----
    obj_coord = inst['obj_coord']
    render_mask = obj_coord[..., 3] == 1.
    obj_coord[..., :3][render_mask] = obj_coord[..., :3][render_mask] * 0.5 + 0.5
    cv2.imshow('correspondence', obj_coord[..., 2::-1])
    class_id_img = RGB_to_class_id(RGB_code_crop)
    class_code_img = class_id_to_class_code_images(class_id_img)  # [h, w, bit]
    for j in range(16):
        cv2.imshow(str(j), class_code_img[..., j])
    corresponding = decoders[inst['obj_id']]['corresponding']
    last_mouse_pos = 0, 0
    for name in range(16):
        param = [name]
        cv2.setMouseCallback(str(name), mouse_cb, param)
    cv2.setMouseCallback('rgb_crop', mouse_cb, [15])

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
        elif key == ord('c'):
            print('compare_different_pnp:')
            estimate_pose(bit=4)
        elif key == ord('h'):
            print('visulize SITE_c')
            c = inst['SITE'][:2] * res_crop + (res_crop - 1) / 2
            z = inst['SITE'][2] * res_crop / (inst['AABB_crop'][2] - inst['AABB_crop'][0])
            recover_t = np.matmul(np.linalg.inv(inst['K_crop']), np.row_stack((c, 1))) * z
            allo_R_6d = inst['allo_rot6d']
            allo_R = ortho6d_to_mat(allo_R_6d)
            ello_pose = allocentric_to_egocentric(np.column_stack((allo_R, recover_t)))
            recover_R = ello_pose[:, :3]

            print("recover:", ello_pose)
            print('gt:', inst['cam_R_obj'], inst['cam_t_obj'])
            visualize_pose(inst['cam_R_obj'], inst['cam_t_obj'], inst['K_crop'], 'gt_pose')
            visualize_pose(inst['cam_R_obj'], inst['cam_t_obj'], inst['K_crop'], 'allo_SITE_pose')

