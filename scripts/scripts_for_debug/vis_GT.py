# This script helps to vis GT and binary code
""" Usage: python vis_GT.py """
import math
import argparse
from pathlib import Path
import numpy as np
import cv2
from utils.config import config
from utils.instance import BopInstanceDataset
from utils.std_auxs import GTLoader, RgbLoader, MaskLoader, RandomRotatedMaskCrop
parser = argparse.ArgumentParser()
parser.add_argument('--dataset_path', default='/home/lyltc/git/ZebraPose/datasets/BOP_DATASETS/tless', help='path to dataset folder')
parser.add_argument('--data_folder', default='test_primesense', help='which training data')
parser.add_argument('--pbr', default=False)
parser.add_argument('--test', default=True)
args = parser.parse_args()

res_crop = 224
dataset_path = Path(args.dataset_path)
dataset = dataset_path.name.split('/')[-1]

auxs = [RgbLoader(), MaskLoader(),GTLoader(), RandomRotatedMaskCrop(res_crop, rgb_interpolation=cv2.INTER_LINEAR)]
cfg = config[dataset]
obj_ids = sorted([int(p.name[4:10]) for p in (dataset_path / cfg.model_folder).glob('*.ply')])
dataset_args = dict(dataset_root=dataset_path, auxs=auxs, cfg=cfg, obj_ids=obj_ids)
data = BopInstanceDataset(**dataset_args, pbr=args.pbr, test=args.test)

window_names = ['rgb_crop', 'mask_visib_crop', 'code_crop']
for j, name in enumerate(window_names):
    cv2.namedWindow(name, cv2.WINDOW_KEEPRATIO)
    cv2.resizeWindow(name, res_crop, res_crop)
    cv2.moveWindow(name, 1 + 250 * j, 1 + 250 *0)
for j in range(16):
    row, col = j // 6, j % 6
    cv2.namedWindow(str(j), cv2.WINDOW_KEEPRATIO)
    cv2.resizeWindow(str(j), res_crop, res_crop)
    cv2.moveWindow(str(j), 1 + 250 * col, 280 + 250 * row)

def RGB_to_class_id(RGB_image):
    # the input of this function has to be numpy array, due to the bit shift
    RGB_image = RGB_image.astype(int)
    class_id_R = RGB_image[:,:,0]
    class_id_G = np.left_shift(RGB_image[:,:,1], 8)
    class_id_B = np.left_shift(RGB_image[:,:,2], 16)
    class_id_image = class_id_B + class_id_G + class_id_R
    return class_id_image

def class_id_to_class_code_images(class_id_image, class_base=2, iteration=16, number_of_class=65536):
    """
        class_id_image: 2D numpy array
    """
    if class_base ** iteration != number_of_class:
        raise ValueError('this combination of base and itration is not possible')
    iteration = int(iteration)
    class_id_image = class_id_image.astype(int)
    class_code_images = np.zeros((class_id_image.shape[0], class_id_image.shape[1], iteration))
    bit_step = math.log2(class_base)
    for i in range(iteration):
        shifted_value_1 = np.right_shift(class_id_image, int(bit_step * (iteration - i - 1)))
        shifted_value_2 = np.right_shift(class_id_image, int(bit_step * (iteration - i)))
        class_code_images[:, :, i] = shifted_value_1 - shifted_value_2 * (2 ** bit_step)
    return class_code_images

print()
print('With an opencv window active:')
print("press 'a', 'd' and 'x'(random) to get a new input image,")
print("press 'q' to quit.")
data_i = 0
while True:
    print()
    print('------------ new input -------------')
    inst = data[data_i]
    obj_idx = inst['obj_idx']
    print(f'i: {data_i}, obj_id: {obj_ids[obj_idx]}')
    rgb_crop = inst['rgb_crop']
    mask_visib_crop = inst['mask_visib_crop']
    code_crop = inst['code_crop']
    if not code_crop.any():
        data_i = np.random.randint(len(data))
        continue
    class_id_image = RGB_to_class_id(code_crop)
    class_code_images = class_id_to_class_code_images(class_id_image)
    cv2.imshow('rgb_crop', rgb_crop[..., ::-1])
    cv2.imshow('mask_visib_crop', mask_visib_crop)
    cv2.imshow('code_crop', code_crop[..., ::-1])
    for j in range(16):
        cv2.imshow(str(j), class_code_images[..., j])
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
