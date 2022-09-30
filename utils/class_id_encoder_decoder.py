import math
from pathlib import Path
import json
import os
import numpy as np

def RGB_to_class_id(RGB_image):
    # the input of this function has to be numpy array, due to the bit shift
    RGB_image = RGB_image.astype(int)
    class_id_R = RGB_image[:, :, 0]
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

def load_decoders(decoder_dir: Path):
    decoders = {}
    obj_ids = sorted([int(p.name[-11:-5]) for p in decoder_dir.glob('*.json')])
    for obj_id in obj_ids:
        decoder_path = str(decoder_dir / f'Class_CorresPoint{obj_id:06d}.json')
        if os.path.exists(decoder_path):
            with open(decoder_path, 'r') as f:
                result = json.loads(f.read())
                decoders[obj_id] = result
    return decoders