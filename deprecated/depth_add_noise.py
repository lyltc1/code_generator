import numpy as np
import cv2

im_list = ["1.png", "2.png"]
noise_list = [1, 2, 3]
downsample_list = [2, 4, 8]
for im in im_list:
    for noise in noise_list:
        for downsample in downsample_list:
            depth_1 = cv2.imread(im)
            # 设置添加白色噪声的坐标位置
            num_salt = np.ceil(0.008 * depth_1.size * noise)
            coords = [np.random.randint(0, i - 1, int(num_salt)) for i in depth_1.shape[0:2]]
            depth_1[coords[0], coords[1]] = 255
            # 设置添加黑色噪声的坐标位置
            num_pepper = np.ceil(0.008 * depth_1.size * noise)
            coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in depth_1.shape[0:2]]
            depth_1[coords[0], coords[1]] = 0
            # 设置添加随机噪声
            coords = np.where(np.logical_and(depth_1[:, :, 0] != 0, depth_1[:, :, 0] != 255))
            depth_1[coords[0], coords[1]] += (np.random.normal(0, 1, coords[0].size)[:, None]).astype('uint8')
            # 降采样
            depth_2 = np.ones_like(depth_1) * 255
            depth_2[::downsample, ::downsample] = depth_1[::downsample, ::downsample]

            cv2.imwrite("noise"+str(noise)+"down"+str(downsample)+im, depth_2)


