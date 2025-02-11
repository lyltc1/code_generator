import numpy as np
import cv2
from scipy.spatial.transform import Rotation
from utils.class_id_encoder_decoder import class_code_to_class_id_and_class_id_max_images
from utils.utils import timer
from utils.renderer import project


def compare_different_pnp(mask_prob, class_code_prob, corresponding, K, bit=15, n_sample=1, visualize=False,
                        gt_R=None, gt_t=None, rgb_crop=None, n_correspondence=100, renderer=None, obj_id=None):
    """
    (1) compute belief map based on mask_prob and class_code_prob
    (2) sample p2d from belief map
    (3) sample correspondence
    (4) use p3p et al. to get pose

    :param mask_prob: (r,r) dtype=float64 0 ~ 1, the predict mask prob
    :param class_code_prob: (r,r,16) dtype=float64 min=0.0 max=1.0, predicted class code prob
    :param corresponding dict( class_id: list(list_of_xyz))
    :param K: camera intrinsics
    :param bit: from 0 to which bit is used in this function
    :param gt_R: ground truth R, used for visualization
    :param gt_t: ground truth t, used for visualization
    """
    r = mask_prob.shape[0]
    with timer('calcu class code'):
        pred_class_code = np.zeros_like(class_code_prob, dtype=int)
        pred_class_code[class_code_prob > 0.5] = 1
        class_id_img, class_id_max_img = class_code_to_class_id_and_class_id_max_images(pred_class_code, bit)
    with timer('claculate belief map'):
        belief_map = mask_prob.copy()
        for i in range(bit + 1):
            belief_map *= (4 * class_code_prob[..., i] * (class_code_prob[..., i] - 1) + 1)
        if visualize:
            cv2.imshow('belief_map', belief_map)
            cv2.waitKey()
    with timer('sample p2d from belief map'):
        belief_map = np.power(belief_map, 1.5)
        belief_map_cumsum = belief_map.cumsum()
        belief_map_cumsum /= belief_map_cumsum[-1]
        idx = np.searchsorted(belief_map_cumsum, np.random.rand(n_sample, n_correspondence))
        yy = np.arange(r)
        xx, yy = np.meshgrid(yy, yy)
        xx, yy = (v.reshape(-1) for v in (xx, yy))
        img_pts = np.column_stack((yy, xx))
        p2d = img_pts[idx]  # (n_pose, 4, 2 yx)
        if visualize:  # visualize sampled p2d
            sampled_p2d_img = np.zeros((r, r), dtype=np.uint8)
            p2d_yy, p2d_xx = p2d.reshape(-1, 2).T
            np.add.at(sampled_p2d_img, (p2d_yy, p2d_xx), 1)
            sampled_p2d_img *= 255 // sampled_p2d_img.max()
            cv2.imshow('sampled_p2d_img', sampled_p2d_img)
            cv2.waitKey()

    p3d = np.zeros((n_sample, n_correspondence, 3))
    with timer('sample correspondence'):
        sampled_class_id = class_id_img[p2d[..., 0], p2d[..., 1]]
        sampled_class_id_max = class_id_max_img[p2d[..., 0], p2d[..., 1]]
        for i in range(n_sample):
            for j in range(n_correspondence):
                possible_points = []
                for k in range(sampled_class_id[i, j], sampled_class_id_max[i, j] + 1):
                    possible_points.extend(corresponding[k])
                print(f'possible point for sample {i} point {j}')
                print(possible_points)
                p3d[i, j] = possible_points[np.random.randint(len(possible_points))]
    # ---- debug for pnp ----
    win_i = 0
    poses = np.zeros((n_sample, 3, 4))
    poses_mask = np.zeros(n_sample, dtype=bool)

    # ---- part 1: debug for pnp algorithm which only use 4 points ----
    vis_point = 4
    for method in ['AP3P', 'P3P', 'SQPNP', 'EPNP']:
        win_correspondence_name = 'cor_' + method + 'point_num_4' + 'bit'+str(bit)
        win_pose_name = 'pose' + method + 'point_num_4' + 'bit'+str(bit)
        print(win_pose_name)
        cv2.namedWindow(win_correspondence_name, cv2.WINDOW_KEEPRATIO)
        cv2.namedWindow(win_pose_name, cv2.WINDOW_KEEPRATIO)
        cv2.resizeWindow(win_correspondence_name, 300, 300)
        cv2.resizeWindow(win_pose_name, 300, 300)
        cv2.moveWindow(win_correspondence_name, 1 + 315 * win_i, 380)
        cv2.moveWindow(win_pose_name, 1 + 315 * win_i, 710)
        win_i += 1
        with timer('pnp'):
            rotvecs = np.zeros((n_sample, 3))
            p2d = p2d.astype(float)
            for i in range(n_sample):
                ret, rvecs, tvecs = cv2.solvePnP(p3d[i, :4], p2d[i, :4, ::-1], K, None, flags=eval('cv2.SOLVEPNP_' + method))
                if ret:
                    rotvecs[i] = rvecs[:, 0]
                    poses[i, :3, 3:] = tvecs
                    poses_mask[i] = False if np.isnan(np.min(tvecs)) else True
            poses[:, :3, :3] = Rotation.from_rotvec(rotvecs).as_matrix()
        poses, p2d, p3d = [a[poses_mask] for a in (poses, p2d, p3d)]
        with timer('vis p2d p3d'):
            p2d_p3d_img = rgb_crop.copy() // 2
            p2d = p2d.astype(int)
            for p in p2d[0, :vis_point]:  # blue circle: chosed p2d points
                cv2.circle(p2d_p3d_img, (p[1], p[0]), 7, (0, 0, 255), 2, cv2.LINE_AA)  # blue
            print('blue circle: chosed p2d points\n', p2d[0, :, ::-1])
            projected_est = project(p3d[0], poses[0, :, :3], poses[0, :, 3:], K).round().astype(int)
            for p in projected_est[:vis_point]:  # red circle: projected p3d by est pose
                cv2.circle(p2d_p3d_img, (p[0], p[1]), 4, (255, 0, 0), 2, cv2.LINE_AA)  # red
            print('red circle: projected p3d by est pose\n', projected_est)
            projected_gt = project(p3d[0], gt_R, gt_t, K).round().astype(int)
            for p in projected_gt[:vis_point]:  # green circle: projected p3d by gt pose
                cv2.circle(p2d_p3d_img, (p[0], p[1]), 2, (0, 255, 0), 3, cv2.LINE_AA)  # green
            print('green circle: projected p3d by gt pose\n', projected_gt)
            cv2.imshow(win_correspondence_name, p2d_p3d_img[..., ::-1])
        with timer('vis pose'):
            # render = renderer.render(obj_id, K, poses[0, :, :3], poses[0, :, 3:])
            # TODO now use gt_t to see R
            render = renderer.render(obj_id, K, poses[0, :, :3], gt_t)
            render_mask = render[..., 3] == 1.
            pose_img = rgb_crop.copy()
            pose_img[render_mask] = pose_img[render_mask] * 0.5 + render[..., :3][render_mask] * 0.25 * 255 + 0.25 * 255
            cv2.imshow(win_pose_name, pose_img[..., ::-1])

    # ---- part 2: debug for pnp algorithm which use more than 4 points ----
    vis_point = 100
    for method in ['EPNP', 'SQPNP', ]:
        win_correspondence_name = 'cor' + method + 'poin_n' + str(n_correspondence) + 'bit'+str(bit)
        win_pose_name = 'pos' + method + 'poin_n' + str(n_correspondence) + 'bit'+str(bit)
        print(win_pose_name)
        cv2.namedWindow(win_correspondence_name, cv2.WINDOW_KEEPRATIO)
        cv2.namedWindow(win_pose_name, cv2.WINDOW_KEEPRATIO)
        cv2.resizeWindow(win_correspondence_name, 300, 300)
        cv2.resizeWindow(win_pose_name, 300, 300)
        cv2.moveWindow(win_correspondence_name, 1 + 315 * win_i, 380)
        cv2.moveWindow(win_pose_name, 1 + 315 * win_i, 710)
        win_i += 1
        with timer('pnp'):
            rotvecs = np.zeros((n_sample, 3))
            p2d = p2d.astype(float)
            for i in range(n_sample):
                ret, rvecs, tvecs = cv2.solvePnP(p3d[i], p2d[i, :, ::-1], K, None, flags = eval('cv2.SOLVEPNP_' + method))
                if ret:
                    rotvecs[i] = rvecs[:, 0]
                    poses[i, :3, 3:] = tvecs
                    poses_mask[i] = False if np.isnan(np.min(tvecs)) else True
            poses[:, :3, :3] = Rotation.from_rotvec(rotvecs).as_matrix()
        poses, p2d, p3d = [a[poses_mask] for a in (poses, p2d, p3d)]
        with timer('vis p2d p3d'):
            p2d_p3d_img = rgb_crop.copy() // 2
            p2d = p2d.astype(int)
            for p in p2d[0, :vis_point]:  # blue circle: chosed p2d points
                cv2.circle(p2d_p3d_img, (p[1], p[0]), 7, (0, 0, 255), 2, cv2.LINE_AA)  # blue
            print('blue circle: chosed p2d points\n', p2d[0, :, ::-1])
            projected_est = project(p3d[0], poses[0, :, :3], poses[0, :, 3:], K).round().astype(int)
            for p in projected_est[:vis_point]:  # red circle: projected p3d by est pose
                cv2.circle(p2d_p3d_img, (p[0], p[1]), 4, (255, 0, 0), 2, cv2.LINE_AA)  # red
            print('red circle: projected p3d by est pose\n', projected_est)
            projected_gt = project(p3d[0], gt_R, gt_t, K).round().astype(int)
            for p in projected_gt[:vis_point]:  # green circle: projected p3d by gt pose
                cv2.circle(p2d_p3d_img, (p[0], p[1]), 2, (0, 255, 0), 3, cv2.LINE_AA)  # green
            print('green circle: projected p3d by gt pose\n', projected_gt)
            cv2.imshow(win_correspondence_name, p2d_p3d_img[..., ::-1])
        with timer('vis pose'):
            # render = renderer.render(obj_id, K, poses[0, :, :3], poses[0, :, 3:])
            # TODO now use gt_t to see R
            render = renderer.render(obj_id, K, poses[0, :, :3], gt_t)
            render_mask = render[..., 3] == 1.
            pose_img = rgb_crop.copy()
            pose_img[render_mask] = pose_img[render_mask] * 0.5 + render[..., :3][render_mask] * 0.25 * 255 + 0.25 * 255
            cv2.imshow(win_pose_name, pose_img[..., ::-1])
    # ---- part 3: debug for other algorithm which use more than 4 points ----
    vis_point = 100
    for method in ['Ransac']:
        win_correspondence_name = 'cor' + method + 'poin_n' + str(n_correspondence) + 'bit'+str(bit)
        win_pose_name = 'pos' + method + 'poin_n' + str(n_correspondence) + 'bit'+str(bit)
        print(win_pose_name)
        cv2.namedWindow(win_correspondence_name, cv2.WINDOW_KEEPRATIO)
        cv2.namedWindow(win_pose_name, cv2.WINDOW_KEEPRATIO)
        cv2.resizeWindow(win_correspondence_name, 300, 300)
        cv2.resizeWindow(win_pose_name, 300, 300)
        cv2.moveWindow(win_correspondence_name, 1 + 315 * win_i, 380)
        cv2.moveWindow(win_pose_name, 1 + 315 * win_i, 710)
        win_i += 1
        with timer('pnp'):
            rotvecs = np.zeros((n_sample, 3))
            p2d = p2d.astype(float)
            for i in range(n_sample):
                ret, rvecs, tvecs, inliers = eval('cv2.solvePnP' + method + '(p3d[i], p2d[i, :, ::-1], K, None)')
                if ret:
                    rotvecs[i] = rvecs[:, 0]
                    poses[i, :3, 3:] = tvecs
                    poses_mask[i] = False if np.isnan(np.min(tvecs)) else True
            poses[:, :3, :3] = Rotation.from_rotvec(rotvecs).as_matrix()
        poses, p2d, p3d = [a[poses_mask] for a in (poses, p2d, p3d)]
        with timer('vis p2d p3d'):
            p2d_p3d_img = rgb_crop.copy() // 2
            p2d = p2d.astype(int)
            for p in p2d[0, :vis_point]:  # blue circle: chosed p2d points
                cv2.circle(p2d_p3d_img, (p[1], p[0]), 7, (0, 0, 255), 2, cv2.LINE_AA)  # blue
            print('blue circle: chosed p2d points\n', p2d[0, :, ::-1])
            projected_est = project(p3d[0], poses[0, :, :3], poses[0, :, 3:], K).round().astype(int)
            for p in projected_est[:vis_point]:  # red circle: projected p3d by est pose
                cv2.circle(p2d_p3d_img, (p[0], p[1]), 4, (255, 0, 0), 2, cv2.LINE_AA)  # red
            print('red circle: projected p3d by est pose\n', projected_est)
            projected_gt = project(p3d[0], gt_R, gt_t, K).round().astype(int)
            for p in projected_gt[:vis_point]:  # green circle: projected p3d by gt pose
                cv2.circle(p2d_p3d_img, (p[0], p[1]), 2, (0, 255, 0), 3, cv2.LINE_AA)  # green
            print('green circle: projected p3d by gt pose\n', projected_gt)
            cv2.imshow(win_correspondence_name, p2d_p3d_img[..., ::-1])
        with timer('vis pose'):
            # render = renderer.render(obj_id, K, poses[0, :, :3], poses[0, :, 3:])
            # TODO now use gt_t to see R
            render = renderer.render(obj_id, K, poses[0, :, :3], gt_t)
            render_mask = render[..., 3] == 1.
            pose_img = rgb_crop.copy()
            pose_img[render_mask] = pose_img[render_mask] * 0.5 + render[..., :3][render_mask] * 0.25 * 255 + 0.25 * 255
            cv2.imshow(win_pose_name, pose_img[..., ::-1])
    return poses[0, :, :3], poses[0, :, 3:]

