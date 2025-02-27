import os
import copy
from os.path import join
import numpy as np
import open3d as o3d
from open3d.camera import PinholeCameraIntrinsic


SAVE_IMAGES = True

# Get the project path
project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# Define the path to the PLY file
ply_path = join(project_path, 'data/models_GT_color_v3/lmo/obj_000001.ply')

# Read the PLY file
ply = o3d.io.read_triangle_mesh(ply_path)
vertex_colors = np.asarray(ply.vertex_colors)
vertex_R = (vertex_colors[:, 0] * 255).astype(int)
vertex_G = (vertex_colors[:, 1] * 255).astype(int)
vertex_B = (vertex_colors[:, 2] * 255).astype(int)
vertex_id = vertex_B * 256 * 256 + vertex_G * 256 + vertex_R

# Read original PLY file
original_ply_path = join(project_path, 'data/bop/lmo/models/obj_000001.ply')
original_ply = o3d.io.read_triangle_mesh(original_ply_path)

iteration = 16
class_base = 2

# Create the visualizer window once
vis = o3d.visualization.Visualizer()
vis.create_window()
opt = vis.get_render_option()
opt.background_color = np.asarray([0.5, 0.5, 0.5])
view_control = vis.get_view_control()
camera_params = view_control.convert_to_pinhole_camera_parameters()
camera_params.intrinsic = PinholeCameraIntrinsic(1920, 1080, 935.30743609, 935.30743609, 959.5, 539.5)
camera_params.extrinsic = np.array([[-2.62707265e-01,  9.64815762e-01, -1.07442119e-02, -3.56879444e-01],
                                    [ 2.59908707e-01,  6.00373796e-02, -9.63765001e-01, -4.99631222e+00],
                                    [-9.29210610e-01, -2.55980582e-01, -2.66536273e-01, 1.11263999e+02],
                                    [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00, 1.00000000e+00]])
view_control.convert_from_pinhole_camera_parameters(camera_params)

# visualize the original object
if SAVE_IMAGES:
    vis.add_geometry(original_ply)
    view_control.convert_from_pinhole_camera_parameters(camera_params)
    vis.poll_events()
    vis.update_renderer()
    saved_image_path = join(project_path, "output", f"output_original.png")
    vis.capture_screen_image(saved_image_path)

    for i in range(0, 16):
        colors = np.zeros_like(vertex_colors)
        shifted_value_1 = np.right_shift(vertex_id, (iteration - i - 1))
        shifted_value_2 = np.right_shift(vertex_id, (iteration - i))
        colors[:] = np.expand_dims(shifted_value_1 - shifted_value_2 * 2, axis=1)
        ply.vertex_colors = o3d.utility.Vector3dVector(colors)
        
        # Update the geometry in the visualizer
        vis.clear_geometries()
        vis.add_geometry(ply)
        view_control.convert_from_pinhole_camera_parameters(camera_params)
        
        if SAVE_IMAGES:
            vis.poll_events()
            vis.update_renderer()
            saved_image_path = join(project_path, "output", f"v1_{i}_.png")
            vis.capture_screen_image(saved_image_path)
        else:
            vis.poll_events()
            vis.update_renderer()
else:
    vis.add_geometry(original_ply)
    vis.run()
    print("You have change the original view points, you can update it into the code several lines before.")
    camera_params = view_control.convert_to_pinhole_camera_parameters()
    print(f"{camera_params.intrinsic=}")
    print(f"{camera_params.extrinsic=}")




# Destroy the visualizer window after the loop
vis.destroy_window()