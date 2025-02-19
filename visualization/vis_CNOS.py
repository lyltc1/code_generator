import os
import copy
from os.path import join
import numpy as np
import open3d as o3d
from open3d.camera import PinholeCameraIntrinsic


# Get the project path
project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# Create the visualizer window once
vis = o3d.visualization.Visualizer()
vis.create_window()
opt = vis.get_render_option()
opt.background_color = np.asarray([1.0, 1.0, 1.0])
view_control = vis.get_view_control()
camera_params = view_control.convert_to_pinhole_camera_parameters()
camera_params.intrinsic = PinholeCameraIntrinsic(1920, 1080, 2000, 2000, 959.5, 539.5)
camera_params.extrinsic = np.array([[-2.62707265e-01,  9.64815762e-01, -1.07442119e-02, -3.56879444e-01],
                                    [ 2.59908707e-01,  6.00373796e-02, -9.63765001e-01, -4.99631222e+00],
                                    [-9.29210610e-01, -2.55980582e-01, -2.66536273e-01, 1.11263999e+02],
                                    [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00, 1.00000000e+00]])
view_control.convert_from_pinhole_camera_parameters(camera_params)


# visualize CNOS
box = o3d.geometry.TriangleMesh.create_box(width=40, 
                                           height=40,
                                           depth=40)
box_colors = np.asarray(box.vertices)
box_colors = (box_colors - box_colors.min(axis=0)) / (box_colors.max(axis=0) - box_colors.min(axis=0))
box.vertex_colors = o3d.utility.Vector3dVector(box_colors)
vis.clear_geometries()
vis.add_geometry(box)
view_control.convert_from_pinhole_camera_parameters(camera_params)
vis.run()
saved_image_path = join(project_path, "output", f"box.png")
vis.capture_screen_image(saved_image_path)

# Destroy the visualizer window after the loop
vis.destroy_window()