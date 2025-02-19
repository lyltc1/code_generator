import os
import copy
from os.path import join
import numpy as np
import open3d as o3d
from open3d.camera import PinholeCameraIntrinsic


# Get the project path
project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# Define the path to the PLY file
ply_path = join(project_path, 'data/models_GT_color_v3/lmo/obj_000001.ply')

# Read the PLY file
assert os.path.exists(ply_path), f"PLY file not found at {ply_path}"
ply = o3d.io.read_triangle_mesh(ply_path)
vertex_colors = np.asarray(ply.vertex_colors)
vertex_R = (vertex_colors[:, 0] * 255).astype(int)
vertex_G = (vertex_colors[:, 1] * 255).astype(int)
vertex_B = (vertex_colors[:, 2] * 255).astype(int)
vertex_id = vertex_B * 256 * 256 + vertex_G * 256 + vertex_R

# Read original PLY file
original_ply_path = join(project_path, 'data/bop/lmo/models/obj_000001.ply')
assert os.path.exists(original_ply_path), f"Original PLY file not found at {original_ply_path}"
original_ply = o3d.io.read_triangle_mesh(original_ply_path)

iteration = 16
class_base = 2

# Create the visualizer window once
vis = o3d.visualization.Visualizer()
vis.create_window()
opt = vis.get_render_option()
opt.background_color = np.asarray([1.0, 1.0, 1.0])
view_control = vis.get_view_control()
camera_params = view_control.convert_to_pinhole_camera_parameters()
camera_params.intrinsic = PinholeCameraIntrinsic(1920, 1080, 935.30743609, 935.30743609, 959.5, 539.5)
camera_params.extrinsic = np.array([[-2.62707265e-01,  9.64815762e-01, -1.07442119e-02, -3.56879444e-01],
                                    [ 2.59908707e-01,  6.00373796e-02, -9.63765001e-01, -4.99631222e+00],
                                    [-9.29210610e-01, -2.55980582e-01, -2.66536273e-01, 1.11263999e+02],
                                    [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00, 1.00000000e+00]])
view_control.convert_from_pinhole_camera_parameters(camera_params)

# visualize ply 
vis.clear_geometries()
vis.add_geometry(ply)
view_control.convert_from_pinhole_camera_parameters(camera_params)
vis.poll_events()
vis.update_renderer()
saved_image_path = join(project_path, "output", f"output_ply.png")
vis.capture_screen_image(saved_image_path)

# visualize the original object
vis.clear_geometries()
vis.add_geometry(original_ply)
view_control.convert_from_pinhole_camera_parameters(camera_params)
vis.poll_events()
vis.update_renderer()
saved_image_path = join(project_path, "output", f"output_original.png")
vis.capture_screen_image(saved_image_path)

# visualize colorful coordinate
colors = np.zeros_like(vertex_colors)
vertices = np.asarray(ply.vertices)
vertices_max = vertices.max(axis=0)
vertices_min = vertices.min(axis=0)
normalized_vertices = (vertices - vertices_min) / (vertices_max - vertices_min)
colors = normalized_vertices
ply.vertex_colors = o3d.utility.Vector3dVector(colors)
vis.clear_geometries()
vis.add_geometry(ply)
view_control.convert_from_pinhole_camera_parameters(camera_params)
vis.poll_events()
vis.update_renderer()
saved_image_path = join(project_path, "output", f"coordinate.png")
vis.capture_screen_image(saved_image_path)

# Create coordinate frame
coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=50.0, origin=[0, 0, 0])
vis.clear_geometries()
vis.add_geometry(coordinate_frame)
view_control.convert_from_pinhole_camera_parameters(camera_params)
vis.poll_events()
vis.update_renderer()
saved_image_path = join(project_path, "output", f"frame.png")
vis.capture_screen_image(saved_image_path)

# visualize colorful coordinate and frame
vis.clear_geometries()
vis.add_geometry(coordinate_frame)
vis.add_geometry(ply)
view_control.convert_from_pinhole_camera_parameters(camera_params)
vis.poll_events()
vis.update_renderer()
saved_image_path = join(project_path, "output", f"coordinate_frame.png")
vis.capture_screen_image(saved_image_path)

# visualize the original object and frame
vis.clear_geometries()
vis.add_geometry(coordinate_frame)
vis.add_geometry(original_ply)
view_control.convert_from_pinhole_camera_parameters(camera_params)
vis.poll_events()
vis.update_renderer()
saved_image_path = join(project_path, "output", f"original_frame.png")
vis.capture_screen_image(saved_image_path)

# visualize mask
vis.clear_geometries()
opt.background_color = np.asarray([1.0, 1.0, 1.0])
colors = np.zeros_like(vertex_colors)
black_ply = copy.deepcopy(ply)
black_ply.vertex_colors = o3d.utility.Vector3dVector(colors)
vis.add_geometry(black_ply)
view_control.convert_from_pinhole_camera_parameters(camera_params)
vis.poll_events()
vis.update_renderer()
saved_image_path = join(project_path, "output", f"mask.png")
vis.capture_screen_image(saved_image_path)

# Destroy the visualizer window after the loop
vis.destroy_window()