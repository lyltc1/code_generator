""" Symmetry annotation tool for models in datasets with BOP format

Using models upsampled to more than 2^16 points, e.g. obj_000001.obj, the tool will generate
directory named models_GT_2_color, which contains file Class_CorresPoint0000XX.txt and obj_0000XX.ply

"""

from bop_toolkit_lib import dataset_params
from bop_toolkit_lib import inout
import glob
import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
import os
import json
import cv2
import warnings

# PARAMETERS.
################################################################################
p = {
  # Folder containing the BOP datasets.
  'datasets_path': '/media/lyltc/mnt2/dataset/zebrapose/zebrapose_data',

  # See dataset_params.py for options.
  'dataset': 'tless',
}
################################################################################

class DatasetInfo:
  def __init__(self, p):
    # Load dataset parameters.
    dp_model = dataset_params.get_model_params(p['datasets_path'], p['dataset'])
    # Load meta info about the models (including symmetries).
    self.dataset_name = p['dataset']
    self.models_info = inout.load_json(dp_model['models_info_path'], keys_to_int=True)
    self.obj_tpath = dp_model['model_tpath'].replace('.ply', '.obj')
    self.obj_ids = dp_model['obj_ids']


class Settings:
  def __init__(self):
    self.bg_color = gui.Color(1, 1, 1)
    self.show_world_axes = False
    self.show_obj_axes = False

    self.obj_material = rendering.MaterialRecord()
    self.obj_material.base_color = [0.9, 0.3, 0.3, 1.0]
    self.obj_material.shader = "defaultUnlit"

    self.axis_material = rendering.MaterialRecord()
    self.axis_material.shader = "unlitLine"
    self.axis_material.line_width = 3

class AppWindow:


  def _on_layout(self, layout_context):
    r = self.window.content_rect
    self._scene.frame = r
    width = 17 * layout_context.theme.font_size
    height = min(r.height, self._settings_panel.calc_preferred_size(layout_context, gui.Widget.Constraints()).height+30)
    self._settings_panel.frame = gui.Rect(r.get_right() - width, r.y, width, height)

  def __init__(self, width, height, dataset_info):
    # ---- basic setting ----
    self.settings = Settings()
    self.dataset_info = dataset_info
    self.window = gui.Application.instance.create_window( "Symmetry annotation tool", width, height)
    w = self.window  # to make the code more concise
    w.set_on_layout(self._on_layout)

    #
    # ---- 3D widget ----
    self._scene = gui.SceneWidget()
    self._scene.scene = rendering.Open3DScene(w.renderer)

    # ---- Settings panel ----
    em = w.theme.font_size
    self._settings_panel = gui.Vert(0, gui.Margins(0.25 * em, 0.25 * em, 0.25 * em, 0.25 * em))

    self._label_1 = gui.Label(f'Dataset : {self.dataset_info.dataset_name}')
    self._label_2 = gui.Label(f'Containing objects : {len(self.dataset_info.obj_ids)}')

    self._collapsable_vert_1 = gui.CollapsableVert("Visual control", 0.33 * em, gui.Margins(em, 0, 0, 0))
    self._collapsable_vert_1.set_is_open(True)
    self._world_axes_checkbox = gui.Checkbox("Show world axes")
    self._world_axes_checkbox.set_on_checked(self._on_world_axes_checkbox)
    self._obj_axes_checkbox = gui.Checkbox("Show obj axes")
    self._obj_axes_checkbox.set_on_checked(self._on_obj_axes_checkbox)

    self._collapsable_vert_2 = gui.CollapsableVert("Object choose panel", 0.33 * em, gui.Margins(em, 0, 0, 0))
    self._collapsable_vert_2.set_is_open(True)
    self._mesh_list = gui.ListView()
    self._mesh_list.set_items(['obj_' + f'{i:06}' for i in self.dataset_info.obj_ids])
    self._mesh_list.set_max_visible_items(10)
    self._choose_mesh_button = gui.Button("Choose Mesh")
    self._choose_mesh_button.set_on_clicked(self._choose_mesh)

    self._object_info_label = gui.Label("object information:")

    # ---- Layout ----
    self.window.add_child(self._scene)
    self.window.add_child(self._settings_panel)
    self._settings_panel.add_child(self._label_1)
    self._settings_panel.add_child(self._label_2)
    self._settings_panel.add_child(self._collapsable_vert_1)
    self._collapsable_vert_1.add_child(self._world_axes_checkbox)
    self._collapsable_vert_1.add_child(self._obj_axes_checkbox)
    self._settings_panel.add_child(self._collapsable_vert_2)
    self._collapsable_vert_2.add_child(self._mesh_list)
    self._collapsable_vert_2.add_child(self._choose_mesh_button)
    self._settings_panel.add_fixed(0.5 * em)
    self._settings_panel.add_child(self._object_info_label)


    # ---- Menu ----

    # ---- tool init setting ----
    self._apply_settings()

  def _apply_settings(self):
    # ---- set background ----
    bg_color = [self.settings.bg_color.red, self.settings.bg_color.green,
                self.settings.bg_color.blue, self.settings.bg_color.alpha]
    self._scene.scene.set_background(bg_color)
    # ---- view world axis
    self._scene.scene.show_axes(self.settings.show_world_axes)
    self._world_axes_checkbox.checked = self.settings.show_world_axes
    # ---- view object axis
    if self.settings.show_obj_axes:
      self.show_obj_axes()
    else:
      self.clear_obj_axes()
    self._obj_axes_checkbox.checked = self.settings.show_obj_axes

  def _on_world_axes_checkbox(self, show):
    self.settings.show_world_axes = show
    self._apply_settings()

  def _on_obj_axes_checkbox(self, show):
    self.settings.show_obj_axes = show
    self._apply_settings()

  def _choose_mesh(self):
    # ---- visualize the object ----
    selected_obj_id = self._mesh_list.selected_index + 1
    mesh = o3d.io.read_triangle_mesh(self.dataset_info.obj_tpath.format(obj_id=selected_obj_id))
    mesh.vertices = o3d.utility.Vector3dVector(np.asarray(mesh.vertices) / 1000)  # convert mm to meter
    self._scene.scene.clear_geometry()
    self._scene.scene.add_geometry(self._mesh_list.selected_value, mesh, self.settings.obj_material)
    model_info = self.dataset_info.models_info[selected_obj_id]
    if 'symmetries_discrete' in model_info and 'symmetries_continuous' in model_info:
      self._object_info_label.text = 'object information: \nthe object is both discrete and ' \
                                               'continuous symmetries, cannot handle it'
    elif 'symmetries_continuous' in model_info:
      self._object_info_label.text = 'object information: \nthe object is continuous symmetries, ' \
                                               'cannot handle it now'
    elif 'symmetries_discrete' in model_info:
      num_discrete = len(model_info['symmetries_discrete'])
      self._object_info_label.text = f'object information: \nthe object has {num_discrete + 1} discrete'
    else:
      self._object_info_label.text = 'the object is not symmetry'


  def show_obj_axes(self):
    points = np.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]) * 0.125
    lines = [[0, 1], [0, 2], [0, 3]]
    colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    line_set = o3d.geometry.LineSet(points=o3d.utility.Vector3dVector(points), lines=o3d.utility.Vector2iVector(lines),)
    line_set.colors = o3d.utility.Vector3dVector(colors)
    self._scene.scene.add_geometry("obj_axis", line_set, self.settings.axis_material)

  def clear_obj_axes(self):
    if self._scene.scene.has_geometry('obj_axis'):
      self._scene.scene.remove_geometry('obj_axis')

def main():
  """ Init GUI and run with pre-defined parameters """
  gui.Application.instance.initialize()
  w = AppWindow(2048, 1536, DatasetInfo(p))
  gui.Application.instance.run()


if __name__ == "__main__":
  main()
















































