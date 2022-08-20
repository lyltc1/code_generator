""" Symmetry annotation tool for models in datasets with BOP format

Using models upsampled to more than 2^16 points, e.g. obj_000001.obj, the tool will generate
directory named models_GT_2_color, which contains file Class_CorresPoint0000XX.txt and obj_0000XX.ply

"""
import copy

from bop_toolkit_lib import inout
from bop_toolkit_lib import transform

import numpy as np
import pandas as pd
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
import os
import json
from sklearn.cluster import KMeans
from k_means_constrained import KMeansConstrained

# PARAMETERS.
################################################################################
p = {
  # Folder containing the BOP datasets.
  'datasets_path': '/media/lyltc/mnt2/dataset/zebrapose/zebrapose_data',

  # See dataset_params.py for options.
  'dataset': 'tless',

  'model_type': None  # tless has cad model type
}

DEBUG = True
################################################################################

def get_model_params(datasets_path, dataset_name, model_type=None):
  """Returns parameters of object models for the specified dataset.

  :param datasets_path: Path to a folder with datasets.
  :param dataset_name: Name of the dataset for which to return the parameters.
  :param model_type: Type of object models.
  :return: Dictionary with object model parameters for the specified dataset.
  """
  # Object ID's.
  obj_ids = {
    'lm': list(range(1, 16)),
    'lmo': [1, 5, 6, 8, 9, 10, 11, 12],
    'tless': list(range(1, 31)),
    'tudl': list(range(1, 4)),
    'tyol': list(range(1, 22)),
    'ruapc': list(range(1, 15)),
    'icmi': list(range(1, 7)),
    'icbin': list(range(1, 3)),
    'itodd': list(range(1, 29)),
    'hbs': [1, 3, 4, 8, 9, 10, 12, 15, 17, 18, 19, 22, 23, 29, 32, 33],
    'hb': list(range(1, 34)),  # Full HB dataset.
    'ycbv': list(range(1, 22)),
    'hope': list(range(1, 29)),
  }[dataset_name]

  # Both versions of the HB dataset share the same directory.
  if dataset_name == 'hbs':
    dataset_name = 'hb'

  # Name of the folder with object models.
  models_folder_name = 'models'
  if model_type is not None:
    models_folder_name += '_' + model_type

  # Path to the folder with object models.
  models_path = os.path.join(datasets_path, dataset_name, models_folder_name)

  p = {
    # ID's of all objects included in the dataset.
    'obj_ids': obj_ids,

    # Path template to an object model file.
    'ply_tpath': os.path.join(models_path, 'obj_{obj_id:06d}.ply'),
    'obj_tpath': os.path.join(models_path, 'obj_{obj_id:06d}.obj'),

    # Path to a file with meta information about the object models.
    'models_info_path': os.path.join(models_path, 'models_info.json')
  }

  return p


class DatasetInfo:
  def __init__(self, p):
    # Load dataset parameters.
    dp_model = get_model_params(p['datasets_path'], p['dataset'], model_type=p['model_type'])
    # Load meta info about the models (including symmetries).
    self.dataset_name = p['dataset']
    self.models_info = inout.load_json(dp_model['models_info_path'], keys_to_int=True)
    self.ply_tpath = dp_model['ply_tpath']
    self.obj_tpath = dp_model['obj_tpath']
    self.obj_ids = dp_model['obj_ids']


class DividedPcd:
  def __init__(self, pcd = None, model_info = None):
    self.origin_pcd = pcd
    self.model_info = model_info
    # ---- ply symmetry info ----
    self.sym_type = list()
    self.sym_type_index = dict()
    self.pairs = dict()
    # ---- pcd final divide list
    self.classified_indice = list()
    self.number_of_iteration = 16
    self.divide_number = 2
    # ---- pcd label sym type ----
    self.threshold = 1.0
    self.clip_x_min = 0.0
    self.clip_x_max = 0.0
    self.clip_y_min = 0.0
    self.clip_y_max = 0.0
    self.clip_z_min = 0.0
    self.clip_z_max = 0.0
    self.clip_index = None
    self.sym_index = None
  def get_sym_type(self):
    return self.sym_type
  def add_sym_type(self, sym_type):
    if not sym_type in self.sym_type:
      self.sym_type.append(sym_type)
  def remove_sym_type(self, sym_type):
    try:
      self.sym_type.remove(sym_type)
    except:
      pass
  def set_pcd(self, pcd):
    self.origin_pcd = pcd
  def set_model_info(self, model_info):
    self.model_info = model_info
  def set_threshold(self, threshold):
    self.threshold = threshold
  def divide_conti_sym(self):
    """ divide point cloud to different based on bop symmetry info

        for all the points in self.origin_pcd, return the sum of distances to all
        the nearest points with continuous transformation
    """
    discrete_steps_count = 36
    for sym in self.model_info['symmetries_continuous']:
      axis = np.array(sym['axis'])
      offset = np.array(sym['offset']).reshape((3, 1))
      all_dists = list()
      for i in range(1, discrete_steps_count):
        R = transform.rotation_matrix(i * 2.0 * np.pi / discrete_steps_count, axis)[:3, :3]
        t = -R.dot(offset) + offset
        transformation_matrix = np.eye(4)
        transformation_matrix[:3, :3] = R
        transformation_matrix[:3, [3]] = t
        pcd_tmp = copy.deepcopy(self.origin_pcd)
        pcd_tmp.transform(transformation_matrix)
        dist = self.origin_pcd.compute_point_cloud_distance(pcd_tmp)
        dist = np.asarray(dist)
        all_dists.append(dist)
      all_dists = np.sum(np.array(all_dists), axis=0)  # (discrete_step, n_point)
      return all_dists
  def divide_n_fold_sym(self, n):
    """ divide point cloud to different based on n_fold info from panel

        for all the points in self.origin_pcd, return the sum of distances to all
        the nearest points with n_fold transformation
    """
    discrete_steps_count = n
    for sym in self.model_info['symmetries_continuous']:
      axis = np.array(sym['axis'])
      offset = np.array(sym['offset']).reshape((3, 1))
      assert np.allclose(offset, np.zeros_like(offset))
      all_dists = list()
      for i in range(1, discrete_steps_count):
        R = transform.rotation_matrix(i * 2.0 * np.pi / discrete_steps_count, axis)[:3, :3]
        t = -R.dot(offset) + offset
        transformation_matrix = np.eye(4)
        transformation_matrix[:3, :3] = R
        transformation_matrix[:3, [3]] = t
        pcd_tmp = copy.deepcopy(self.origin_pcd)
        pcd_tmp.transform(transformation_matrix)
        dist = self.origin_pcd.compute_point_cloud_distance(pcd_tmp)
        dist = np.asarray(dist)
        all_dists.append(dist)
      all_dists = np.sum(np.array(all_dists), axis=0)  # (discrete_step, n_point)
      return all_dists
  def divide_discrete_sym(self):
    """ divide point cloud to different based on bop symmetry info

        for all the points in self.origin_pcd, return the sum of distances to all
        the nearest points with discrete transformation
    """
    symmetries_discrete = self.model_info['symmetries_discrete']
    all_dists = list()
    for sym in symmetries_discrete:
      transformation_matrix = np.asarray(sym).reshape(4,4)
      pcd_tmp = copy.deepcopy(self.origin_pcd)
      pcd_tmp.transform(transformation_matrix)
      dist = self.origin_pcd.compute_point_cloud_distance(pcd_tmp)
      dist = np.asarray(dist)
      all_dists.append(dist)
    all_dists = np.sum(np.array(all_dists), axis=0)  # (discrete_step, n_point)
    return all_dists
  def pair_discrete_sym_pcd(self):
    """ pair points in discrete sym

    1. get the sym part pcd (named local_pcd)
    2. for each symmetry, build the index pair correspondence
    """
    symmetries_discrete = self.model_info['symmetries_discrete']
    local_pcd = self.origin_pcd.select_by_index(self.sym_type_index['discrete sym'])
    pcd_tree = o3d.geometry.KDTreeFlann(local_pcd)
    local_to_origin_index_map = dict(zip(range(len(local_pcd.points)), self.sym_type_index['discrete sym'].tolist()))
    all_group = []
    for sym in symmetries_discrete:
      transformation_matrix = np.asarray(sym).reshape(4, 4)
      local_pcd_transformed = copy.deepcopy(local_pcd)
      local_pcd_transformed.transform(transformation_matrix)
      for i in range(len(local_pcd.points)):
        [k, idx, _] = pcd_tree.search_knn_vector_3d(local_pcd_transformed.points[i], 1)
        all_group.append([i, idx[0]])
    # ---- save the final_pair index in final_pairs, graph and queue is assist variable ----
    final_pairs = list()
    # construct graph, graph[i] store all the associated index with i
    graph = {i:[] for i in range(len(local_pcd.points))}
    for all_pair in all_group:
      graph[all_pair[0]].append(all_pair[1])
      graph[all_pair[1]].append(all_pair[0])
    for i in range(len(local_pcd.points)):
      if i in graph.keys():
        pair = [i]
        queue = graph[i]  # used for broad first search
        graph.pop(i)
        while queue:
          node = queue.pop(0)
          if node in graph.keys():
            pair.append(node)
            queue.extend(graph[node])
            graph.pop(node)
        final_pairs.append(pair)
    # ---- map local_index in final_pairs to global_index
    for pair in final_pairs:
      for i in range(len(pair)):
        pair[i] = local_to_origin_index_map[pair[i]]
    self.pairs['symmetries_discrete'] = final_pairs

  def pair_n_fold_sym_pcd(self, n):
    """ pair points in n_fold sym

    1. get the sym part pcd (named local_pcd)
    2. for each symmetry, build the index pair correspondence
    """
    # ---- parse sym info ----
    discrete_steps_count = n
    sym = self.model_info['symmetries_continuous'][0]
    axis = np.array(sym['axis'])
    offset = np.array(sym['offset']).reshape((3, 1))
    assert np.allclose(offset, np.zeros_like(offset))
    # ---- build KD Tree and all_group (all paired points)
    local_pcd = self.origin_pcd.select_by_index(self.sym_type_index[str(n) + '_fold'])
    local_to_origin_index_map = dict(zip(range(len(local_pcd.points)), self.sym_type_index[str(n) + '_fold'].tolist()))
    pcd_tree = o3d.geometry.KDTreeFlann(local_pcd)
    all_group = []
    for step in range(1, discrete_steps_count):
      R = transform.rotation_matrix(step * 2.0 * np.pi / discrete_steps_count, axis)[:3, :3]
      t = -R.dot(offset) + offset
      transformation_matrix = np.eye(4)
      transformation_matrix[:3, :3] = R
      transformation_matrix[:3, [3]] = t
      local_pcd_transformed = copy.deepcopy(local_pcd)
      local_pcd_transformed.transform(transformation_matrix)
      for i in range(len(local_pcd.points)):
        [k, idx, _] = pcd_tree.search_knn_vector_3d(local_pcd_transformed.points[i], 1)
        all_group.append([i, idx[0]])
    # ---- save the final_pair index in final_pairs, graph and queue is assist variable ----
    final_pairs = list()
    # construct graph, graph[i] store all the associated index with i
    graph = {i:[] for i in range(len(local_pcd.points))}
    for all_pair in all_group:
      graph[all_pair[0]].append(all_pair[1])
      graph[all_pair[1]].append(all_pair[0])
    for i in range(len(local_pcd.points)):
      if i in graph.keys():
        pair = [i]
        queue = graph[i]  # used for broad first search
        graph.pop(i)
        while queue:
          node = queue.pop(0)
          if node in graph.keys():
            pair.append(node)
            queue.extend(graph[node])
            graph.pop(node)
        final_pairs.append(pair)
    # ---- map local_index in final_pairs to global_index
    for pair in final_pairs:
      for i in range(len(pair)):
        pair[i] = local_to_origin_index_map[pair[i]]
    self.pairs[str(n)+'_fold'] = final_pairs

  def pair_continuous_sym_pcd(self, threshold=0.00001):
    """ pair points in continuous sym

    1. get the sym part pcd (named local_pcd)
    2. for each symmetry, build the index pair correspondence
    """
    # ---- parse sym info ----
    sym = self.model_info['symmetries_continuous'][0]
    axis = np.array(sym['axis'])
    offset = np.array(sym['offset']).reshape((3, 1))
    assert np.allclose(offset, np.zeros_like(offset))
    # ---- adjust threshold based on object diameter ----
    threshold = self.model_info['diameter'] * threshold
    # ---- map the coordinate of object to a plane
    local_pcd = self.origin_pcd.select_by_index(self.sym_type_index['continuous sym'])
    local_to_origin_index_map = dict(zip(range(len(local_pcd.points)), self.sym_type_index['continuous sym'].tolist()))
    local_pcd_point = np.asarray(local_pcd.points)
    axis_orthogonal = np.array([1, 1, 1]) - axis
    local_pcd_point_plane = np.column_stack((np.linalg.norm(local_pcd_point * axis_orthogonal,axis=1),local_pcd_point[:,np.where(axis==1)[0]]))
    # ---- save final pairs in final_pairs ----
    pairs_to_be_divided = [list(range(len(local_pcd.points)))]
    final_pairs = []
    while pairs_to_be_divided:
      pair = pairs_to_be_divided.pop(0)
      kmeans = KMeans(n_clusters=2, random_state=0).fit(local_pcd_point_plane[pair])
      sub_pair_1 = np.array(pair)[np.where(kmeans.labels_ == 0)].tolist()
      sub_pair_2 = np.array(pair)[np.where(kmeans.labels_ == 1)].tolist()
      var_sub_pair_1 = np.sum(np.var(local_pcd_point_plane[sub_pair_1],axis=0))
      var_sub_pair_2 = np.sum(np.var(local_pcd_point_plane[sub_pair_1], axis=0))
      if var_sub_pair_1 < threshold or len(sub_pair_1) <= 1:
        final_pairs.append(sub_pair_1)
      else:
        pairs_to_be_divided.append(sub_pair_1)
      if var_sub_pair_2 < threshold or len(sub_pair_2) <= 1:
        final_pairs.append(sub_pair_2)
      else:
        pairs_to_be_divided.append(sub_pair_2)

    for pair in final_pairs:
      for i in range(len(pair)):
        pair[i] = local_to_origin_index_map[pair[i]]
    self.pairs['continuous sym'] = final_pairs

  def pair_no_sym_pcd(self):
    """ pair points in no sym

    it's easy because every points belong to one pair
    """
    index = self.sym_type_index['no sym']
    final_pairs = index[:,np.newaxis].tolist()
    self.pairs['no sym'] = final_pairs

  def choose_one_point_each_pair(self, coordinate_shift=False):
    axis = None  # define which axis is the continuous sym axis, e.g. y: np.ndarray([0, 1, 0])
    axis_orthogonal = None  # define which plane is orthogonal the sym axis, e.g. y: np.ndarray([1, 0, 1])
    axis_keep = None  # define the index of 1 in axis, e.g. 1
    multi = 1  # increase distance in axis direction, benefit for n-system-code generation
    if 'continuous sym' in self.sym_type:
      sym = self.model_info['symmetries_continuous'][0]
      axis = np.array(sym['axis'])
      axis_orthogonal = np.array([1, 1, 1]) - axis
      axis_keep = np.where(axis == 1)[0][0]
      axis_map = dict(zip([0, 1, 2], ['size_x', 'size_y', 'size_z']))
      multi = self.model_info['diameter'] / self.model_info[axis_map[axis_keep]] * 5.0
    # ---- 1.2 modify and save coordinate in select_coordinate and save select_index in select_index
    select_indexes = []
    select_coordinates = []
    points = np.array(self.origin_pcd.points)
    for i, (sym_type, pairs) in enumerate(self.pairs.items()):
      for pair in pairs:
        local_index = np.argmax(np.sum(points[pair], axis=1))
        select_indexes.append(pair[local_index])
        coordinate = points[pair[local_index]]
        if sym_type == 'continuous sym':
          coordinate[(axis_keep + 1) % 3] = np.linalg.norm(coordinate * axis_orthogonal)
          coordinate[(axis_keep + 2) % 3] = 0.0
          if coordinate_shift is True:
            coordinate[axis_keep] = coordinate[axis_keep] * multi + \
                                    0.55 * (multi + 1) * self.model_info[axis_map[axis_keep]]
        select_coordinates.append(coordinate)
    select_coordinates = np.asarray(select_coordinates)
    return select_indexes, select_coordinates

  def divide_pointcloud_iterative(self, select_coordinates, number_of_iteration=16, divide_number=2):
    """ divide pcd in select_coordinates:
        hierarchy_indices = {0:[[0, 1, ..., num_points]                 ],
                             1:[[...], [...]                            ],  # if divide_number = 2
                             2:[[...], [...], [...], [...]              ],  # if divide_number = 2
                             3:[divide_number^3 list                    ],
                             ......
                             number_of_iteration:[divide_number^16 list]]
                            }
        for example: i:[divide_number^i list] corresponding to code range(divide_number^i-1)
        return hierarchy_indices[number_of_iteration]
    """
    # ---- iteratively generate n system code for select_coordinates in local index ----
    num_points = len(select_coordinates)
    hierarchy_indices = {i: list() for i in range(number_of_iteration+1)}
    hierarchy_indices[0].append(list(range(num_points)))
    for i in range(0, number_of_iteration):
      print(f"divide {divide_number}-system-code iteration {i+1}/{number_of_iteration}")
      for j in range(len(hierarchy_indices[i])):
        output_indices = self._divide_cluster(hierarchy_indices[i][j], select_coordinates, divide_number)
        hierarchy_indices[i+1].extend(output_indices)
    return hierarchy_indices[number_of_iteration]
  def _divide_cluster(self, input_indices, coordinates, divide_number):
    """ divide input_indices into {divide_number} output_indices """
    num_points = len(input_indices)
    if num_points < divide_number:
      output_indices = []
      for i in range(num_points):
        output_indices.append([input_indices[i]])
      for i in range(divide_number - num_points):
        output_indices.append([])
      return output_indices
    size_max = pow(divide_number, int(np.ceil(np.log2(num_points)/np.log2(divide_number)))-1)
    clf = KMeansConstrained(n_clusters=divide_number, size_max=size_max, random_state=0)
    clf.fit_predict(coordinates[input_indices])
    output_indices = []
    for i in range(divide_number):
      output_indices.append(np.array(input_indices)[np.where(clf.labels_ == i)].tolist())
    return output_indices
  def indice_completion(self, input_indices):
    # construct graph, graph[i] store all the associated index with i
    graph = {i: [] for i in range(len(self.origin_pcd.points))}
    for pairs in self.pairs.values():
      for pair in pairs:
        for item in pair:
          graph[item].extend(pair)
    # calculate output_indice
    output_indice = []
    for pair in input_indices:
      tmp_pair = []
      for item in pair:
        tmp_pair.extend(graph[item])
      output_indice.append(tmp_pair)
    return output_indice
class Settings:
  def __init__(self):
    # ---- scene visual setting ----
    self.bg_color = gui.Color(1, 1, 1)
    self.show_world_axes_state = False
    self.show_obj_axes_state = False
    # ---- rendering setting ----
    self.obj_material = rendering.MaterialRecord()
    self.obj_material.shader = "defaultLit"
    self.axis_material = rendering.MaterialRecord()
    self.axis_material.shader = "unlitLine"
    self.axis_material.line_width = 3
    self.clip_box_material = rendering.MaterialRecord()
    self.clip_box_material.shader = "unlitLine"
    self.clip_box_material.line_width = 1


class AppWindow:


  def _on_layout(self, layout_context):
    r = self.window.content_rect
    self._scene.frame = r
    width = 17 * layout_context.theme.font_size
    height = min(r.height, self._settings_panel.calc_preferred_size(layout_context, gui.Widget.Constraints()).height+30)
    self._settings_panel.frame = gui.Rect(r.get_right() - width, r.y, width, height)

  def __init__(self, width, height, dataset_info):
    # ---- basic setting ----
    self.divided_pcd = None
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
    self._collapsable_vert_1.set_is_open(False)
    self._world_axes_checkbox = gui.Checkbox("Show world axes")
    self._world_axes_checkbox.set_on_checked(self._on_world_axes_checkbox)
    self._obj_axes_checkbox = gui.Checkbox("Show obj axes")
    self._obj_axes_checkbox.set_on_checked(self._on_obj_axes_checkbox)

    self._collapsable_vert_2 = gui.CollapsableVert("Object choose panel", 0.33 * em, gui.Margins(em, 0, 0, 0))
    self._collapsable_vert_2.set_is_open(True)
    self._mesh_list = gui.ListView()
    self._mesh_list.set_items(['obj_' + f'{i:06}' for i in self.dataset_info.obj_ids])
    self._mesh_list.set_max_visible_items(3)
    self._choose_mesh_button = gui.Button("Choose Mesh")
    self._choose_mesh_button.set_on_clicked(self._on_choose_mesh)

    self._collapsable_vert_3 = gui.CollapsableVert("Symmetry Info panel", 0.33 * em, gui.Margins(em, 0, 0, 0))
    self._collapsable_vert_3.set_is_open(True)
    self._label_4 = gui.Label("symmetry information from bop:")
    self._object_info_label = gui.Label("")

    self._collapsable_vert_4 = gui.CollapsableVert("Edit Symmetry Panel", 0.33 * em, gui.Margins(em, 0, 0, 0))
    self._listview_sym_available = gui.ListView()
    self._listview_sym_available.set_items(['no sym', 'continuous sym'] + [f'{i}'+'_fold' for i in [2,3,4,5,6,8,16]] + ['discrete sym'])
    self._listview_sym_available.set_max_visible_items(3)
    self._button_choose_sym_type = gui.Button("Choose Sym type")
    self._button_choose_sym_type.set_on_clicked(self._on_button_choose_sym_type)
    self._listview_sym_chosed = gui.ListView()
    self._listview_sym_chosed.set_on_selection_changed(self._on_listview_sym_chosen)
    self._listview_sym_available.set_max_visible_items(3)
    self._horiz_2 = gui.Horiz()
    self._button_remove_sym_type = gui.Button("Remove Sym type")
    self._button_remove_sym_type.set_on_clicked(self._on_button_remove_sym_type)
    self._horiz_threshold = gui.Horiz()
    self._label_threshold = gui.Label("threshold:")
    self._number_edit_threshold = gui.NumberEdit(gui.NumberEdit.DOUBLE)
    self._number_edit_threshold.set_on_value_changed(self._on_number_edit_threshold)
    self._confirm_symmetry_button = gui.Button("Divide Point by Threshold")
    self._confirm_symmetry_button.set_on_clicked(self._on_confirm_symmetry)
    
    self._label_x_range = gui.Label("X range ")
    self._slider_x_min = gui.Slider(gui.Slider.DOUBLE)
    self._slider_x_min.set_on_value_changed(self._on_slider_x_min)
    self._slider_x_max = gui.Slider(gui.Slider.DOUBLE)
    self._slider_x_max.set_on_value_changed(self._on_slider_x_max)
    self._label_y_range = gui.Label("Y range ")
    self._slider_y_min = gui.Slider(gui.Slider.DOUBLE)
    self._slider_y_min.set_on_value_changed(self._on_slider_y_min)
    self._slider_y_max = gui.Slider(gui.Slider.DOUBLE)
    self._slider_y_max.set_on_value_changed(self._on_slider_y_max)
    self._label_z_range = gui.Label("Z range ")
    self._slider_z_min = gui.Slider(gui.Slider.DOUBLE)
    self._slider_z_min.set_on_value_changed(self._on_slider_z_min)
    self._slider_z_max = gui.Slider(gui.Slider.DOUBLE)
    self._slider_z_max.set_on_value_changed(self._on_slider_z_max)

    self._horiz_1 = gui.Horiz()
    self._button_add_green_yellow = gui.Button("Add Green-Yellow")
    self._button_add_green_yellow.set_on_clicked(self._on_button_add_green_yellow)  # TODO
    self._button_rest_points = gui.Button("Add Rest Points")
    self._button_rest_points.set_on_clicked(self._on_button_add_rest_points)  # TODO
    self._horiz_3 = gui.Horiz()
    self._button_remove_yellow = gui.Button("Remove Yellow")
    self._button_remove_yellow.set_on_clicked(self._on_button_remove_yellow)
    self._button_remove_cyan = gui.Button("Remove Cyan")
    self._button_remove_cyan.set_on_clicked(self._on_button_remove_cyan)

    self._collapsable_vert_5 = gui.CollapsableVert("Save/Load Panel", 0.33 * em, gui.Margins(em, 0, 0, 0))
    self._collapsable_vert_5.set_is_open(True)
    self._horiz_4 = gui.Horiz()
    self._button_save_result = gui.Button("Save Result")
    self._button_save_result.set_on_clicked(self._on_buttion_save_result)
    self._button_load_result = gui.Button("Load Result")
    self._button_load_result.set_on_clicked(self._on_buttion_load_result)

    self._collapsable_vert_6 = gui.CollapsableVert("Generate n System Code", 0.33 * em, gui.Margins(em, 0, 0, 0))
    self._collapsable_vert_6.set_is_open(True)
    self._horiz_5 = gui.Horiz()
    self._button_pair_pcd = gui.Button("Pair")
    self._button_pair_pcd.set_on_clicked(self._on_buttion_pair_pcd)
    self._button_vis_pcd = gui.Button("Vis Pair")
    self._button_vis_pcd.set_on_clicked(self._on_buttion_vis_pcd)
    self._button_divide_iter = gui.Button("Divide Iter")
    self._button_divide_iter.set_on_clicked(self._on_button_divide_iter)
    self._horiz_6 = gui.Horiz()
    self._button_vis_divide = gui.Button("Vis Divide")
    self._button_vis_divide.set_on_clicked(self._on_button_vis_divide)

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
    self._settings_panel.add_child(self._collapsable_vert_3)
    self._collapsable_vert_3.add_child(self._label_4)
    self._collapsable_vert_3.add_fixed(0.5 * em)
    self._collapsable_vert_3.add_child(self._object_info_label)
    self._collapsable_vert_3.add_fixed(1.5 * em)

    self._settings_panel.add_child(self._collapsable_vert_4)
    self._collapsable_vert_4.add_child(self._listview_sym_available)
    self._collapsable_vert_4.add_child(self._horiz_2)
    self._horiz_2.add_child(self._button_choose_sym_type)
    self._horiz_2.add_child(self._button_remove_sym_type)
    self._collapsable_vert_4.add_child(self._listview_sym_chosed)
    self._collapsable_vert_4.add_child(self._horiz_threshold)
    self._horiz_threshold.add_child(self._label_threshold)
    self._horiz_threshold.add_child(self._number_edit_threshold)
    self._collapsable_vert_4.add_child(self._confirm_symmetry_button)
    self._collapsable_vert_4.add_child(self._label_x_range)
    self._collapsable_vert_4.add_child(self._slider_x_min)
    self._collapsable_vert_4.add_child(self._slider_x_max)
    self._collapsable_vert_4.add_child(self._label_y_range)
    self._collapsable_vert_4.add_child(self._slider_y_min)
    self._collapsable_vert_4.add_child(self._slider_y_max)
    self._collapsable_vert_4.add_child(self._label_z_range)
    self._collapsable_vert_4.add_child(self._slider_z_min)
    self._collapsable_vert_4.add_child(self._slider_z_max)
    self._collapsable_vert_4.add_child(self._horiz_1)
    self._horiz_1.add_child(self._button_add_green_yellow)
    self._horiz_1.add_child(self._button_rest_points)
    self._collapsable_vert_4.add_child(self._horiz_3)
    self._horiz_3.add_child(self._button_remove_yellow)
    self._horiz_3.add_child(self._button_remove_cyan)

    self._settings_panel.add_child(self._collapsable_vert_5)
    self._collapsable_vert_5.add_child(self._horiz_4)
    self._horiz_4.add_child(self._button_save_result)
    self._horiz_4.add_child(self._button_load_result)

    self._settings_panel.add_child(self._collapsable_vert_6)
    self._collapsable_vert_6.add_child(self._horiz_5)
    self._horiz_5.add_child(self._button_pair_pcd)
    self._horiz_5.add_child(self._button_vis_pcd)
    self._horiz_5.add_child(self._button_divide_iter)
    self._collapsable_vert_6.add_child(self._horiz_6)
    self._horiz_6.add_child(self._button_vis_divide)

    # ---- Menu ----

    # ---- tool init setting ----
    self._init_settings()

  def _init_settings(self):
    # ----
    self.divided_pcd = DividedPcd()
    # ---- init background ----
    bg_color = [self.settings.bg_color.red, self.settings.bg_color.green,
                self.settings.bg_color.blue, self.settings.bg_color.alpha]
    self._scene.scene.set_background(bg_color)
    # ---- init axes checkbox state ----
    self._world_axes_checkbox.checked = self.settings.show_world_axes_state
    self._obj_axes_checkbox.checked = self.settings.show_obj_axes_state

  def _on_world_axes_checkbox(self, show):
    self.settings.show_world_axes_state = show
    self._scene.scene.show_axes(self.settings.show_world_axes_state)
    self._world_axes_checkbox.checked = self.settings.show_world_axes_state

  def _on_obj_axes_checkbox(self, show):
    self.settings.show_obj_axes_state = show
    if self.settings.show_obj_axes_state:
      points = np.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]) * 125
      lines = [[0, 1], [0, 2], [0, 3]]
      colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
      line_set = o3d.geometry.LineSet(points=o3d.utility.Vector3dVector(points),
                                      lines=o3d.utility.Vector2iVector(lines), )
      line_set.colors = o3d.utility.Vector3dVector(colors)
      self._scene.scene.add_geometry("obj_axis", line_set, self.settings.axis_material)
    else:
      if self._scene.scene.has_geometry('obj_axis'):
        self._scene.scene.remove_geometry('obj_axis')
    self._obj_axes_checkbox.checked = self.settings.show_obj_axes_state

  def _on_button_choose_sym_type(self):
    self.divided_pcd.add_sym_type(self._listview_sym_available.selected_value)
    sym_type = self.divided_pcd.get_sym_type()
    self._listview_sym_chosed.set_items(sym_type)
    self._listview_sym_chosed.selected_index = len(sym_type) - 1

  def _on_button_remove_sym_type(self):
    active_sym_type = self._listview_sym_chosed.selected_value
    self.divided_pcd.remove_sym_type(active_sym_type)  # TODO
    # update list after adding removing object
    sym_type = self.divided_pcd.get_sym_type()
    self._listview_sym_chosed.set_items(sym_type)

  def _on_listview_sym_chosen(self, active_sym_type, _):
    if self._scene.scene.has_geometry('active_pcd'):
      self._scene.scene.remove_geometry('active_pcd')
    exist_index = self.divided_pcd.sym_type_index.get(active_sym_type)
    if exist_index is not None:
      active_pcd = self.divided_pcd.origin_pcd.select_by_index(self.divided_pcd.sym_type_index[active_sym_type])
      active_pcd.translate((0, 2.4 * self.divided_pcd.model_info['size_y'], 0))
      active_pcd.paint_uniform_color([0.0, 0.1, 0.6])
      self._scene.scene.add_geometry('active_pcd', active_pcd, self.settings.obj_material)

  def _on_button_add_green_yellow(self):
    """
    add intersection green part(disym part) and yellow part(clip part) in visualization to selected Sym type

    1. get active sym type from ListView
    1. get the points of disym part (not self.divided_pcd.sym_index)
    2. get the points of clip part
    3. get the intersection
    4. save to self.divided_pcd.sym_type_index[active_sym_type]
    """
    active_sym_type = self._listview_sym_chosed.selected_value
    all_index = np.arange(len(self.divided_pcd.origin_pcd.points))
    sym_index = self.divided_pcd.sym_index
    disym_index = np.setdiff1d(all_index, sym_index)
    clip_index = self.divided_pcd.clip_index
    intersection_index = np.intersect1d(disym_index, clip_index, assume_unique=True)
    exist_index = self.divided_pcd.sym_type_index.get(active_sym_type)
    if exist_index is None:
      self.divided_pcd.sym_type_index[active_sym_type] = intersection_index
    else:
      union_index = np.union1d(exist_index, intersection_index)
      self.divided_pcd.sym_type_index[active_sym_type] = union_index

    active_pcd = self.divided_pcd.origin_pcd.select_by_index(self.divided_pcd.sym_type_index[active_sym_type])
    active_pcd.translate((0, 2.4 * self.divided_pcd.model_info['size_y'], 0))
    active_pcd.paint_uniform_color([0.0, 0.1, 0.6])
    if self._scene.scene.has_geometry('active_pcd'):
      self._scene.scene.remove_geometry('active_pcd')
    self._scene.scene.add_geometry('active_pcd', active_pcd, self.settings.obj_material)

  def _on_button_add_rest_points(self):
    """
    add rest points to active_sym_type

    1. get active sym type from ListView
    2. get all the points
    3. get points that not in any sym_type
    4. save to self.divided_pcd.sym_type_index[active_sym_type]
    """
    if self._scene.scene.has_geometry('active_pcd'):
      self._scene.scene.remove_geometry('active_pcd')
    active_sym_type = self._listview_sym_chosed.selected_value
    rest_index = np.arange(len(self.divided_pcd.origin_pcd.points))

    for sym_type in self.divided_pcd.sym_type_index:
      if not sym_type == active_sym_type:
        sym_type_index = self.divided_pcd.sym_type_index.get(sym_type)
        if sym_type_index is not None:
          rest_index = np.setdiff1d(rest_index, sym_type_index)
    self.divided_pcd.sym_type_index[active_sym_type] = rest_index
    active_pcd = self.divided_pcd.origin_pcd.select_by_index(self.divided_pcd.sym_type_index[active_sym_type])
    active_pcd.translate((0, 2.4 * self.divided_pcd.model_info['size_y'], 0))
    active_pcd.paint_uniform_color([0.0, 0.1, 0.6])
    self._scene.scene.add_geometry('active_pcd', active_pcd, self.settings.obj_material)

  def _on_button_remove_yellow(self):
    """
    remove intersection of active_sym_type and yellow part(clip part) in visualization to selected Sym type

    1. get active sym type from ListView
    2. get the points of clip part
    3. get the intersection
    4. save to self.divided_pcd.sym_type_index[active_sym_type]
    """
    if self._scene.scene.has_geometry('active_pcd'):
      self._scene.scene.remove_geometry('active_pcd')
    active_sym_type = self._listview_sym_chosed.selected_value
    exist_index = self.divided_pcd.sym_type_index.get(active_sym_type)
    if exist_index is not None:
      clip_index = self.divided_pcd.clip_index
      intersection_index = np.intersect1d(exist_index, clip_index, assume_unique=True)
      exist_index = np.setdiff1d(exist_index, intersection_index)
      self.divided_pcd.sym_type_index[active_sym_type] = exist_index
      active_pcd = self.divided_pcd.origin_pcd.select_by_index(self.divided_pcd.sym_type_index[active_sym_type])
      active_pcd.translate((0, 2.4 * self.divided_pcd.model_info['size_y'], 0))
      active_pcd.paint_uniform_color([0.0, 0.1, 0.6])
      self._scene.scene.add_geometry('active_pcd', active_pcd, self.settings.obj_material)
    else:
      print(f'can not find {active_sym_type}')
      return

  def _on_button_remove_cyan(self):
    """
    remove intersection of active_sym_type and cyan part(non-clip part) in visualization to selected Sym type

    1. get active sym type from ListView
    2. get the points of clip part
    3. get the intersection
    4. save to self.divided_pcd.sym_type_index[active_sym_type]
    """
    active_sym_type = self._listview_sym_chosed.selected_value
    exist_index = self.divided_pcd.sym_type_index.get(active_sym_type)
    if self._scene.scene.has_geometry('active_pcd'):
      self._scene.scene.remove_geometry('active_pcd')
    if exist_index is not None:
      clip_index = self.divided_pcd.clip_index
      intersection_index = np.intersect1d(exist_index, clip_index, assume_unique=True)
      self.divided_pcd.sym_type_index[active_sym_type] = intersection_index
      active_pcd = self.divided_pcd.origin_pcd.select_by_index(self.divided_pcd.sym_type_index[active_sym_type])
      active_pcd.translate((0, 2.4 * self.divided_pcd.model_info['size_y'], 0))
      active_pcd.paint_uniform_color([0.0, 0.1, 0.6])
      self._scene.scene.add_geometry('active_pcd', active_pcd, self.settings.obj_material)
    else:
      print(f'can not find defined {active_sym_type} points')

  def _on_buttion_save_result(self):
    result = dict()
    result['dataset_name'] = self.dataset_info.dataset_name
    result['obj_id'] = self._mesh_list.selected_index + 1
    result['num_points'] = len(self.divided_pcd.origin_pcd.points)
    result['sym_type'] = self.divided_pcd.get_sym_type()
    result['sym_type_index'] = dict()
    result['pairs'] = self.divided_pcd.pairs
    num_points = 0
    for k, v in self.divided_pcd.sym_type_index.items():
      result['sym_type_index'][k] = v.tolist()
      num_points = num_points + len(v)
    assert num_points == result['num_points']
    json_path = self.dataset_info.obj_tpath.replace('.obj', '.json').format(obj_id=result['obj_id'])
    if os.path.exists(json_path):
      self.window.show_message_box('Error Saving', f'json_path exists, remove {json_path} before save')
      print(f'json_path exists, remove {json_path} before save')
      return
    with open(json_path, 'w') as f:
      f.write(json.dumps(result))

  def _on_buttion_load_result(self):
    """
    load result which saved by the button Save_Result

    1. choose the mesh in _mesh_list and load result
    2. mimic the operation of choose mesh and load
    3. assert information
    4. save the sym type information to self.divided_pcd
    """

    obj_id = self._mesh_list.selected_index + 1
    if obj_id == 0:
      print("should choose Mesh in Object choose panel")
      return
    json_path = self.dataset_info.obj_tpath.replace('.obj', '.json').format(obj_id=obj_id)
    if os.path.exists(json_path):
      with open(json_path, 'r') as f:
        result = json.loads(f.read())
    # ---- mimic the operation of choose mesh and load ----
    self._mesh_list.selected_index = obj_id - 1
    self._on_choose_mesh()
    # ---- assert information ----
    assert result['dataset_name'] == self.dataset_info.dataset_name
    assert result['num_points'] == len(self.divided_pcd.origin_pcd.points)
    # ---- save the sym type information to self.divided_pcd ----
    self.divided_pcd.sym_type = result['sym_type']
    self.divided_pcd.pairs = result['pairs'] if 'pairs' in result.keys() else dict()
    self._listview_sym_chosed.set_items(self.divided_pcd.sym_type)
    num_points = 0
    for k, v in result['sym_type_index'].items():
      self.divided_pcd.sym_type_index[k] = np.array(v)
      num_points = num_points + len(v)
    assert result['num_points'] == num_points

  def _on_number_edit_threshold(self, num):
    self.divided_pcd.set_threshold(num)

  def _on_slider_x_min(self, num):
    self.divided_pcd.clip_x_min = num
    self._visualize_clip_box()

  def _on_slider_x_max(self, num):
    self.divided_pcd.clip_x_max = num
    self._visualize_clip_box()

  def _on_slider_y_min(self, num):
    self.divided_pcd.clip_y_min = num
    self._visualize_clip_box()

  def _on_slider_y_max(self, num):
    self.divided_pcd.clip_y_max = num
    self._visualize_clip_box()

  def _on_slider_z_min(self, num):
    self.divided_pcd.clip_z_min = num
    self._visualize_clip_box()

  def _on_slider_z_max(self, num):
    self.divided_pcd.clip_z_max = num
    self._visualize_clip_box()

  def _visualize_clip_box(self):
    x_min = self.divided_pcd.clip_x_min
    x_max = self.divided_pcd.clip_x_max
    y_min = self.divided_pcd.clip_y_min
    y_max = self.divided_pcd.clip_y_max
    z_min = self.divided_pcd.clip_z_min
    z_max = self.divided_pcd.clip_z_max
    if x_min < x_max and y_min < y_max and z_min < z_max:
      if self._scene.scene.has_geometry('clip_box'):
        self._scene.scene.remove_geometry('clip_box')
      clip_box_points = np.asarray([[x_max, y_max, z_max],
                                    [x_max, y_max, z_min],
                                    [x_max, y_min, z_max],
                                    [x_max, y_min, z_min],
                                    [x_min, y_max, z_max],
                                    [x_min, y_max, z_min],
                                    [x_min, y_min, z_max],
                                    [x_min, y_min, z_min]])
      clip_box_lines = [[0, 1], [0, 2], [0, 4], [1, 3], [1, 5], [2, 3],
                        [2, 6], [3, 7], [4, 5], [4, 6], [5, 7], [6, 7]]
      clip_box_colors = [[0, 0, 0] for line in clip_box_lines]
      line_set = o3d.geometry.LineSet(points=o3d.utility.Vector3dVector(clip_box_points),
                                      lines=o3d.utility.Vector2iVector(clip_box_lines), )
      line_set.translate((0, 1.2 * self.divided_pcd.model_info['size_y'], 0))
      line_set.colors = o3d.utility.Vector3dVector(clip_box_colors)
      self._scene.scene.add_geometry('clip_box', line_set, self.settings.clip_box_material)
      pcd = np.asarray(self.divided_pcd.origin_pcd.points)
      clip_pcd_index = list()
      for i, p in enumerate(pcd):
        if p[0]>= x_min and p[0]<=x_max and p[1]>=y_min and p[1]<=y_max and p[2]>=z_min and p[2]<=z_max:
          clip_pcd_index.append(i)
      self.divided_pcd.clip_index = np.array(clip_pcd_index)
      clip_pcd = self.divided_pcd.origin_pcd.select_by_index(self.divided_pcd.clip_index)
      clip_pcd.translate((0, 1.2 * self.divided_pcd.model_info['size_y'], 0))
      clip_pcd.paint_uniform_color([0.5, 0.5, 0.0])
      clip_pcd_inverse = self.divided_pcd.origin_pcd.select_by_index(self.divided_pcd.clip_index, invert=True)
      clip_pcd_inverse.translate((0, 1.2 * self.divided_pcd.model_info['size_y'], 0))
      clip_pcd_inverse.paint_uniform_color([0.0, 0.5, 0.5])
      if self._scene.scene.has_geometry('clip_pcd'):
        self._scene.scene.remove_geometry('clip_pcd')
      self._scene.scene.add_geometry('clip_pcd', clip_pcd, self.settings.obj_material)
      if self._scene.scene.has_geometry('clip_pcd_inverse'):
        self._scene.scene.remove_geometry('clip_pcd_inverse')
      self._scene.scene.add_geometry('clip_pcd_inverse', clip_pcd_inverse, self.settings.obj_material)
    else:
      if self._scene.scene.has_geometry('clip_box'):
        self._scene.scene.remove_geometry('clip_box')
      if self._scene.scene.has_geometry('clip_pcd'):
        self._scene.scene.remove_geometry('clip_pcd')
      if self._scene.scene.has_geometry('clip_pcd_inverse'):
        self._scene.scene.remove_geometry('clip_pcd_inverse')

  def _on_choose_mesh(self):
    '''
    1. clear scene and visualize chose point cloud file (romoved outliners)
    2. get symmetry information from bop, store in model_info
    3. print information
    '''
    self._init_settings()
    # ---- visualize the object ----
    selected_obj_id = self._mesh_list.selected_index + 1
    if selected_obj_id == 0:
      print("should choose obj in Object Choose Panel")
      return
    mesh = o3d.io.read_triangle_mesh(self.dataset_info.obj_tpath.format(obj_id=selected_obj_id))
    pcd = o3d.io.read_point_cloud("None")
    pcd.points = o3d.utility.Vector3dVector(np.asarray(mesh.vertices))

    self._scene.scene.clear_geometry()
    self._scene.scene.add_geometry("origin_pcd", pcd, self.settings.obj_material)
    model_info = self.dataset_info.models_info[selected_obj_id]

    self.divided_pcd.set_pcd(pcd)
    self.divided_pcd.set_model_info(model_info)
    self._label_4.text = f"x = [{model_info['min_x']}, {model_info['min_x']+model_info['size_x']}]" \
                         f"y = [{model_info['min_y']}, {model_info['min_y'] + model_info['size_y']}]" \
                         f"z = [{model_info['min_z']}, {model_info['min_z'] + model_info['size_z']}]"

    self._number_edit_threshold.set_value(self.divided_pcd.threshold)
    self._slider_x_min.set_limits(model_info['min_x']*1.01, (model_info['min_x']+model_info['size_x'])*1.01)
    self._slider_x_min.double_value = self._slider_x_min.get_minimum_value
    self._slider_x_max.set_limits(model_info['min_x']*1.01, (model_info['min_x']+model_info['size_x'])*1.01)
    self._slider_x_max.double_value = self._slider_x_max.get_maximum_value
    self._slider_y_min.set_limits(model_info['min_y']*1.01, (model_info['min_y'] + model_info['size_y'])*1.01)
    self._slider_y_min.double_value = self._slider_y_min.get_minimum_value
    self._slider_y_max.set_limits(model_info['min_y']*1.01, (model_info['min_y'] + model_info['size_y'])*1.01)
    self._slider_y_max.double_value = self._slider_y_max.get_maximum_value
    self._slider_z_min.set_limits(model_info['min_z']*1.01, (model_info['min_z'] + model_info['size_z'])*1.01)
    self._slider_z_min.double_value = self._slider_z_min.get_minimum_value
    self._slider_z_max.set_limits(model_info['min_z']*1.01, (model_info['min_z'] + model_info['size_z'])*1.01)
    self._slider_z_max.double_value = self._slider_z_max.get_maximum_value

    if 'symmetries_discrete' in model_info and 'symmetries_continuous' in model_info:
      self._object_info_label.text = 'the object is both discrete and ' \
                                     'continuous symmetries, cannot handle it'
    elif 'symmetries_continuous' in model_info:
      self._object_info_label.text = f'the object is continuous symmetries. ' \
                                     f'{model_info["symmetries_continuous"]}'
    elif 'symmetries_discrete' in model_info:
      num_discrete = len(model_info['symmetries_discrete'])
      self._object_info_label.text = f'object information: \nthe object has {num_discrete + 1} discrete symmetry poses'
    else:
      self._object_info_label.text = 'the object is not symmetry'

  def _on_confirm_symmetry(self):
    active_sym_type = self._listview_sym_chosed.selected_value
    if active_sym_type == '':
      print('Warning: should choose active_sym_type in listview')
      return
    elif active_sym_type == 'continuous sym' or active_sym_type == 'no sym':
      all_dists = self.divided_pcd.divide_conti_sym()
      self.divided_pcd.sym_index = np.where(all_dists < self.divided_pcd.threshold)[0]
      print(f'the threshold is: {self.divided_pcd.threshold} now')
      print('make red part (sym part) bigger, edit threshold bigger')
      print('make green part (disym part) bigger, edit threshold smaller')
      sym_pcd = self.divided_pcd.origin_pcd.select_by_index(self.divided_pcd.sym_index)
      disym_pcd = self.divided_pcd.origin_pcd.select_by_index(self.divided_pcd.sym_index, invert=True)
    elif active_sym_type.endswith("_fold"):
      n = int(active_sym_type.split('_')[0])
      all_dists = self.divided_pcd.divide_n_fold_sym(n)
      self.divided_pcd.sym_index = np.where(all_dists < self.divided_pcd.threshold)[0]
      print(f'the threshold is: {self.divided_pcd.threshold} now')
      print('make red part (sym part) bigger, edit threshold bigger')
      print('make green part (disym part) bigger, edit threshold smaller')
      sym_pcd = self.divided_pcd.origin_pcd.select_by_index(self.divided_pcd.sym_index)
      disym_pcd = self.divided_pcd.origin_pcd.select_by_index(self.divided_pcd.sym_index, invert=True)
    elif active_sym_type == 'discrete sym':
      all_dists = self.divided_pcd.divide_discrete_sym()
      self.divided_pcd.sym_index = np.where(all_dists < self.divided_pcd.threshold)[0]
      print(f'the threshold is: {self.divided_pcd.threshold} now')
      print('make red part (sym part) bigger, edit threshold bigger')
      print('make green part (disym part) bigger, edit threshold smaller')
      sym_pcd = self.divided_pcd.origin_pcd.select_by_index(self.divided_pcd.sym_index)
      disym_pcd = self.divided_pcd.origin_pcd.select_by_index(self.divided_pcd.sym_index, invert=True)

    sym_pcd.paint_uniform_color([0.5, 0.0, 0.0])
    disym_pcd.paint_uniform_color([0.0, 0.5, 0.0])
    if self._scene.scene.has_geometry('origin_pcd'):
      self._scene.scene.remove_geometry('origin_pcd')
    if self._scene.scene.has_geometry('sym_pcd'):
      self._scene.scene.remove_geometry('sym_pcd')
    self._scene.scene.add_geometry('sym_pcd', sym_pcd, self.settings.obj_material)
    if self._scene.scene.has_geometry('disym_pcd'):
      self._scene.scene.remove_geometry('disym_pcd')
    self._scene.scene.add_geometry('disym_pcd', disym_pcd, self.settings.obj_material)

  def _clear_scene_for_clip(self):
    if self._scene.scene.has_geometry('clip_box'):
      self._scene.scene.remove_geometry('clip_box')
    if self._scene.scene.has_geometry('clip_pcd'):
      self._scene.scene.remove_geometry('clip_pcd')
    if self._scene.scene.has_geometry('clip_pcd_inverse'):
      self._scene.scene.remove_geometry('clip_pcd_inverse')

  def _clear_scene_for_pair(self):
    if self._scene.scene.has_geometry('painted_without_sym_pcd'):
      self._scene.scene.remove_geometry('painted_without_sym_pcd')
    if self._scene.scene.has_geometry('painted_with_sym_pcd'):
      self._scene.scene.remove_geometry('painted_with_sym_pcd')

  def _on_buttion_pair_pcd(self):
    """ pair points in self.divided_pcd.origin_pcd based on self.divided_pcd.sym_type_index,
        the result pair points will be saved as dict in self.divided_pcd.pairs

        e.g. pairs = dict( 'no sym': list(list())  # a list of paired points
                           'continuous sym': list(list())  # a list of paired points
                           '2_fold': list(list())  # a list of paired points
                         )
    """
    # clear unrelated scene for better visualize
    self._clear_scene_for_clip()
    for sym_type in self.divided_pcd.sym_type:
      if sym_type.endswith("_fold"):
        n = int(sym_type.split('_')[0])
        print(f"start pairing {n}-fold sym pcd")
        self.divided_pcd.pair_n_fold_sym_pcd(n)
        print(f"finish pairing {n}-fold sym pcd")
      elif sym_type in ['discrete sym']:
        print("start pairing discrete sym pcd")
        self.divided_pcd.pair_discrete_sym_pcd()
        print("finish pairing discrete sym pcd")
      elif sym_type in ['continuous sym']:
        print("start pairing continuous sym pcd, will cost about 30 seconds")
        print("note : if the final pair is not fine enough, you should set smaller val_threshold in function "
              "pair_continuous_sym_pcd(val_threshold), default val_threshold = 0.00001*obj_diameter")
        self.divided_pcd.pair_continuous_sym_pcd()
        print("finish pairing continuous sym pcd")
      elif sym_type in ['no sym']:
        print("start pairing no sym pcd")
        self.divided_pcd.pair_no_sym_pcd()
        print("finish pairing no sym pcd")
    self._on_buttion_vis_pcd()

  def _on_buttion_vis_pcd(self):
    """ vis points in dissym color and sym color """
    # clear unrelated scene for better visualize
    self._clear_scene_for_clip()
    colors = self.divided_pcd.origin_pcd.points
    colors = (colors-np.min(colors, 0)) / (np.max(colors, 0) - np.min(colors, 0))
    painted_without_sym_pcd = copy.deepcopy(self.divided_pcd.origin_pcd)
    painted_without_sym_pcd.translate((0, -1.2 * self.divided_pcd.model_info['size_y'], 0))
    painted_without_sym_pcd.colors = o3d.utility.Vector3dVector(colors)
    if self._scene.scene.has_geometry('painted_without_sym_pcd'):
      self._scene.scene.remove_geometry('painted_without_sym_pcd')
    self._scene.scene.add_geometry('painted_without_sym_pcd', painted_without_sym_pcd,  self.settings.obj_material)
    sym_colors = np.zeros([len(self.divided_pcd.origin_pcd.points), 3])
    for sym_type, pairs in self.divided_pcd.pairs.items():
      for pair in pairs:
        coordinate = np.empty((len(pair), 3))
        for i, index in enumerate(pair):
          coordinate[i] = self.divided_pcd.origin_pcd.points[index]
        if sym_type == 'continuous sym':
          chosen_index = np.random.randint(len(pair))
        else:
          chosen_index = np.argmin(np.sum(coordinate,axis=1))
        sym_colors[pair] = coordinate[chosen_index]
    sym_colors = (sym_colors-np.min(sym_colors, 0)) / (np.max(sym_colors, 0) - np.min(sym_colors, 0))
    painted_with_sym_pcd = copy.deepcopy(self.divided_pcd.origin_pcd)
    painted_with_sym_pcd.translate((0, -2.4 * self.divided_pcd.model_info['size_y'], 0))
    painted_with_sym_pcd.colors = o3d.utility.Vector3dVector(sym_colors)
    if self._scene.scene.has_geometry('painted_with_sym_pcd'):
      self._scene.scene.remove_geometry('painted_with_sym_pcd')
    self._scene.scene.add_geometry('painted_with_sym_pcd', painted_with_sym_pcd,  self.settings.obj_material)

  def _on_button_divide_iter(self):
    """ divide point in self.divided_pcd.pairs """
    # ---- clear unrelated scene for better visualize -----
    self._clear_scene_for_clip()
    self._clear_scene_for_pair()
    # ---- visualize the one point from one pair result without coordinate shift ----
    # if coordinate_shift = False, then the coordinate will not change for better visualization
    select_indexes, select_coordinates = self.divided_pcd.choose_one_point_each_pair(coordinate_shift=False)
    one_point_each_pair_pcd_non_shift = o3d.io.read_point_cloud("None")
    one_point_each_pair_pcd_non_shift.points = o3d.utility.Vector3dVector(select_coordinates)
    colors = one_point_each_pair_pcd_non_shift.points
    colors = (colors-np.min(colors, 0)) / (np.max(colors, 0) - np.min(colors, 0))
    one_point_each_pair_pcd_non_shift.translate((0, -1.2 * self.divided_pcd.model_info['size_y'], 0))
    one_point_each_pair_pcd_non_shift.colors = o3d.utility.Vector3dVector(colors)
    self._scene.scene.add_geometry('one_point_each_pair_pcd_non_shift', one_point_each_pair_pcd_non_shift, self.settings.obj_material)
    # ---- visualize the the one point from one pair result with coordinate shift ----
    # if coordinate_shift = True, the coordinate will multiply in sym axis for better cluster later
    select_indexes, select_coordinates = self.divided_pcd.choose_one_point_each_pair(coordinate_shift=True)
    one_point_each_pair_pcd_shift = o3d.io.read_point_cloud("None")
    one_point_each_pair_pcd_shift.points = o3d.utility.Vector3dVector(select_coordinates)
    colors = one_point_each_pair_pcd_shift.points
    colors = (colors-np.min(colors, 0)) / (np.max(colors, 0) - np.min(colors, 0))
    one_point_each_pair_pcd_shift.translate((0, -2.4 * self.divided_pcd.model_info['size_y'], 0))
    one_point_each_pair_pcd_shift.colors = o3d.utility.Vector3dVector(colors)
    self._scene.scene.add_geometry('one_point_each_pair_pcd_shift', one_point_each_pair_pcd_shift, self.settings.obj_material)
    # divide select_coordinates iteratively
    print("you can change number of iteration and divide number in DividedPcd")
    classified_indice_local = self.divided_pcd.divide_pointcloud_iterative(select_coordinates, self.divided_pcd.number_of_iteration, self.divided_pcd.divide_number)
    local_to_origin_index_map = dict(zip(range(len(select_coordinates)), select_indexes))
    classified_indice_global = []
    for l_local in classified_indice_local:
      l_global = [local_to_origin_index_map[l] for l in l_local]
      classified_indice_global.append(l_global)
    self.divided_pcd.classified_indice = self.divided_pcd.indice_completion(classified_indice_global)
  def _on_button_vis_divide(self):
    # self._scene.scene.clear_geometry()
    number_of_iteration = self.divided_pcd.number_of_iteration
    divide_number = self.divided_pcd.divide_number
    visual_pcd = []

    for i in range(number_of_iteration):
      visual_pcd.append(copy.deepcopy(self.divided_pcd.origin_pcd))
      colors = np.empty([len(self.divided_pcd.origin_pcd.points), 3])
      for j, l in enumerate(self.divided_pcd.classified_indice):
        bit = j // pow(divide_number, number_of_iteration - 1 - i) % divide_number
        colors[l] = bit * np.array([1.0, 1.0, 1.0]) / divide_number
      visual_pcd[i].colors = o3d.utility.Vector3dVector(colors)
      visual_pcd[i].translate((0, 1.2 * self.divided_pcd.model_info['size_y'] * (i+1), 0))
      self._scene.scene.add_geometry('visual_pcd_'+str(i), visual_pcd[i], self.settings.obj_material)
    print(1)


def main():
  """ Init GUI and run with pre-defined parameters """
  gui.Application.instance.initialize()
  w = AppWindow(2048, 1536, DatasetInfo(p))
  gui.Application.instance.run()

if __name__ == "__main__":
  main()
















































