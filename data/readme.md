bop/             bop datasets
models_GT_color_v1/       origin labeled ply from zebrapose
models_GT_color_v2/       使用main_v1.py生成，使用更多的点，渲染不插值
models_GT_color_v2.2/     使用main_v1.py生成，没有更多的点，渲染时插值
models_GT_color_v3/       新版本的点划分，没有更多的点

binary_code_v1/           origin label from zebrapose
binary_code_v2/           使用models_GT_color_v2/生成的标签
binary_code_v2.2/           使用models_GT_color_v2.2/生成的标签

binary_code_v3/           使用models_GT_color_v3/和generate_labels_for_datasets.py生成的标签
