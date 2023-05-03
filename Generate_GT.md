# Generate Ground Truth Binary Code for Symnet from scratch

# Main Dependencies:
- [`bop_toolkit`](https://github.com/thodan/bop_toolkit)
- open3d
- opencv-python

## Step 1: remesh models
`cd scripts

 python surface_samples_remesh_visible.py --dataset tless
 `

## Step 2: generate mesh with binary code

### download [`bop_toolkit`](https://github.com/thodan/bop_toolkit) or make soft link

### download the dataset from [`BOP benchmark`](https://bop.felk.cvut.cz/datasets/)

The expected data structure:
```
    code_generator/
    ├── scripts
    ├── utils
    ├── bop_toolkit
    ├── data/
        ├── bop/
        │   ├── ycbv
        │   ├── tless
        │   └── ...               #(other datasets from BOP page)
        ├── remesh_visible
        ├── models_GT_color
        └── binary_code
```

### 


## different between main.py and main_v2.py
In main.py, the symmetry points will be viewed as one point, which leads to the biased points divided.
在脚本main.py中，对称的点被当成同一个点，导致不对称区域和对称区域点的数量和面积不匹配，导致对称区域一直不被聚类

