# Generate Ground Truth Binary Code for Symnet from Scratch

## Build Conda Environment
```bash
conda create -n code python=3.9
conda activate code
pip install open3d opencv-python pymeshlab==v2022.2 trimesh imageio
# install bop_toolkit
git clone https://github.com/lyltc1/bop_toolkit.git
cd bop_toolkit
pip install -e .
# install code_generator
pip install -e . # need -e
```

## Prepare Data
Download the dataset from the [BOP benchmark](https://bop.felk.cvut.cz/datasets/).

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
    │   └── ...               # other datasets from BOP page
    ├── remesh_visible        # will be generated
    ├── models_GT_color       # will be generated
    └── binary_code           # will be generated
```

make a softlink for bop dataset, for example
```
ln -s [source bop path] [target bop path]  # i.e. ln -s /mnt/2T/Data/BOP/ ./data/bop
```

## Step 1: Remesh Models
``` bash
cd scripts
python surface_samples_remesh_visible.py --dataset tless
```

## Step 2: Generate Code
change the parameter in generate_meshes.py # PARAMETERS. the path and dataset_name
``` bash
cd scripts
python generate_meshes.py 
```

1. **Open the Annotation Tool**
    - You should see the annotation tool opened.
    - ![Annotation Tool](imgs/annotation_tool.png)

2. **Load an Object**
    - From the right-side Object choose panel, you can choose an object to load.
    - Click `Choose Mesh` to load the object.
    - The symmetry information will be displayed in the `Symmetry Info Panel`.

3. **Choose Symmetry Type**
    - In the `Edit Symmetry Panel`, you can choose different symmetry types.
    - For example, for `tless-object27`, it only contains `discrete sym`.
      - Choose `discrete sym` in the `Edit Symmetry Panel`.
      - Click `Add Rest Points` to annotate all points as discrete sym.
    - ![tless-object27](imgs/tless-obj27.png)

4. **Annotate Complex Objects**
    - For more complex objects, use `threshold` and `x/y/z range` to help locate points.

5. **Save and Load Results**
    - Click `Load Result` or `Save Result` to save or load the annotation symmetry type.

6. **Generate Code**
    - Click the following buttons to generate code:
    - `Pair`
        - `Vis Pair`
        - `Divide Iter`
        - `Vis Divide`
            - ![tless-object27](imgs/tless-obj27-visDivide.png)
        - `Vis Ply`
            - ![tless-object27](imgs/tless-obj27-visPly.png)
        - `Save Final`, which saves the following files:
            - `code_generator/data/models_GT_color_v3/tless/obj_000027.ply`
            - `code_generator/data/models_GT_color_v3/tless/Class_CorresPoint000027.json`
