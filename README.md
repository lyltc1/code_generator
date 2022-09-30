# code_generator

## usage:

### step1: download bop_toolkit or make soft link

```ln -s path/to/bop_toolkit/ .```


## different between main.py and main_v2.py
In main.py, the symmetry points will be viewed as one point, which leads to the biased points divided.
在脚本main.py中，对称的点被当成同一个点，导致不对称区域和对称区域点的数量和面积不匹配，导致对称区域一直不被聚类

