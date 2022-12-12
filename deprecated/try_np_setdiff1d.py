import numpy as np
a1 = np.asarray([[1,2,3],[3,4,5],[4,5,6]])
a2 = np.asarray([[1,2,3], [3,4,5]])

a1_rows = a1.view([('', a1.dtype)] * a1.shape[1])
a2_rows = a2.view([('', a2.dtype)] * a2.shape[1])

print(np.setdiff1d(a1_rows, a2_rows).view(a1.dtype).reshape(-1, a1.shape[1]))
