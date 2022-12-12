import numpy as np

code = np.linspace(0, 1, 25).reshape((5, 5))

print(code)
print(1 + 4*code*(code-1))
