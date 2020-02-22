import numpy as np




x = np.array([[1, 0, 3, 1], [1, 9, 9, 2], [1, 8, 8, 3],[1, 4, 4, 3]])

x = x[:, 1::4]
print(f'x = {x}')

