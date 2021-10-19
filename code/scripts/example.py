import numpy as np

array = [[[1,2,3],[1,2,3],[1,2,3]],
         [[1,2,3],[1,2,3],[1,2,3]],
         [[1,2,3],[1,2,3],[1,2,3]]]

print(np.shape(array))


array = np.divide(array,2)
print(array)
