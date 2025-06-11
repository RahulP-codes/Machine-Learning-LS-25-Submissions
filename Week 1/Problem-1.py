import numpy as np

arr = np.random.randint(1, 50, size=(5,4))

# 1st Part
print(f"numpy array :\n {arr}\n")

anti_diag_elems = np.fliplr(arr).diagonal()

# 2nd Part
max_in_rows = arr.max(axis=1)
print(f"maximum value in each row of array : {max_in_rows}\n")

# 3rd Part
mean = arr.mean()
n_arr = arr[arr<=mean]

# 4th Part
def numpy_boundry_traversal (matrix: np.ndarray):
     top = matrix[0, :]
     right = matrix[1:, -1]
     bottom = matrix[-1, :-1]
     left = matrix[1:-1, 0]

     return np.concatenate((top, right, bottom[::-1], left[::-1])).tolist()

#END
print(f"anti-diagonal elements : {anti_diag_elems}")
print(f"elements less than overall mean of array: {n_arr}")
print(f"Travelling along boundry of matrix : {numpy_boundry_traversal(arr)}")