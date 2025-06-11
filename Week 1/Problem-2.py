import numpy as np

arr = np.random.random(20) * 10

# 1st Part
arr = np.round(arr, 2)

print(f"array :\n{arr}\n")


# 2nd Part
min_arr = arr.min()
max_arr = arr.max()
mean_arr = arr.mean()

print(f"min : {min_arr}, max : {max_arr}, mean : {mean_arr}\n")

# 3rd Part
arr = np.where(arr<5, arr**2, arr)

print(f"replaced values less than 5 with square: \n{arr}\n")

# 4th Part
def numpy_alternate_sort(array: np.ndarray):
     # taking only 1d array as input
     if array.ndim != 1:
          raise ValueError("Input array must be a 1D numpy array.")

     # 
     array = np.sort(array)
     noElems = array.size
     sorted_ls = []

     for i in range((noElems+1)//2):
          if i != noElems-1-i:
               sorted_ls.extend([array[i], array[noElems-i-1]])
          else :
               sorted_ls.append(array[i])

     return np.array(sorted_ls, dtype=array.dtype)

# End
print(f"sorted array : \n{numpy_alternate_sort(arr)}")