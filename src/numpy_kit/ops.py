import math

import numpy as np


# Softmax loop
def softmax_loop_1d_loop(arr):
  if arr.size == 0:
    raise ValueError("Array cannot be empty")
  softmax_arr = np.zeros(len(arr))
  exp_arr = np.zeros(len(arr))

  max_val = arr[0]
  for i in range(1, len(arr)):
    max_val = max(max_val, arr[i])
  
  for i in range(len(arr)):
    exp_arr[i] = np.exp(arr[i] - max_val)

  # Divide by the total sum of exponentials
  sum_of_exp = 0
  for i in range(len(arr)):
    sum_of_exp += exp_arr[i]

  for i in range(len(arr)):
    softmax_arr[i] = exp_arr[i] / sum_of_exp
  
  return softmax_arr


def softmax_loop(arr):
  float_arr = arr.astype(float)
  if float_arr.ndim == 1:
    return softmax_loop_1d_loop(float_arr)
  elif float_arr.ndim == 2:
    for i in range(float_arr.shape[0]):
      float_arr[i] = softmax_loop_1d_loop(float_arr[i])
    
    return float_arr
  else: 
    raise ValueError('Input array must be either 1D or 2D array')
  

# Softmax Vectorized
def softmax(arr):
  float_arr = arr.astype(float)
  row_max = np.max(float_arr, axis = -1, keepdims= True)
  float_arr = np.exp(float_arr - row_max)
  row_sum = np.sum(float_arr, axis = -1, keepdims= True)
  r = float_arr  / row_sum
  return r



# ------------------------------------------

# Pairwise distance loop 

def pairwise_loop(a, b):
  if a.ndim != 2 or b.ndim != 2:
    raise ValueError("Both arrays must be 2D arrays")
  if a.shape[1] != b.shape[1]:
    raise ValueError("Both arrays must have the same number of coordinates")

  d = np.zeros((a.shape[0], b.shape[0]))
  for i in range(a.shape[0]):
    for j in range(b.shape[0]):
      total = 0
      for c in range(a.shape[1]):
        total += (a[i,c] - b[j,c]) ** 2
      d[i,j] =math.sqrt(total)
  
  return d

# Pairwise vectorized

def pairwise(a,b):
  if a.ndim != 2 or b.ndim != 2:
    raise ValueError("Both arrays must be 2D arrays")
  if a.shape[1] != b.shape[1]:
    raise ValueError("Both arrays must have the same number of coordinates")
  
  diff = a[:, None, :] - b[None, :, :]
  diff = np.square(diff)
  d = np.sum(diff, axis = 2)
  d = np.sqrt(d)

  return d


# Pairwise Distance Fast

def pairwise_v2(a,b):
  if a.ndim != 2 or b.ndim != 2:
    raise ValueError("Both arrays must be 2D arrays")
  if a.shape[1] != b.shape[1]:
    raise ValueError("Both arrays must have the same number of coordinates")

  a_sq = np.sum(np.square(a), axis = 1)
  b_sq = np.sum(np.square(b), axis = 1)

  ab = a @ b.T

  squared_diff = a_sq[:, None] + b_sq[None, :] - (2 * ab)

  d = np.sqrt(np.clip(squared_diff, 0, None))
  
  return d

# -------------------------------------
# Standardize

def standardize_loop(x):
  if x.ndim != 2:
    raise ValueError("Array must be a 2D array")
  
  z = np.zeros(x.shape)

  mean_col = np.zeros(x.shape[1])
  n = x.shape[0]
  for j in range(x.shape[1]):
    total = 0
    for i in range(x.shape[0]):
      total += x[i,j]
    mean_col[j] = total/n
  
  std_col = np.zeros(x.shape[1])

  for j in range(x.shape[1]):
    temp = 0
    for i in range(x.shape[0]):
      temp += (x[i,j] - mean_col[j]) ** 2

    std_col[j] = math.sqrt((1/n) * (temp))
  

  for i in range(x.shape[0]):
    for j in range(x.shape[1]):
      if std_col[j] < 1e-12:
        z[i,j] = 0
      else:
        z[i,j] = (x[i,j] - mean_col[j]) / std_col[j]



  return z, mean_col, std_col


# Standardize Vectorized

def standardize(x):
  if x.ndim != 2:
    raise ValueError("Array must be a 2D array")
  
  means = x.mean(axis = 0)
  stds = x.std(axis = 0)
  
  safe_stds = np.where(stds < 1e-12, 1, stds)
  z = (x - means) / safe_stds


  return z, means, stds



    














