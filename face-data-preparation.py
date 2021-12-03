import os
import numpy as np
import helper

proj_path = os.getcwd()
FACE_DATA_PATH = proj_path + "\\dataset\\faces"
NUMPY_DATA_PATH = proj_path + "\\dataset\\numpy"

X_data_name = 'X_data.npy'
Y_data_name = 'Y_data.npy'

TOTAL_SAMPLE_SIZE = 20000
SIZE_TO_REDUCE = 2

X, Y = helper.get_data(FACE_DATA_PATH, SIZE_TO_REDUCE, TOTAL_SAMPLE_SIZE)

if not os.path.exists(NUMPY_DATA_PATH):
  os.mkdir(NUMPY_DATA_PATH)

X_file = os.path.join(NUMPY_DATA_PATH, X_data_name)
Y_file = os.path.join(NUMPY_DATA_PATH, Y_data_name) 

if os.path.exists(X_file):
  os.remove(X_file)

if os.path.exists(Y_file):
  os.remove(Y_file)

np.save(X_file, X)
np.save(Y_file, Y)