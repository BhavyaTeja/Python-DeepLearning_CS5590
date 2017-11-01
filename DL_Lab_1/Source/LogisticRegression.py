import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import xlrd
from mpl_toolkits.mplot3d import Axes3D

dataset = '/Users/bhavyateja/Github_Projects/Python-DeepLearning_CS5590/DL_Lab_1/Documentation/Dataset/dataset_Facebook.xls'

# Step 1: read in data from the .xls file

book = xlrd.open_workbook(dataset, encoding_override="utf-8")
sheet = book.sheet_by_index(0)
data = np.asarray([sheet.row_values(i) for i in range(1, sheet.nrows)])
n_samples = sheet.nrows - 1



