# Logistic Regression
# ----------------------------------
#
# This function shows how to use TensorFlow to
# solve logistic regression.
# y = sigmoid(Ax + b)
#
# We will use the low birth weight data, specifically:
#  y = 0 or 1 = low birth weight
#  x = demographic and medical history data

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import requests
from tensorflow.python.framework import ops
import os.path
import csv

ops.reset_default_graph()

# Create graph

sess = tf.Session()

# Link to the dataset that is stored locally

dataset = '/Users/bhavyateja/Github_Projects/Python-DeepLearning_CS5590/DL_Lab_1/Documentation/Dataset/dataset.csv'

# Reading the dataset by appending every row

data = []
with open(dataset, newline='') as file:
    csv_reader = csv.reader(file)
    birth_header = next(csv_reader)
    for row in csv_reader:
        data.append(row)

# Coverting the strings to the float values for computation later

data = [[float(x) for x in row] for row in data]

# Pulling out target variable which is dependant

y_vals = np.array([x[0] for x in data])

# Pull out predictor variables (not id, not target, and not birthweight) which are independant

x_vals = np.array([x[1:8] for x in data])

# set for reproducible results to randomize the test and train datasets

seed = 108
np.random.seed(seed)
tf.set_random_seed(seed)

# Split dataset into train/test sets = 80%/20%

train_indices = np.random.choice(len(x_vals), round(len(x_vals) * 0.8), replace=False)
test_indices = np.array(list(set(range(len(x_vals))) - set(train_indices)))
x_train = x_vals[train_indices]
x_test = x_vals[test_indices]
y_train = y_vals[train_indices]
y_test = y_vals[test_indices]


# Normalize by column (min-max norm) as a part of feature scaling

def normalize_cols(m):
    col_max = m.max(axis=0)
    col_min = m.min(axis=0)
    return (m - col_min) / (col_max - col_min)


x_train = np.nan_to_num(normalize_cols(x_train))
x_test = np.nan_to_num(normalize_cols(x_test))

# Declare batch size

batch_size = 25

# Initialize placeholders

x_data = tf.placeholder(shape=[None, 7], dtype=tf.float32)
y_target = tf.placeholder(shape=[None, 1], dtype=tf.float32)

# Create variables for linear regression

A = tf.Variable(tf.random_normal(shape=[7, 1]))
b = tf.Variable(tf.random_normal(shape=[1, 1]))

# Declare model operations

model_output = tf.add(tf.matmul(x_data, A), b)

# Declare loss function (Cross Entropy loss)

loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=model_output, labels=y_target))

# Declare optimizer

my_opt = tf.train.GradientDescentOptimizer(0.01)
train_step = my_opt.minimize(loss)

# Initialize variables

init = tf.global_variables_initializer()
sess.run(init)

# Actual Prediction

prediction = tf.round(tf.sigmoid(model_output))
predictions_correct = tf.cast(tf.equal(prediction, y_target), tf.float32)
accuracy = tf.reduce_mean(predictions_correct)

# Training loop

loss_vec = []
train_acc = []
test_acc = []

try:
    for i in range(1500):
        rand_index = np.random.choice(len(x_train), size=batch_size)
        rand_x = x_train[rand_index]
        rand_y = np.transpose([y_train[rand_index]])
        sess.run(train_step, feed_dict={x_data: rand_x, y_target: rand_y})
        temp_loss = sess.run(loss, feed_dict={x_data: rand_x, y_target: rand_y})
        loss_vec.append(temp_loss)
        temp_acc_train = sess.run(accuracy, feed_dict={x_data: x_train, y_target: np.transpose([y_train])})
        train_acc.append(temp_acc_train)
        temp_acc_test = sess.run(accuracy, feed_dict={x_data: x_test, y_target: np.transpose([y_test])})
        test_acc.append(temp_acc_test)
        if (i + 1) % 300 == 0:
            print('Loss = ' + str(temp_loss))
except ValueError as e:
    print(e)

# Displaying the performance of the model

# Plot loss over time

plt.plot(loss_vec, 'k-')
plt.title('Cross Entropy Loss per Generation')
plt.xlabel('Generation')
plt.ylabel('Cross Entropy Loss')
plt.show()

# Plot train and test accuracy

plt.plot(train_acc, 'k-', label='Training Set Accuracy')
plt.plot(test_acc, 'r--', label='Test Set Accuracy')
plt.title('Train and Test set Accuracy')
plt.xlabel('Generation')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.show()
