"""This is a script for heart disease data analysis"""

# Imports
import tensorflow as tf
import numpy as np
import time
import pandas as pd
import gc

# Constants
FILE_PATH = './data/heart.txt'
test_ratio = 0.1
learning_rate = 0.01

# Reading the data
data = pd.read_csv(FILE_PATH, sep = "\s+")

# Data processing

data['famhist'] = (data['famhist'] == "Present").astype(int)

X = data.drop('chd', axis = 1)
y = data.chd

# Switch to numpy from pandas
X = X.values
y = y.values

# Train-test split
mask = np.random.choice([True, False], size = X.shape[0], p = [1 - test_ratio, test_ratio])

X_train = X[mask]
y_train = y[mask]

X_test = X[np.invert(mask)]
y_test = y[np.invert(mask)]

del data, X, y
gc.collect()

# Building a graph: placeholders

X = tf.placeholder(shape = [None, 9], name = "X")
y = tf.placeholder(shape = [None, 1], name = "y")

# Structure

normed = tf.contrib.layers.batch_norm(inputs = X, name = "batch_norm")

dense_1 = tf.layers.dense(inputs = normed, units = 40, activation = 'tahn')
dense_2 = tf.layers.dense(inputs = dense_1, units = 20, activation = 'tahn')
dense_3 = tf.layers.dense(inputs = dense_2, units = 10, activation = 'tahn')

output = tf.layers.dense(inputs = dense_3, units = 1)

# Define loss
loss = tf.nn.softmax_cross_entropy_with_logits(labels = y, logits = output)

# Optimizer
optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(loss)
