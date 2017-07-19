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

dense_1 = tf.layers.dense(inputs = normed, units = 200, activation = 'tahn')