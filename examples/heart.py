"""This is a script for heart disease data analysis"""

# Imports
import tensorflow as tf
import numpy as np
import time
import pandas as pd
import gc

# Constants
FILE_PATH = './data/heart.txt'
test_ratio = 0.2
learning_rate = 0.01
n_epochs = 15

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
y_train = np.array([y_train, 1 - y_train]).T

X_test = X[np.invert(mask)]
y_test = y[np.invert(mask)]
y_test = np.array([y_test, 1 - y_test]).T

print (X_train.shape, X_test.shape, y_train.shape, y_test.shape)

del data, X, y
gc.collect()

# Building a graph: placeholders

X = tf.placeholder(shape = [None, 9], dtype = tf.float32, name = "X")
y = tf.placeholder(shape = [None, 2], dtype = tf.float32, name = "y")

# Structure

normed = tf.contrib.layers.batch_norm(inputs = X)

dense_1 = tf.layers.dense(inputs = normed, units = 40)
dense_2 = tf.layers.dense(inputs = dense_1, units = 20)
dense_3 = tf.layers.dense(inputs = dense_2, units = 10)

logits = tf.layers.dense(inputs = dense_3, units = 2)

# Define loss
loss = tf.nn.softmax_cross_entropy_with_logits(labels = y, logits = logits)

# Optimizer
optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(loss)


with tf.Session() as sess:
	writer = tf.summary.FileWriter('./graphs', sess.graph)
	start_time = time.time()
	sess.run(tf.global_variables_initializer())
	for i in range(n_epochs):

		_, total_loss = sess.run([optimizer, loss], feed_dict = {X: X_train, y: y_train})

		print ("Loss this epoch: " + str(sum(total_loss)))

	print('Total time: {0} seconds'.format(time.time() - start_time))

	print('Optimization Finished!')

	# test the model
	preds = tf.nn.softmax(logits)
	correct_preds = tf.equal(tf.argmax(preds, 1), tf.argmax(y, 1))
	accuracy = tf.reduce_sum(tf.cast(correct_preds, tf.float32))

	acc = sess.run([accuracy], feed_dict = {X: X_test, y: y_test})

	print ('Accuracy on the test set: ' + str(sum(acc) / len(y_test)))

writer.close()