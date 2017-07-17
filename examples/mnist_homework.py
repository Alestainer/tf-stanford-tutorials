"""Based on the file 03_logistic_regression_mnist_starter.py
Here I am improving accuracy by using more sophisticated model"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import time

# Define paramaters for the model
learning_rate = 0.0001
batch_size = 128
n_epochs = 140

# Step 1: Read in data
# using TF Learn's built in function to load MNIST data to the folder data/mnist
mnist = input_data.read_data_sets('./data/mnist', one_hot=True) 

# Step 2: create placeholders for features and labels
# each image in the MNIST data is of shape 28*28 = 784
# therefore, each image is represented with a 1x784 tensor
# there are 10 classes for each image, corresponding to digits 0 - 9. 
# Features are of the type float, and labels are of the type int

X = tf.placeholder(tf.float32, shape = [None , 784], name = 'X')
Y = tf.placeholder(tf.int32, shape = [None , 10], name = 'y')


# Step 3: create weights and bias
# weights and biases are initialized to 0
# shape of w depends on the dimension of X and Y so that Y = X * w + b
# shape of b depends on Y

# Hidden layers of ReLU and dropouts
normed = tf.contrib.layers.batch_norm(inputs = X)
h1 = tf.contrib.layers.fully_connected(inputs = normed, num_outputs = 40)
h2 = tf.contrib.layers.fully_connected(inputs = h1, num_outputs = 20)
hidden_out = tf.contrib.layers.fully_connected(inputs = h2, num_outputs = 40)

# Final logistic regression
W = tf.Variable(tf.zeros(shape = [40, 10]), name = 'weights')
b = tf.Variable(tf.zeros(shape = [10]), name = 'biases')

# Step 4: build model
# the model that returns the logits.
# this logits will be later passed through softmax layer
# to get the probability distribution of possible label of the image
# DO NOT DO SOFTMAX HERE

logits = tf.matmul(hidden_out, W) + b

# Step 5: define loss function
# use cross entropy loss of the real labels with the softmax of logits
# use the method:
# tf.nn.softmax_cross_entropy_with_logits(logits, Y)
# then use tf.reduce_mean to get the mean loss of the batch

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = Y, logits = logits))

# Step 6: define training op
# using gradient descent to minimize loss

optimizer = tf.train.AdamOptimizer(learning_rate = 0.001).minimize(loss)

with tf.Session() as sess:
	start_time = time.time()
	sess.run(tf.global_variables_initializer())	
	n_batches = int(mnist.train.num_examples/batch_size)
	for i in range(n_epochs): # train the model n_epochs times
		total_loss = 0

		for _ in range(n_batches):
			X_batch, Y_batch = mnist.train.next_batch(batch_size)
			# TO-DO: run optimizer + fetch loss_batch
			# 
			_, loss_batch = sess.run([optimizer, loss], feed_dict = {X: X_batch, Y: Y_batch})

			total_loss += loss_batch
		print('Average loss epoch {0}: {1}'.format(i, total_loss/n_batches))

	print('Total time: {0} seconds'.format(time.time() - start_time))

	print('Optimization Finished!') # should be around 0.35 after 25 epochs

	# test the model
	preds = tf.nn.softmax(logits)
	correct_preds = tf.equal(tf.argmax(preds, 1), tf.argmax(Y, 1))
	accuracy = tf.reduce_sum(tf.cast(correct_preds, tf.float32)) # need numpy.count_nonzero(boolarr) :(
	
	n_batches = int(mnist.test.num_examples/batch_size)
	total_correct_preds = 0
	
	for i in range(n_batches):
		X_batch, Y_batch = mnist.test.next_batch(batch_size)
		accuracy_batch = sess.run([accuracy], feed_dict={X: X_batch, Y:Y_batch})
		total_correct_preds += accuracy_batch[0]	
	
	print('Accuracy {0}'.format(total_correct_preds/mnist.test.num_examples))
