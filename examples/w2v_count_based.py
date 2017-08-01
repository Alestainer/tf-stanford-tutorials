""" The mo frills implementation of word2vec skip-gram model using NCE loss. 
Author: Chip Huyen
Prepared for the class CS 20SI: "TensorFlow for Deep Learning Research"
cs20si.stanford.edu
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import numpy as np
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector

from process_data import process_data

VOCAB_SIZE = 10000
BATCH_SIZE = 128
EMBED_SIZE = 128 # dimension of the word embedding vectors
SKIP_WINDOW = 1 # the context window
NUM_SAMPLED = 64    # Number of negative examples to sample.
LEARNING_RATE = 1.0
NUM_TRAIN_STEPS = 10000
SKIP_STEP = 2000 # how many steps to skip before reporting the loss

def word2vec(batch_gen):
    """ Build the graph for word2vec model and train it """
    # Step 1: define the placeholders for input and output
    with tf.name_scope('data'):
        center_words = tf.placeholder(tf.int32, shape=[BATCH_SIZE], name='center_words')
        target_words = tf.placeholder(tf.int32, shape=[BATCH_SIZE, 1], name='target_words')

    # Assemble this part of the graph on the CPU. You can change it to GPU if you have GPU
    # Step 2: define weights. In word2vec, it's actually the weights that we care about

    with tf.name_scope('co_occurence_matrix'):
        cooc_matrix = tf.Variable(tf.zeros([VOCAB_SIZE, VOCAB_SIZE]), 
                            name='co_occurence_matrix')

    with tf.name_scope('matrix_update'):
        target_reshape = tf.reshape(target_words, shape = [BATCH_SIZE])
        indices = tf.stack([center_words, target_reshape], axis = 1)
        updates = tf.ones(shape = [BATCH_SIZE])
        add = tf.scatter_nd_add(cooc_matrix, indices, updates)

    # # Step 3: define the inference
    # with tf.name_scope('loss'):
    #     embed = tf.nn.embedding_lookup(embed_matrix, center_words, name='embed')

    #     # Step 4: construct variables for NCE loss
    #     nce_weight = tf.Variable(tf.truncated_normal([VOCAB_SIZE, EMBED_SIZE],
    #                                                 stddev=1.0 / (EMBED_SIZE ** 0.5)), 
    #                                                 name='nce_weight')
    #     nce_bias = tf.Variable(tf.zeros([VOCAB_SIZE]), name='nce_bias')

    #     # define loss function to be NCE loss function
    #     loss = tf.reduce_mean(tf.nn.nce_loss(weights=nce_weight, 
    #                                         biases=nce_bias, 
    #                                         labels=target_words, 
    #                                         inputs=embed, 
    #                                         num_sampled=NUM_SAMPLED, 
    #                                         num_classes=VOCAB_SIZE), name='loss')

    # # Step 5: define optimizer
    # optimizer = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(loss)

    with tf.name_scope('SVD'):
        _, embeddings, _ = tf.svd(cooc_matrix, full_matrices = False)
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # total_loss = 0.0 # we use this to calculate late average loss in the last SKIP_STEP steps
        writer = tf.summary.FileWriter('./graphs/cooc/', sess.graph)
        for index in range(NUM_TRAIN_STEPS):
            centers, targets = next(batch_gen)
            # loss_batch, _ = sess.run([loss, optimizer], 
            #                         feed_dict={center_words: centers, target_words: targets})
            # total_loss += loss_batch
            # if (index + 1) % SKIP_STEP == 0:
            #     print('Average loss at step {}: {:5.1f}'.format(index, total_loss / SKIP_STEP))
            #     total_loss = 0.0
            sess.run(add, feed_dict = {center_words: centers, target_words: targets})
            if (index % 1000 == 0):
                print ("Step #" + str(index))
        print (sess.run(embeddings))
        writer.close()

def main():
    batch_gen = process_data(VOCAB_SIZE, BATCH_SIZE, SKIP_WINDOW)
    word2vec(batch_gen)

if __name__ == '__main__':
    main()