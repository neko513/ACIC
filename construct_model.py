import tensorflow as tf
FLAGS = tf.app.flags.FLAGS
from utils import *
import numpy as np


def MNIST_model(data, weights, reuse=False):
    
    width = height = FLAGS.imagesize
    inp = tf.reshape(data, [-1, width, height, 1])
    scope = ''
    hidden1 = conv_block(inp, weights['conv1'], weights['b1'], reuse, scope + '0')
    hidden2 = conv_block(hidden1, weights['conv2'], weights['b2'], reuse, scope + '1')
    hidden3 = conv_block(hidden2, weights['conv3'], weights['b3'], reuse, scope + '2')
    hidden4 = conv_block(hidden3, weights['conv4'], weights['b4'], reuse, scope + '3')
    hidden4 = tf.reduce_mean(hidden4, [1, 2])

    return tf.nn.softmax(tf.matmul(hidden4, weights['w5']) + weights['b5'])


def cifar_model(data, weights, reuse=False):
    
    width = height = FLAGS.imagesize
    inp = tf.reshape(data, [-1, width, height, 3])
    scope = ''
    hidden1 = conv_block(inp, weights['conv1'], weights['b1'], reuse, scope + '0')
    hidden2 = conv_block(hidden1, weights['conv2'], weights['b2'], reuse, scope + '1')
    hidden3 = conv_block(hidden2, weights['conv3'], weights['b3'], reuse, scope + '2')
    hidden4 = conv_block(hidden3, weights['conv4'], weights['b4'], reuse, scope + '3')
    hidden4 = tf.reshape(hidden4, [-1, np.prod([int(dim) for dim in hidden4.get_shape()[1:]])])

    return tf.nn.softmax(tf.matmul(hidden4, weights['w5']) + weights['b5'])


def fashion_model(data, weights, reuse=False):
    
    width = height = FLAGS.imagesize
    inp = tf.reshape(data, [-1, width, height, 1])
    scope = ''
    hidden1 = conv_block(inp, weights['conv1'], weights['b1'], reuse, scope + '0')
    hidden2 = conv_block(hidden1, weights['conv2'], weights['b2'], reuse, scope + '1')
    hidden3 = conv_block(hidden2, weights['conv3'], weights['b3'], reuse, scope + '2')
    hidden4 = conv_block(hidden3, weights['conv4'], weights['b4'], reuse, scope + '3')
    hidden4 = tf.reduce_mean(hidden4, [1, 2])

    return tf.nn.softmax(tf.matmul(hidden4, weights['w5']) + weights['b5'])


def STL_model(data, weights, reuse=False):

    width = height = FLAGS.imagesize
    inp = tf.reshape(data, [-1, width, height, 3])
    scope = ''
    hidden1 = conv_block(inp, weights['conv1'], weights['b1'], reuse, scope + '0')
    hidden2 = conv_block(hidden1, weights['conv2'], weights['b2'], reuse, scope + '1')
    hidden3 = conv_block(hidden2, weights['conv3'], weights['b3'], reuse, scope + '2')
    hidden4 = conv_block(hidden3, weights['conv4'], weights['b4'], reuse, scope + '3')
    hidden5 = conv_block(hidden4, weights['conv5'], weights['b5'], reuse, scope + '4')
    hidden6 = conv_block(hidden5, weights['conv6'], weights['b6'], reuse, scope + '5')

    # last hidden layer is 6x6x64-ish, reshape to a vector
    hidden6 = tf.reshape(hidden6, [-1, np.prod([int(dim) for dim in hidden6.get_shape()[1:]])])
    return tf.nn.softmax(tf.matmul(hidden6, weights['w7']) + weights['b7'])
