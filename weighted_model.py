import tensorflow as tf
FLAGS = tf.app.flags.FLAGS
from utils import *



def MNIST_weights(class_num):
    weights = {}

    filters_num = [64,64,64,64]
    dtype = tf.float32
    conv_initializer = tf.contrib.layers.xavier_initializer_conv2d(dtype=dtype)
    fc_initializer = tf.contrib.layers.xavier_initializer(dtype=dtype)
    k = 3  # channel
    
    weights['conv1'] = tf.get_variable('conv1', [k, k, 1, filters_num[0]],
                                       initializer=conv_initializer, dtype=dtype)
    weights['b1'] = tf.Variable(tf.zeros([filters_num[0]]), name='b1')
    weights['conv2'] = tf.get_variable('conv2', [k, k, filters_num[0], filters_num[1]],
                                       initializer=conv_initializer, dtype=dtype)
    weights['b2'] = tf.Variable(tf.zeros([filters_num[1]]), name='b2')
    weights['conv3'] = tf.get_variable('conv3', [k, k, filters_num[1], filters_num[2]],
                                       initializer=conv_initializer, dtype=dtype)
    weights['b3'] = tf.Variable(tf.zeros([filters_num[2]]), name='b3')
    weights['conv4'] = tf.get_variable('conv4', [k, k, filters_num[2], filters_num[3]],
                                       initializer=conv_initializer, dtype=dtype)
    weights['b4'] = tf.Variable(tf.zeros([filters_num[3]]), name='b4')


    weights['w5'] = tf.Variable(tf.random_normal([filters_num[3], class_num]), name='w5')
    weights['b5'] = tf.Variable(tf.zeros([class_num]), name='b5')
    return weights
   

def cifar_weights(class_num):
    weights = {}
 
    filters_num = [64, 64, 64, 64]
    dtype = tf.float32
    conv_initializer = tf.contrib.layers.xavier_initializer_conv2d(dtype=dtype)
    fc_initializer = tf.contrib.layers.xavier_initializer(dtype=dtype)
    k = 3  # channel
    
    weights['conv1'] = tf.get_variable('conv1', [k, k, 3, filters_num[0]],
                                       initializer=conv_initializer, dtype=dtype)
    weights['b1'] = tf.Variable(tf.zeros([filters_num[0]]), name='b1')
    weights['conv2'] = tf.get_variable('conv2', [k, k, filters_num[0], filters_num[1]],
                                       initializer=conv_initializer, dtype=dtype)
    weights['b2'] = tf.Variable(tf.zeros([filters_num[1]]), name='b2')
    weights['conv3'] = tf.get_variable('conv3', [k, k, filters_num[1], filters_num[2]],
                                       initializer=conv_initializer, dtype=dtype)
    weights['b3'] = tf.Variable(tf.zeros([filters_num[2]]), name='b3')
    weights['conv4'] = tf.get_variable('conv4', [k, k, filters_num[2], filters_num[3]],
                                       initializer=conv_initializer, dtype=dtype)
    weights['b4'] = tf.Variable(tf.zeros([filters_num[3]]), name='b4')

    weights['w5'] = tf.get_variable('w5', [filters_num[3] * 4, class_num],initializer=fc_initializer)
    weights['b5'] = tf.Variable(tf.zeros([class_num]), name='b5')

    return weights
    

def fashion_weights(class_num):
    weights = {}

    
    filters_num = [64,64,64,64]
    dtype = tf.float32
    conv_initializer = tf.contrib.layers.xavier_initializer_conv2d(dtype=dtype)
    fc_initializer = tf.contrib.layers.xavier_initializer(dtype=dtype)
    k = 3  # channel
   
    weights['conv1'] = tf.get_variable('conv1', [k, k, 1, filters_num[0]],
                                       initializer=conv_initializer, dtype=dtype)
    weights['b1'] = tf.Variable(tf.zeros([filters_num[0]]), name='b1')
    weights['conv2'] = tf.get_variable('conv2', [k, k, filters_num[0], filters_num[1]],
                                       initializer=conv_initializer, dtype=dtype)
    weights['b2'] = tf.Variable(tf.zeros([filters_num[1]]), name='b2')
    weights['conv3'] = tf.get_variable('conv3', [k, k, filters_num[1], filters_num[2]],
                                       initializer=conv_initializer, dtype=dtype)
    weights['b3'] = tf.Variable(tf.zeros([filters_num[2]]), name='b3')
    weights['conv4'] = tf.get_variable('conv4', [k, k, filters_num[2], filters_num[3]],
                                       initializer=conv_initializer, dtype=dtype)
    weights['b4'] = tf.Variable(tf.zeros([filters_num[3]]), name='b4')

    weights['w5'] = tf.Variable(tf.random_normal([filters_num[3], class_num]), name='w5')
    weights['b5'] = tf.Variable(tf.zeros([class_num]), name='b5')
    return weights
    

def STL_weights(class_num):
    weights = {}
    
    filters_num = [64,64,64,64,32,8]
    dtype = tf.float32
    conv_initializer = tf.contrib.layers.xavier_initializer_conv2d(dtype=dtype)
    fc_initializer = tf.contrib.layers.xavier_initializer(dtype=dtype)
    k = 3  # channel
    print('network structure:', filters_num, 'kernel:', k)
    weights['conv1'] = tf.get_variable('conv1', [k, k, 3, filters_num[0]],
                                       initializer=conv_initializer, dtype=dtype)
    weights['b1'] = tf.Variable(tf.zeros([filters_num[0]]), name='b1')
    weights['conv2'] = tf.get_variable('conv2', [k, k, filters_num[0], filters_num[1]],
                                       initializer=conv_initializer, dtype=dtype)
    weights['b2'] = tf.Variable(tf.zeros([filters_num[1]]), name='b2')
    weights['conv3'] = tf.get_variable('conv3', [k, k, filters_num[1], filters_num[2]],
                                       initializer=conv_initializer, dtype=dtype)
    weights['b3'] = tf.Variable(tf.zeros([filters_num[2]]), name='b3')
    weights['conv4'] = tf.get_variable('conv4', [k, k, filters_num[2], filters_num[3]],
                                       initializer=conv_initializer, dtype=dtype)
    weights['b4'] = tf.Variable(tf.zeros([filters_num[3]]), name='b4')
    weights['conv5'] = tf.get_variable('conv5', [k, k, filters_num[3], filters_num[4]],
                                       initializer=conv_initializer, dtype=dtype)
    weights['b5'] = tf.Variable(tf.zeros([filters_num[4]]), name='b5')
    weights['conv6'] = tf.get_variable('conv6', [k, k, filters_num[4], filters_num[5]],
                                       initializer=conv_initializer, dtype=dtype)
    weights['b6'] = tf.Variable(tf.zeros([filters_num[5]]), name='b6')

    weights['w7'] = tf.get_variable('w7', [filters_num[5] * 4, class_num],initializer=fc_initializer)
    weights['b7'] = tf.Variable(tf.zeros([class_num]), name='b7')

    return weights
   