from __future__ import print_function, division, absolute_import, unicode_literals

import tensorflow as tf

def weight_variable(shape, stddev=0.1):
    initial = tf.truncated_normal(shape, stddev=stddev, dtype=tf.float32)
    return tf.Variable(initial)

def weight_variable_devonc(shape, stddev=0.1):
    return tf.Variable(tf.truncated_normal(shape, stddev=stddev, dtype=tf.float32))

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape, dtype=tf.float32)
    return tf.Variable(initial)

def conv2d(x, W,keep_prob_):
    conv_2d = tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
    return tf.nn.dropout(conv_2d, keep_prob_)

def deconv2d(x, W,stride):
    x_shape = tf.shape(x)
    output_shape = tf.stack([x_shape[0], x_shape[1]*2, x_shape[2]*2, x_shape[3]//2])
    return tf.nn.conv2d_transpose(x, W, output_shape, strides=[1, stride, stride, 1], padding='SAME')

def max_pool(x,n):
    return tf.nn.max_pool(x, ksize=[1, n, n, 1], strides=[1, n, n, 1], padding='SAME')

def crop_and_concat(x1,x2):
    x1_shape = tf.shape(x1)
    x2_shape = tf.shape(x2)
    # offsets for the top left corner of the crop
    offsets = [0, (x1_shape[1] - x2_shape[1]) // 2, (x1_shape[2] - x2_shape[2]) // 2, 0]
    size = [-1, x2_shape[1], x2_shape[2], -1]
    x1_crop = tf.slice(x1, offsets, size)
    return tf.concat([x1_crop, x2], 3)   

def pixel_wise_softmax(output_map):
    exponential_map = tf.exp(output_map)
    evidence = tf.add(exponential_map,tf.reverse(exponential_map,axis=[3]))
    return tf.divide(exponential_map,evidence, name="pixel_wise_softmax")

def pixel_wise_softmax_2(output_map):
    exponential_map = tf.exp(output_map)
    sum_exp = tf.reduce_sum(exponential_map, 3, keep_dims=True)
    tensor_sum_exp = tf.tile(sum_exp, tf.stack([1, 1, 1, tf.shape(output_map)[3]]))
    return tf.divide(exponential_map,tensor_sum_exp)



def cross_entropy(y_,output_map):
    pred = tf.nn.softmax(output_map)
    return -tf.reduce_mean(y_*tf.log(tf.clip_by_value(pred,1e-5,1.0)), name="cross_entropy")
