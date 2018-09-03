from __future__ import print_function, division, absolute_import, unicode_literals

import os
import shutil
import numpy as np
from collections import OrderedDict
import logging

import tensorflow as tf

from layers import (weight_variable, weight_variable_devonc, bias_variable, 
                            conv2d, deconv2d, max_pool, crop_and_concat, pixel_wise_softmax,
                            cross_entropy)

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

def create_conv_net(x, keep_prob, channels, n_class, layers=3, features_root=16, filter_size=3, pool_size=2, summaries=False):
    """
    Creates a new convolutional unet for the given parametrization.
    
    :param x: input tensor, shape [?,nx,ny,channels]
    :param keep_prob: dropout probability tensor
    :param channels: number of channels in the input image
    :param n_class: number of output labels
    :param layers: number of layers in the net
    :param features_root: number of features in the first layer
    :param filter_size: size of the convolution filter
    :param pool_size: size of the max pooling operation
    :param summaries: Flag if summaries should be created
    """
    
    logging.info("Layers {layers}, features {features}, filter size {filter_size}x{filter_size}, pool size: {pool_size}x{pool_size}".format(layers=layers,
                                                                                                           features=features_root,
                                                                                                           filter_size=filter_size,
                                                                                                           pool_size=pool_size))
    # Placeholder for the input image
    nx = tf.shape(x)[1]
    ny = tf.shape(x)[2]
    x_image = tf.reshape(x, tf.stack([-1,nx,ny,channels]))
    in_node = x_image
    batch_size = tf.shape(x_image)[0]
 
    weights = []
    biases = []
    convs = []
    pools = OrderedDict()
    deconv = OrderedDict()
    dw_h_convs = OrderedDict()
    up_h_convs = OrderedDict()
    
    in_size = 1000
    size = in_size
    # down layers
    for layer in range(0, layers):
        features = 2**layer*features_root
        stddev = np.sqrt(2 / (filter_size**2 * features))
        if layer == 0:
            w1 = weight_variable([filter_size, filter_size, channels, features], stddev)
        else:
            w1 = weight_variable([filter_size, filter_size, features//2, features], stddev)
            
        w2 = weight_variable([filter_size, filter_size, features, features], stddev)
        b1 = bias_variable([features])
        b2 = bias_variable([features])
        
        conv1 = conv2d(in_node, w1, keep_prob)
        tmp_h_conv = tf.nn.relu(conv1 + b1)
        conv2 = conv2d(tmp_h_conv, w2, keep_prob)
        dw_h_convs[layer] = tf.nn.relu(conv2 + b2)
        
        weights.append((w1, w2))
        biases.append((b1, b2))
        convs.append((conv1, conv2))
        
        size -= 4
        if layer < layers-1:
            pools[layer] = max_pool(dw_h_convs[layer], pool_size)
            in_node = pools[layer]
            size /= 2
        
    in_node = dw_h_convs[layers-1]
        
    # up layers
    for layer in range(layers-2, -1, -1):
        features = 2**(layer+1)*features_root
        stddev = np.sqrt(2 / (filter_size**2 * features))
        wd = weight_variable_devonc([pool_size, pool_size, features//2, features], stddev)
        bd = bias_variable([features//2])
        h_deconv = tf.nn.relu(deconv2d(in_node, wd, pool_size) + bd)
        h_deconv_concat = crop_and_concat(dw_h_convs[layer], h_deconv)
        deconv[layer] = h_deconv_concat
        
        w1 = weight_variable([filter_size, filter_size, features, features//2], stddev)
        w2 = weight_variable([filter_size, filter_size, features//2, features//2], stddev)
        b1 = bias_variable([features//2])
        b2 = bias_variable([features//2])
        
        conv1 = conv2d(h_deconv_concat, w1, keep_prob)
        h_conv = tf.nn.relu(conv1 + b1)
        conv2 = conv2d(h_conv, w2, keep_prob)
        in_node = tf.nn.relu(conv2 + b2)
        up_h_convs[layer] = in_node

        weights.append((w1, w2))
        biases.append((b1, b2))
        convs.append((conv1, conv2))
        
        size *= 2
        size -= 4

    # Output Map
    weight = weight_variable([1, 1, features_root, n_class], stddev)
    bias = bias_variable([n_class])
    #conv = conv2d(in_node, weight, tf.constant(1.0))
    conv = conv2d(in_node, weight, keep_prob)
    output_map = tf.nn.relu(conv + bias)
    up_h_convs["out"] = output_map
    
    # if summaries:
    #     for i, (c1, c2) in enumerate(convs):
    #         tf.summary.image('summary_conv_%02d_01'%i, get_image_summary(c1))
    #         tf.summary.image('summary_conv_%02d_02'%i, get_image_summary(c2))
    #         
    #     for k in pools.keys():
    #         tf.summary.image('summary_pool_%02d'%k, get_image_summary(pools[k]))
    #     
    #     for k in deconv.keys():
    #         tf.summary.image('summary_deconv_concat_%02d'%k, get_image_summary(deconv[k]))
    #         
    #     for k in dw_h_convs.keys():
    #         tf.summary.histogram("dw_convolution_%02d"%k + '/activations', dw_h_convs[k])

    #     for k in up_h_convs.keys():
    #         tf.summary.histogram("up_convolution_%s"%k + '/activations', up_h_convs[k])
            
    variables = []
    for w1,w2 in weights:
        variables.append(w1)
        variables.append(w2)
        
    for b1,b2 in biases:
        variables.append(b1)
        variables.append(b2)

    
    return output_map, variables, int(in_size - size)


class Unet(object):
    
    def __init__(self, conf):
        tf.reset_default_graph()
        self.n_class = conf.class_num
        
        self.images = tf.placeholder(tf.float32, shape=[None, None, None, conf.channels])

        for i in range(self.images.get_shape().as_list()[-1]):
            tf.summary.image('input_'+str(i), get_image_summary(self.images, i))
        
        self.labels = tf.placeholder(tf.uint8, shape=[None, None, None]) 
               
        self.y = tf.one_hot(indices = self.labels, depth = conf.class_num)
        
        tf.summary.image('label', get_image_summary(tf.expand_dims(self.labels, axis=3), 0))
        
        self.keep_prob = tf.placeholder(tf.float32) #dropout (keep probability)
        
        logits, self.variables, self.offset = create_conv_net(self.images, self.keep_prob, conf.channels, conf.class_num, layers=conf.layer_num, features_root=conf.root_features, filter_size=conf.filter_size, pool_size=conf.pool_size)
        
        #logits = tf.Print(logits, [logits], "kernel_{}: ".format("output"))

        self.loss = self._get_cost(logits, conf.cost, conf.class_weights)
        
        self.cross_entropy = tf.reduce_mean(cross_entropy(tf.reshape(self.y, [-1, conf.class_num]),
                                                          tf.reshape(logits+1e-10, [-1, conf.class_num])))
        
        #self.gradients_node = tf.gradients(self.conf.cost, self.variables)
        
        #self.predicter = pixel_wise_softmax(logits)
        self.prediction = tf.nn.softmax(logits + 1e-10)
        tf.summary.image('output', get_image_summary(tf.expand_dims(tf.argmax(self.prediction, 3),axis=3), 0))
        
        tf.summary.image('output_prob', get_image_summary(self.prediction,1))
        
        self.correct_pred = tf.equal(tf.argmax(self.prediction, 3), tf.argmax(self.y, 3))
        
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))
        
    def _get_cost(self, logits, cost_name, class_weights):
        """
        Constructs the cost function, either cross_entropy, weighted cross_entropy or dice_coefficient.
        Optional arguments are: 
        class_weights: weights for the different classes in case of multi-class imbalance
        regularizer: power of the L2 regularizers added to the loss function
        """
        
        #flat_logits = tf.reshape(logits, [-1, self.n_class])
        #flat_labels = tf.reshape(self.y, [-1, self.n_class])
        if cost_name == "cross_entropy":
            
            if class_weights is not None:
                class_weights = tf.constant(np.array(class_weights, dtype=np.float32))
        
                #weight_map = tf.multiply(flat_labels, class_weights)
                #weight_map = tf.reduce_sum(weight_map, axis=1)
                
                #loss_map = tf.nn.softmax_cross_entropy_with_logits(logits=flat_logits, labels=flat_labels)
                #weighted_loss = tf.multiply(loss_map, weight_map)
        
                #loss = tf.reduce_mean(weighted_loss)
               
                weight_map = tf.multiply(self.y, class_weights)
                #prob = pixel_wise_softmax(logits_input)
                prob = tf.nn.softmax(logits + 1e-10)
                loss_map = tf.multiply(self.y,tf.log(tf.clip_by_value(prob,1e-10,1.0)))
                loss = -tf.reduce_mean(tf.multiply(loss_map,class_weights))
            else:
                loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=flat_logits, 
                                                                              labels=flat_labels))
        elif cost_name == "dice_coefficient":
            eps = 1e-5
            prediction = pixel_wise_softmax_2(logits)
            intersection = tf.reduce_sum(prediction * self.y)
            union =  eps + tf.reduce_sum(prediction) + tf.reduce_sum(self.y)
            loss = -(2 * intersection/ (union))
            
        else:
            raise ValueError("Unknown cost function: "%cost_name)

        # regularizer = cost_kwargs.pop("regularizer", None)
        # if regularizer is not None:
        #     regularizers = sum([tf.nn.l2_loss(variable) for variable in self.variables])
        #     loss += (regularizer * regularizers)
            
        return loss

    def predict(self, model_path, x_test):
        """
        Uses the model to create a prediction for the given data
        
        :param model_path: path to the model checkpoint to restore
        :param x_test: Data to predict on. Shape [n, nx, ny, channels]
        :returns prediction: The unet prediction Shape [n, px, py, labels] (px=nx-self.offset/2) 
        """
        
        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            # Initialize variables
            sess.run(init)
        
            # Restore model weights from previously saved model
            self.restore(sess, model_path)
            
            y_dummy = np.empty((x_test.shape[0], x_test.shape[1], x_test.shape[2]))
            prediction = sess.run(self.prediction, feed_dict={self.x: x_test, self.y_specific: y_dummy, self.keep_prob: 1.})
            
        return prediction
    
    
    def eval(self, sess, x_test):
        #y_dummy = np.empty((x_test.shape[0], x_test.shape[1], x_test.shape[2]))
        prediction = sess.run(self.prediction, feed_dict={self.x: x_test, self.keep_prob: 1.})
        return prediction
            
    def save(self, sess, model_path):
        """
        Saves the current session to a checkpoint
        
        :param sess: current session
        :param model_path: path to file system location
        """
        
        saver = tf.train.Saver()
        save_path = saver.save(sess, model_path)
        return save_path
    
    def restore(self, sess, model_path):
        """
        Restores a session from a checkpoint
        
        :param sess: current session instance
        :param model_path: path to file system checkpoint location
        """
        
        saver = tf.train.Saver()
        saver.restore(sess, model_path)
        logging.info("Model restored from file: %s" % model_path)

def get_image_summary(img, idx=0):
    """
    Make an image summary for 4d tensor image with index idx
    """
    
    V = tf.slice(img, (0, 0, 0, idx), (1, -1, -1, 1))
    V -= tf.reduce_min(V)
    V /= tf.reduce_max(V)
    V *= 255
    
    img_w = tf.shape(img)[1]
    img_h = tf.shape(img)[2]
    V = tf.reshape(V, tf.stack((img_w, img_h, 1)))
    V = tf.transpose(V, (2, 0, 1))
    V = tf.reshape(V, tf.stack((-1, img_w, img_h, 1)))
    return V
