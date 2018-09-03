#!/usr/bin/env python
# coding=utf-8

from __future__ import print_function, division
from unet import Unet

import os
import glob
import copy
import shutil
import datetime
import copy
import skimage.measure
import numpy as np
import nibabel as nib
import tensorflow as tf
from medpy import metric
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

class Operationer(object):
    def __init__(self, net, data_provider, conf):
        np.random.seed(datetime.datetime.now().second)
        self.data_provider = data_provider
        self.net = net
        self.need_restore = conf.need_restore
        self.batch_size = conf.batch_size
        self.learning_rate = conf.learning_rate
        self.epochs = conf.epochs
        self.training_iters = len(data_provider.train_sample_list)//conf.batch_size
        self.display_step = conf.display_step
        self.global_step = tf.Variable(0)
        self.optimizer = conf.optimizer
        self.decay_rate = conf.decay_rate
        self.momentum = conf.momentum
        self.train_op = self.get_train_op(self.training_iters, self.global_step)
        self.log_path = conf.log_path
        self.model_path = conf.model_path
        self.prediction_path = conf.prediction_path
        if not os.path.exists(self.prediction_path):
            os.makedirs(self.prediction_path)
        logging.info('batch_size: {}, learning_rate: {}, optimizer: {}'.format(self.batch_size, self.learning_rate, self.optimizer))
        logging.info('epochs: {}, training_iters: {}'.format(self.epochs, self.training_iters))

    def get_train_op(self, training_iters, global_step):
        if self.optimizer == "momentum":
            self.learning_rate_node= tf.train.exponential_decay(learning_rate=self.learning_rate, 
                                                        global_step=global_step, 
                                                        decay_steps=training_iters,  
                                                        decay_rate=self.decay_rate, 
                                                        staircase=True)
            
            train_op = tf.train.MomentumOptimizer(learning_rate=self.learning_rate_node, momentum=self.momentum).minimize(self.net.loss, global_step=global_step)
        elif self.optimizer == "adam":
            self.learning_rate_node = tf.Variable(self.learning_rate)
            train_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.net.loss, global_step=global_step)
        
        return train_op


    def cal_coordinates(self, mask):
        mask = np.squeeze(mask)
        ind = np.where(mask>0)
        pad_multiple = 16

        x_min = ind[0].min()
        x_max = ind[0].max()
        y_min = ind[1].min()
        y_max = ind[1].max()

        width = np.max((1, x_max - x_min))
        height = np.max((1, y_max - y_min))

        new_half_width = (int(np.ceil(width / pad_multiple)) * pad_multiple) // 2
        new_half_height = (int(np.ceil(height / pad_multiple)) * pad_multiple) // 2

        center_x = (x_min + x_max) // 2
        center_y = (y_min + y_max) // 2

        new_x_min = np.max((0, center_x - new_half_width)) 
        new_x_max = new_x_min + new_half_width * 2
        new_y_min = np.max((0, center_y - new_half_height))
        new_y_max = new_y_min + new_half_height * 2

        return new_x_min, new_x_max, new_y_min, new_y_max

    def preprocess(self, img, mask):
        # crop_coordinates = self.cal_coordinates(mask)
        # img = img[:, crop_coordinates[0]:crop_coordinates[1], crop_coordinates[2]:crop_coordinates[3], :]
        # mask = mask[:, crop_coordinates[0]:crop_coordinates[1], crop_coordinates[2]:crop_coordinates[3]]

        # img[mask == 0] = 0

        mask[mask > 0] = 1

        return img, mask

    def train(self):
        tf.summary.scalar('loss', self.net.loss)
        tf.summary.scalar('accuracy', self.net.accuracy)
        tf.summary.scalar('learning_rate', self.learning_rate_node)

        logging.info("Start optimization")
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            if self.need_restore:
                ckpt = tf.train.get_checkpoint_state(self.model_path)
                if ckpt and ckpt.model_checkpoint_path:
                    self.restore(sess, ckpt.model_checkpoint_path)
            else:
                logging.info("Removing '{:}'".format(self.log_path))
                shutil.rmtree(self.log_path, ignore_errors=True)
                logging.info("Removing '{:}'".format(self.model_path))
                shutil.rmtree(self.model_path, ignore_errors=True)

            if not os.path.exists(self.log_path):
                logging.info("Allocating '{:}'".format(self.log_path))
                os.makedirs(self.log_path)
            if not os.path.exists(self.model_path):
                logging.info("Allocating '{:}'".format(self.model_path))
                os.makedirs(self.model_path)


            self.summary_op = tf.summary.merge_all()        
            summary_writer = tf.summary.FileWriter(self.log_path, graph=sess.graph)

            for epoch in range(self.epochs):

                for step in range((epoch*self.training_iters), ((epoch+1)*self.training_iters)):
                    batch_x, batch_y = self.data_provider.get_random_batch(self.batch_size)
                    batch_x, batch_y = self.preprocess(batch_x, batch_y)

                    # batch_x, batch_y = self.data_provider.random_2d_rotation(batch_x, batch_y)
                    _, loss = sess.run((self.train_op, self.net.loss), 
                                                      feed_dict={self.net.images: batch_x,
                                                                 self.net.labels: batch_y,
                                                                 self.net.keep_prob: 1.0})
                    if step % self.display_step == 0:
                        self.output_minibatch_stats(sess, summary_writer, epoch, step, batch_x, batch_y)

                self.eval(sess)
                save_path = self.save(sess, os.path.join(self.model_path, 'model_' + str(epoch) + '.ckpt'))
            logging.info("Optimization Finished!")
            return save_path

    def single_connected_domain_filter(self, volume, label):
        output = copy.deepcopy(volume)
        label_3D = skimage.measure.label(volume == label)
        max_label = np.max(label_3D)
        max_size = 0
        max_label_id = 0
        for label in range(1, max_label + 1):
            label_size = len(np.where(label_3D == label)[0])
            if label_size > max_size:
               max_size = label_size
               max_label_id = label
        output[np.where(label_3D != max_label_id)] = 0
        return output


    def eval(self, sess, save_result=False):
        logging.info('Validation Start')

        score_list = []

        val_volume_list = self.data_provider.get_val_volume_list()
        for volume_filename in val_volume_list:
            data_volume, gt_seg_volume, affine, case_id = self.data_provider.read_volume(volume_filename, with_seg=True)
            seg_list = []
            for i in range(data_volume.shape[2]):
                data_sample = self.data_provider.read_sample_from_volume(i, data_volume)

                seg_slice = sess.run([self.net.prediction], feed_dict={self.net.images: data_sample[np.newaxis, ...], self.net.keep_prob: 1.0})
                seg_slice = np.argmax(seg_slice, -1)
                seg_slice = np.reshape(seg_slice, (512, 512))

                # plt.figure(figsize=(15, 35))
                # plt.subplot(121)
                # plt.imshow(np.squeeze(new_seg_slice==1), 'gray')
                # plt.subplot(122)
                # plt.imshow(np.squeeze(gt_seg_volume[:, :, i]==2), 'gray')
                # plt.show()
                # plt.savefig('./imgs/'+str(volume_filename.split('/')[-1]+'_'+str(i))+'.png')
                # plt.close('all')
                seg_list.append(seg_slice)


            seg_volume = np.transpose(np.asarray(seg_list), (1, 2, 0))
            seg_volume = seg_volume.astype(np.uint8)

            seg_volume = self.single_connected_domain_filter(seg_volume, label=1)
            
            dice = metric.binary.dc(seg_volume>0, gt_seg_volume>0)
            logging.info('case: {}, len: {}, dice: {}'.format(case_id, data_volume.shape[2], dice))
            score_list.append(dice)

        logging.info('validation len: {}, liver dice per case: {}'.format(len(score_list), np.mean(score_list)))


    def eval_with_volumes(self):
        logging.info('Validation Start')
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            ckpt = tf.train.get_checkpoint_state(self.model_path)
            if ckpt and ckpt.model_checkpoint_path:
                self.restore(sess, ckpt.model_checkpoint_path)
            self.eval(sess, save_result=True)


    def test_with_volumes(self):
        logging.info('Inference Start')
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            ckpt = tf.train.get_checkpoint_state(self.model_path)
            if ckpt and ckpt.model_checkpoint_path:
                self.restore(sess, ckpt.model_checkpoint_path)

            test_volume_list = self.data_provider.get_test_volume_list()
            duration_list = []
            for volume_filename in test_volume_list:
                data_volume, affine, case_id = self.data_provider.read_volume(volume_filename, with_seg=False)
                seg_list = []
                for i in range(data_volume.shape[2]):
                    data_sample = self.data_provider.read_sample_from_volume(i, data_volume)

                    # data_sample = self.data_provider.normalize_image(data_sample)
                    seg_slice = sess.run([self.net.prediction], feed_dict={self.net.images: data_sample[np.newaxis, ...], self.net.keep_prob: 1.0})
                    seg_slice = np.argmax(seg_slice, -1)
                    seg_slice = np.reshape(seg_slice, (512, 512))
                    seg_list.append(seg_slice)
                seg_volume = np.transpose(np.asarray(seg_list), (1, 2, 0))
                seg_volume = seg_volume.astype(np.uint8)

                seg_volume = self.single_connected_domain_filter(seg_volume, label=1)

                nii_volume = nib.Nifti1Image(seg_volume, affine)
                nib.save(nii_volume, os.path.join(self.prediction_path, case_id+'.nii'))

                duration = time.time() - start_time
                duration_list.append(duration)
                logging.info('case: {}, len: {}, duration: {}'.format(case_id, data_volume.shape[2], duration))
            logging.info('mean duration: {}'.format(np.mean(duration_list)))
                
            
    def output_minibatch_stats(self, sess, summary_writer, epoch, step, batch_x, batch_y):
        # Calculate batch loss and accuracy
        summary_str, loss, ce, acc, prediction = sess.run([self.summary_op, 
                                                            self.net.loss, 
                                                            self.net.cross_entropy, 
                                                            self.net.accuracy, 
                                                            self.net.prediction], 
                                                           feed_dict={self.net.images: batch_x,
                                                                      self.net.labels: batch_y,
                                                                      self.net.keep_prob: 1.0})
        summary_writer.add_summary(summary_str, step)
        summary_writer.flush()
        error_rate = 100.0 - ( 100.0 * np.sum(np.argmax(prediction, 3) == batch_y) / (prediction.shape[0]*prediction.shape[1]*prediction.shape[2]))
        logging.info("Epoch {}, Iter {}/{}, Minibatch Loss= {:.4f}, Minibatch cross_entropy= {:.4f}, Training Accuracy= {:.4f}, Minibatch error= {:.1f}%".format(epoch, step%self.training_iters, self.training_iters, loss, ce, acc, error_rate))


    def save(self, sess, model_path):
        """
        Saves the current session to a checkpoint
        
        :param sess: current session
        :param model_path: path to file system location
        """
        
        saver = tf.train.Saver()
        save_path = saver.save(sess, model_path)
        logging.info("Model save at file: %s" % model_path)
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
