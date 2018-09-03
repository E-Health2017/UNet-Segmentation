#!/usr/bin/env python
# coding=utf-8
import numpy as np
import datetime
import random
import os
from unet import *
from data_provider import *
from operationer import *
np.random.seed(datetime.datetime.now().second)

class Configuration(object):
    def __init__(self):
        gpu_no = '0'
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_no)

        # model conf
        self.channels = 3
        self.class_num = 2
        self.class_weights=[1, 3]
        self.cost = 'cross_entropy'
        self.root_features = 64
        self.layer_num = 5
        self.filter_size = 3
        self.pool_size = 2

        # data_provider conf
        self.HU_range=[-75, 175]
        self.slice_range = self.channels
        self.background_keep_prob = 0.1
        self.data_slice_dir = '/data/zhangyao/data/raw_images/npy/slice/'
        self.seg_slice_dir = '/data/zhangyao/data/combine/npy/slice/'
        self.data_volume_dir = '/data/zhangyao/data/raw_images/nii/'
        self.seg_volume_dir = '/data/zhangyao/data/combine/nii/'

        self.val_case_id_list = [x for x in os.listdir('/data/zhangyao/data/raw_images/npy/slice') if x.startswith('S')]
        self.train_case_id_list = [x for x in os.listdir('/data/zhangyao/data/raw_images/npy/slice') if x not in self.val_case_id_list]
        self.test_case_id_list = []
        self.train_slice_id_list = []

        # operationer conf
        self.log_path = './log/'
        self.model_path = './trained/'
        self.prediction_path = './prediction/'
        self.need_restore = True
        self.learning_rate = 0.0001
        self.batch_size = 1
        self.epochs = 50
        self.display_step = 200
        self.optimizer = 'adam'
        self.decay_rate = 0.5
        self.momentum = 0.9
    

if __name__ == '__main__':
    conf = Configuration()
    
    net = Unet(conf)
    dp = DataProvider(conf)
    op = Operationer(net=net, data_provider=dp, conf=conf)
    
    op.train()
    # op.test_with_volumes()
    # op.eval_with_volumes()

