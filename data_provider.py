from __future__ import division, print_function
import numpy as np
import nibabel as nib
import datetime
import random
import glob
import os

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

class DataProvider(object):
    """
    DataProvider is the data interface for tensorflow model training, validation and testing

    functions:
    get_train_sample_list
    get_val_sample_list
    get_random_batch
    read_sample
    normalize_image
    random_2d_rotation
    """
    def __init__(self, conf):
        """
        :param slice_range: the number of consecutive slices to sample
        :param background_keep_prob: the probability of background slices involved for training, validation or testing
        :param data_slice_dir: the dir of all slices, where the slices are stored as '$data_slice_dir/$case_id/($case_id)-(slice_id).npy'
        For example, 'data/liver/data/K10/K10_0.npy', the data_slice_dir is "data/liver/data/"
        :param seg_slice_dir: the dir of all segmentations of slices, where the segmentations are stored as '$seg_slice_dir/$case_id/(case_id)-(slice_id).npy'
        For example, 'data/liver/segs/K10/K10_0.npy', the seg_slice_dir is "data/liver/segs/"
        :param test_volume_dir: the dir of all volumes for testing
        :param train_case_id_list: the list of case id for training
        :param train_slice_id_list: the list of slice id for training
        :param val_case_id_list: the list of case id for validation
        :param val_slice_id_list: the list of slice id for validation
        :param test_case_id_list: the list of case id for testing
        """
        np.random.seed(datetime.datetime.now().second)
        self.HU_range = conf.HU_range
        self.slice_range = conf.slice_range
        self.background_keep_prob = conf.background_keep_prob

        # data dir
        self.data_slice_dir = conf.data_slice_dir
        self.seg_slice_dir = conf.seg_slice_dir
        self.data_volume_dir = conf.data_volume_dir
        self.seg_volume_dir = conf.seg_volume_dir

        # train list
        self.train_case_id_list = conf.train_case_id_list
        if len(self.train_case_id_list) < 1 and os.path.exists(self.data_slice_dir):
            self.train_case_id_list = os.listdir(self.data_slice_dir)
        self.train_slice_id_list = conf.train_slice_id_list
        self.train_sample_list = self.get_train_sample_list()
        logging.info('########### train list ##############')
        logging.info('train_case_id_list: {}'.format(self.train_case_id_list))
        logging.info('train_case_id_list length: {}'.format(len(self.train_case_id_list)))
        # logging.info('train_slice_id_list length: {}'.format(len(self.train_slice_id_list)))
        logging.info('train_sample_list length: {}'.format(len(self.train_sample_list)))

        # val list
        self.val_case_id_list = conf.val_case_id_list
        logging.info('########### val list ##############')
        logging.info('val_case_id_list: {}'.format(self.val_case_id_list))
        logging.info('val_case_id_list length: {}'.format(len(self.val_case_id_list)))

        # test list
        self.test_case_id_list = conf.test_case_id_list
        logging.info('########### test list ##############')
        logging.info('test_case_id_list: {}'.format(self.test_case_id_list))
        logging.info('test_case_id_list length: {}'.format(len(self.test_case_id_list)))
        logging.info('####################################')

    def get_val_volume_list(self):
        val_volume_list = self.get_volume_list(self.val_case_id_list, self.data_volume_dir)

        logging.info('########### val list ##############')
        logging.info('val_volume_list: {}'.format(val_volume_list))
        logging.info('val_volume_list length: {}'.format(len(val_volume_list)))
        logging.info('####################################')

        return val_volume_list

    def get_test_volume_list(self):
        test_volume_list = self.get_volume_list(self.test_case_id_list, self.data_volume_dir)

        logging.info('########### test list ##############')
        logging.info('test_volume_list: {}'.format(test_volume_list))
        logging.info('test_volume_list length: {}'.format(len(test_volume_list)))
        logging.info('####################################')

        return test_volume_list

    def get_volume_list(self, case_id_list, volume_dir):
        volume_list = glob.glob(os.path.join(volume_dir, '*.nii'))

        # case id filter
        if len(case_id_list) > 0:
            volume_list = [x for x in volume_list if self.get_case_id_from_volume_path(x) in case_id_list]

        volume_list.sort()
        return volume_list

    @staticmethod
    def get_case_id_from_volume_path(volume_path):
        """
        :param volume_path: the full path of a volume
        :return: the case id of the volume
        """
        return volume_path.split('/')[-1][:-4]

    @staticmethod
    def get_case_id_from_slice_path(slice_fullpath):
        """
        :param slice_fullpath: the full path of a slice
        :return: the case id of the slice
        """
        return slice_fullpath.split('/')[-1][:-4].split('-')[0]

    @staticmethod
    def get_slice_id_from_slice_path(slice_fullpath):
        """
        :param slice_fullpath: the full path of a slice
        :return: the slice id of the slice
        """
        return slice_fullpath.split('/')[-1][:-4].split('-')[-1]

    def _make_a_sample(self, index, data_slice_list):
        """
        :param index: the index of a slice in data_slice_list
        :param data_slice_list: the list of the full paths of slice
        :return: a sample consists of the full paths of consecutive slices
        If the range is out of the index of the list, the first or the last slice will be copied
        """
        sample = []
        if index < 0 or index > len(data_slice_list)-1:
            raise Exception('index out of range [{}, {}]'.format(0, len(data_slice_list)-1))
        for i in range(index-self.slice_range//2, index+self.slice_range//2+1):
            sample.append(data_slice_list[np.clip(i, 0, len(data_slice_list)-1)])
        return sample

    def _get_sample_list(self, case_id_list=[], slice_id_list=[], background_keep_prob=1.0):
        """
        :param case_id_list: the list of case id for training, validation or testing
        if it is empty, all cases will be involved
        :param slice_id_list: the list of slice id for training, validation or testing
        if it is empty, all slices will be involved
        :param background_keep_prob: the probability of background slices involved for training, validation or testing
        :return: the sample list for training, validation or testing
        """
        sample_list = []
        data_slice_list = glob.glob(os.path.join(self.data_slice_dir, '*/*.npy')) #'myData'
        for case_id in case_id_list:
            # filter the list by case_id_list
            case_data_slice_list = [i for i in data_slice_list if self.get_case_id_from_slice_path(i) == case_id]
            case_data_slice_list.sort(key=lambda x: int(self.get_slice_id_from_slice_path(x)))
            sample_list += self._make_sample_list_of_a_case(case_data_slice_list, slice_id_list, background_keep_prob)
        return sample_list 

    def _make_sample_list_of_a_case(self, case_data_slice_list, slice_id_list=[], background_keep_prob=1.0):
        """
        :param case_data_slice_list: the list of the full paths of slices of a case
        :param slice_id_list: the list of slice id
        :param background_keep_prob: the probability of background slices involved for training, validation or testing
        :return: the sample list of a case for training, validation or testing
        """
        case_sample_list = []
        for i in range(len(case_data_slice_list)):
            # filter the list by slice_id_list
            if (len(slice_id_list) >= 1 and self.get_slice_id_from_slice_path(case_data_slice_list[i]) in slice_id_list) or len(slice_id_list) < 1:
                # filter the list by background_keep_prob
                seg_slice = np.load(self.get_seg_slice_from_data(case_data_slice_list[i], self.seg_slice_dir))
                if np.max(seg_slice) > 0 and np.sum(seg_slice>0) > 64:
                    case_sample_list.append(self._make_a_sample(i, case_data_slice_list))
                elif np.random.random() < background_keep_prob:
                    case_sample_list.append(self._make_a_sample(i, case_data_slice_list))
        return case_sample_list

    # def _make_a_data_seg_pair(self, data_slice_fullpath):
    #     seg_slice_fullpath = self._get_seg_slice_fullpath(data_slice_fullpath)
    #     data_seg_pair = data_slice_fullpath + ' ' + seg_slice_fullpath
    #     return data_seg_pair


    def get_seg_slice_from_data(self, data_fullpath, seg_slice_dir):
        data_filename = data_fullpath.split('/')[-1]
        slice_seg_fullpath = os.path.join(os.path.join(seg_slice_dir, self.get_case_id_from_slice_path(data_fullpath)), data_filename)
        return slice_seg_fullpath

    def get_seg_volume_from_data(self, data_fullpath, volume_seg_dir):
        data_filename = data_fullpath.split('/')[-1]
        volume_seg_fullpath = os.path.join(volume_seg_dir, data_filename)
        return volume_seg_fullpath

    def get_train_sample_list(self):
        """
        :return: the sample list for training
        """
        return self._get_sample_list(self.train_case_id_list, self.train_slice_id_list, self.background_keep_prob)

    def get_val_sample_list(self):
        """
        :return: the sample list for validation
        """
        return self._get_sample_list(self.val_case_id_list, self.val_slice_id_list, background_keep_prob=1.0)

    def get_random_batch(self, batch_size):
        """
        :param batch_size: the batch size for training
        :return: the data for training in the specified batch size
        """
        if batch_size < 0:
            raise Exception('batch size should be positive')
        batch = random.sample(self.train_sample_list, batch_size)
        data_batch = []
        seg_batch = []
        for sample in batch:
            data, seg, _, __ = self.read_sample(sample, with_seg=True)
            data_batch.append(data)
            seg_batch.append(seg)
        return np.asarray(data_batch, dtype=np.float32), np.asarray(seg_batch, dtype=np.uint8)

    def read_sample(self, sample, with_seg=False):
        """
        Given a list of npy file paths (i.e. a sample), convert to a numpy tensor [H, W, SliceId]
        The npy file name should be in the format of \caseId_sliceId.npy
        :param sample: a list of npy file paths. Each of the element in the list is a slice npy file.
        :param with_seg: whether to read corresponding segmentation npy file to return in seg_sample
        :return:
        data_sample: the sample in numpy array
        seg_sample: the corresponding segmentation in numpy array (only valid when with_seg=True)
        case_id: the case id of the sample
        slice_id: the slice id of the sample
        """
        if len(sample) < 0:
            raise Exception('input sample is empty!')
        data_sample = []
        case_id = self.get_case_id_from_slice_path(sample[self.slice_range//2])
        slice_id = self.get_slice_id_from_slice_path(sample[self.slice_range//2])
        for data_fullpath in sample:
            data_slice = np.clip(np.load(data_fullpath), self.HU_range[0], self.HU_range[1])
            data_sample.append(data_slice)
        data_sample = np.transpose(np.asarray(data_sample, dtype=np.float32), (1, 2, 0))

        if with_seg:
            seg_sample = np.load(self.get_seg_slice_from_data(sample[self.slice_range//2], self.seg_slice_dir))
            return data_sample, seg_sample, case_id, slice_id
        else:
            return data_sample, case_id, slice_id

    def read_sample_from_volume(self, index, volume):
        """
        read consecutive slices from a given volume centered at the index
        If the index is at the beginning of ending of the volume, this method will duplicate the starting and ending slices

        :param index: the index of the slice in a volume
        :param volume: the input volume data in numpy array format
        :return: the consecutive slices with slice range of a volume in numpy array
        """
        data_sample = []
        for i in range(index-self.slice_range//2, index+self.slice_range//2+1):
            data_sample.append(volume[:, :, np.clip(i, 0, volume.shape[2]-1)])
        data_sample = np.transpose(np.asarray(data_sample, dtype=np.float32), (1, 2, 0))
        return data_sample

    def read_volume(self, nii_volume_path, with_seg=True):
        """
        Read one nii volume with name format of case_id.nii
        :param volume_path: the full path of a volume
        :return: the volume data, affine matrix, case_id
        """
        nii_volume = nib.load(nii_volume_path)
        data_volume = np.clip(nii_volume.get_data(), self.HU_range[0], self.HU_range[1])
        affine = nii_volume.get_affine()
        case_id = nii_volume_path.split('/')[-1][:-4]
        if with_seg:
            seg_volume = nib.load(self.get_seg_volume_from_data(nii_volume_path, self.seg_volume_dir)).get_data()
            return data_volume, seg_volume, affine, case_id
        else:
            return data_volume, affine, case_id

    @staticmethod
    def normalize_image(tensor):
        """
        normalize a tensor between [0, 1]
        :param tensor: a tensor in an arbitrary shape 
        """
        max_value = np.max(tensor)
        min_value = np.min(tensor)
        return (tensor - min_value) / (max_value - min_value)

    @staticmethod
    def random_2d_rotation(batch_x, batch_y):
        """
        random rotate the 2d images and labels in the training batch 
        :param batch_x: image batch in shape [batch_size, height, width, channels] 
        :param batch_y: image batch in shape [batch_size, height, width] 
        """
        for i in range(batch_x.shape[0]):
            rot_degree = np.random.randint(4)
            for j in range(batch_x.shape[3]):
                batch_x[i, :, :, j] = np.rot90(batch_x[i, :, :, j], rot_degree)
            batch_y[i, :, :] = np.rot90(batch_y[i, :, :], rot_degree)
        return batch_x, batch_y

