# coding: utf-8
# @email: enoche.chow@gmail.com
#
# updated: Mar. 25, 2022
# Filled non-existing raw features with non-zero after encoded from encoders

"""
Data pre-processing
##########################
"""
from logging import getLogger
from collections import Counter
import os
import pandas as pd
import numpy as np
import torch
from utils.data_utils import (ImageResize, ImagePad, image_to_tensor, load_decompress_img_from_lmdb_value)
import lmdb


class RecDataset(object):
    def __init__(self, config, df=None, negativeSamples=None):
        self.config = config
        self.logger = getLogger()

        # data path & files
        self.dataset_name = config['dataset']
        self.dataset_path = os.path.abspath(config['data_path']+self.dataset_name)

        # dataframe
        self.uid_field = self.config['USER_ID_FIELD']
        self.iid_field = self.config['ITEM_ID_FIELD']
        self.splitting_label = self.config['inter_splitting_label']
        self.negative_samples  = self.config['negative_sample']
        self.user_fixed_negative_sampling = self.config['user_fixed_negative_sampling']
        self.negativeSamples = {}
        if df is not None:
            self.df = df
            self.negativeSamples = negativeSamples
            return
        # if all files exists
        check_file_list = [self.config['inter_file_name']]
        for i in check_file_list:
            file_path = os.path.join(self.dataset_path, i)
            if not os.path.isfile(file_path):
                raise ValueError('File {} not exist'.format(file_path))
        if self.user_fixed_negative_sampling:
            filePath = os.path.join(self.dataset_path, self.negative_samples)
            if not os.path.isfile(filePath):
                raise ValueError('File {} not exist'.format(filePath))
            self.negativeSamples = self.parse_txt_file_to_dict(filePath)
        
        if self.config['use_400_samples']:
            filePath = os.path.join(self.dataset_path, self.config['samples_400_file'])
            if not os.path.isfile(filePath):
                raise ValueError('File {} not exist'.format(filePath))
            self.samples400UserID = self.parse_txt_file_to_list(filePath)
        
        # load rating file from data path?
        self.load_inter_graph(config['inter_file_name'])
        self.item_num = int(max(self.df[self.iid_field].values)) + 1
        self.user_num = int(max(self.df[self.uid_field].values)) + 1

    def parse_txt_file_to_list(self, file_path):
        userIDs = []
        with open(file_path, 'r') as file:
            for line in file:
                userIDs.append(int(line.strip()))
        return userIDs

    def parse_txt_file_to_dict(self, file_path):
        data_dict = {}
        with open(file_path, 'r') as file:
            for line in file:
                # 将每行通过空格分割，得到一个列表
                numbers = list(map(int, line.strip().split()))
                numbers = (np.array(numbers)-1).tolist()
                if numbers:
                    # 使用第一个数字作为key，剩余部分作为value（列表）
                    data_dict[numbers[0]] = numbers[1:]
        return data_dict
    
    def load_inter_graph(self, file_name):
        inter_file = os.path.join(self.dataset_path, file_name)
        cols = [self.uid_field, self.iid_field, self.splitting_label]
        self.df = pd.read_csv(inter_file, usecols=cols, sep=self.config['field_separator'])
        print(f"debugging Recdataset self.df: ")
        print(self.df.head())
        if not self.df.columns.isin(cols).all():
            raise ValueError('File {} lost some required columns.'.format(inter_file))

    def split(self):
        dfs = []
        # splitting into training/validation/test
        for i in range(3):
            temp_df = self.df[self.df[self.splitting_label] == i].copy()
            temp_df.drop(self.splitting_label, inplace=True, axis=1)        # no use again
            if i == 2 and self.config['use_400_samples']:
                temp_df = temp_df[temp_df[self.uid_field].isin(self.samples400UserID)]
                print(f"debugging 400 samples temp_df : {temp_df}")
            dfs.append(temp_df)
            
        if self.config['filter_out_cod_start_users']:
            # filtering out new users in val/test sets
            train_u = set(dfs[0][self.uid_field].values)
            for i in [1, 2]:
                dropped_inter = pd.Series(True, index=dfs[i].index)
                dropped_inter ^= dfs[i][self.uid_field].isin(train_u)
                dfs[i].drop(dfs[i].index[dropped_inter], inplace=True)

        # wrap as RecDataset
        full_ds = [self.copy(df, self.negativeSamples.copy()) for df in dfs]
        return full_ds

    def copy(self, new_df, negativeSamples):
        """Given a new interaction feature, return a new :class:`Dataset` object,
                whose interaction feature is updated with ``new_df``, and all the other attributes the same.

                Args:
                    new_df (pandas.DataFrame): The new interaction feature need to be updated.

                Returns:
                    :class:`~Dataset`: the new :class:`~Dataset` object, whose interaction feature has been updated.
                """
        nxt = RecDataset(self.config, new_df, negativeSamples)

        nxt.item_num = self.item_num
        nxt.user_num = self.user_num
        return nxt

    def get_user_num(self):
        return self.user_num

    def get_item_num(self):
        return self.item_num

    def shuffle(self):
        """Shuffle the interaction records inplace.
        """
        self.df = self.df.sample(frac=1, replace=False).reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Series result
        return self.df.iloc[idx]

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        info = [self.dataset_name]
        self.inter_num = len(self.df)
        uni_u = pd.unique(self.df[self.uid_field])
        uni_i = pd.unique(self.df[self.iid_field])
        tmp_user_num, tmp_item_num = 0, 0
        if self.uid_field:
            tmp_user_num = len(uni_u)
            avg_actions_of_users = self.inter_num/tmp_user_num
            info.extend(['The number of users: {}'.format(tmp_user_num),
                         'Average actions of users: {}'.format(avg_actions_of_users)])
        if self.iid_field:
            tmp_item_num = len(uni_i)
            avg_actions_of_items = self.inter_num/tmp_item_num
            info.extend(['The number of items: {}'.format(tmp_item_num),
                         'Average actions of items: {}'.format(avg_actions_of_items)])
        info.append('The number of inters: {}'.format(self.inter_num))
        if self.uid_field and self.iid_field:
            sparsity = 1 - self.inter_num / tmp_user_num / tmp_item_num
            info.append('The sparsity of the dataset: {}%'.format(sparsity * 100))
        return '\n'.join(info)
