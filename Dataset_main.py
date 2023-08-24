# -*- coding: utf-8 -*-
"""
Created on Sat Jun 4 16:25:41 2022

@author: user
"""


import numpy as np
import os
from GIMTEC_data import GIMTEC
from sklearn.model_selection import train_test_split
from tqdm import tqdm


DATA_NAME = 'GIM'
DATASET_PATH = 'I:/IRSL/ConvLSTM_dataset/'+DATA_NAME
NORMALIZE_CONSTANT = 80


class DataPreprocessing():
    def __init__(self, time_resolution, data_size, data_start_time, data_end_time, 
        input_time_steps, output_time_steps, channels):
        # Data format: channels last
        g = GIMTEC(time_resolution, data_size, data_start_time, data_end_time)
        raw_data, folder_name = g.download_raw_data() # raw_data.shape=(totaldays*24,71,73, 1)
        self.raw_data = raw_data
        self.folder_name = folder_name
        self.input_time_steps = input_time_steps
        self.output_time_steps = output_time_steps
        self.num_samples = raw_data.shape[0]-self.input_time_steps+1-self.output_time_steps # totaldays*24 - 8 + 1 - 1
        self.X = np.zeros(
            (self.num_samples, self.input_time_steps, raw_data.shape[1], raw_data.shape[2], channels), 
            dtype = np.float32) # X.shape=(num_samples, 8(time_steps), 71, 73, 1)
        
    def generate_X_y_from_raw_data(self):
        for i in tqdm(range(self.num_samples), desc='Arrange'):
            self.X[i, :, :, :, :] = self.raw_data[i:i+self.input_time_steps, :, :, :]

        pred_y = self.raw_data[self.input_time_steps:, :, :, :] # pred_y.shape=(num_samples,71,73,1)
        y = np.zeros_like(self.X) # y.shape=(num_samples, 8(time_steps), 71, 73, 1)
        for i in range(len(y)): # num_samples
            for j in range(self.output_time_steps): # output_time_steps=1
                y[i, j] = pred_y[i+j]
        self.y = y/NORMALIZE_CONSTANT

    def split_into_train_test(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42)

    def read_dataset(self):
        if not os.path.exists(f'{DATASET_PATH}/{self.folder_name}/dataset_output_time_steps_{self.output_time_steps}.npz'):
            self.generate_X_y_from_raw_data()
            self.split_into_train_test()
            np.savez(f'{DATASET_PATH}/{self.folder_name}/dataset_output_time_steps_{self.output_time_steps}.npz', 
                X_train=self.X_train, X_test=self.X_test, y_train=self.y_train, y_test=self.y_test)
        else:
            print('Already have the dataset')
            dataset = np.load(f'{DATASET_PATH}/{self.folder_name}/dataset_output_time_steps_{self.output_time_steps}.npz')
            self.X_train = dataset['X_train']
            self.X_test = dataset['X_test']
            self.y_train = dataset['y_train']
            self.y_test = dataset['y_test']
        return self.X_train, self.X_test, self.y_train, self.y_test


def read_train_test_dataset(time_resolution, data_size, data_start_time, data_end_time, input_time_steps, output_time_steps, channels):
    dp = DataPreprocessing(
            time_resolution, 
            data_size, 
            data_start_time, 
            data_end_time, 
            input_time_steps, 
            output_time_steps, 
            channels
        )
    train_X, test_X, train_y, test_y = dp.read_dataset()
    return train_X, test_X, train_y, test_y

if __name__ == '__main__':
    read_train_test_dataset()

