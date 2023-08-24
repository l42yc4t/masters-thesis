# -*- coding: utf-8 -*-
"""
Created on Sat Jun 4 16:08:12 2022

@author: user
"""


import numpy as np
import datetime as dt
import os
from tqdm import tqdm


DATA_NAME = 'GIM'
DATA_PATH = 'I:/IRSL/CODEGIM_py'
DATASET_PATH = 'I:/IRSL/ConvLSTM_dataset/'+DATA_NAME


class GIMTEC():
    def __init__(self, time_resolution, data_size, start_time, end_time):
        self.time_resolution = time_resolution
        self.row, self.col = data_size[0], data_size[1]
        self.start_time = dt.datetime(start_time[0],start_time[1],start_time[2],0,0,0)
        self.end_time = dt.datetime(end_time[0],end_time[1],end_time[2],23,0,0)
        self.total_days = self.end_time.toordinal()-self.start_time.toordinal()+1

    def load_all_data(self):
        self.gimTEC = np.zeros(
            (self.total_days*self.time_resolution, self.row, self.col), 
            dtype = np.float32)

        for i in tqdm(range(self.total_days), desc='Download'):
            year = (self.start_time + dt.timedelta(days=i)).year
            doy = (self.start_time + dt.timedelta(days=i)).timetuple().tm_yday
            gim = np.load(f'{DATA_PATH}/GIMTEC{year}/gim{year}{doy:03}.npz')

            gimtec = gim['gim'][:self.time_resolution*self.row, :].reshape(self.time_resolution, self.row, self.col)  # shape=(24,71,73)

            if np.any(gimtec == 999.9):
                gimtec[gimtec == 999.9] = 0

            self.gimTEC[i*self.time_resolution:(i+1)*self.time_resolution, :, :] = gimtec/10  # shape=(totaldays*24,71,73)

    def expand_dim(self):
        self.gimTEC = self.gimTEC[:, :, :, np.newaxis]

    def download_raw_data(self):
        folder_name = 'gim'+self.start_time.strftime('%Y%m%d')+'_'+self.end_time.strftime('%Y%m%d')
        if not os.path.exists(f'{DATASET_PATH}/{folder_name}/gimTEC.npy'):
            self.load_all_data()
            self.expand_dim() # gimTEC.shape=(totaldays*24,71,73, 1)
            np.save(f'{DATASET_PATH}/{folder_name}/gimTEC.npy', self.gimTEC)
        else:
            print('Already have the raw data')
            self.gimTEC = np.load(f'{DATASET_PATH}/{folder_name}/gimTEC.npy')
        return self.gimTEC, folder_name


def operate():
    time_resolution = 24               # hrs
    data_size = [71, 73]               # map size
    data_start_time = [2019, 1, 1]
    data_end_time = [2021, 12, 31]

    g = GIMTEC(time_resolution, data_size, data_start_time, data_end_time)
    gimTEC, folder_name = g.download_raw_data()


if __name__ == '__main__':
    operate()

