# -*- coding: utf-8 -*-
"""
Created on Thu Feb 24 16:59:04 2022

@author: user
"""


import numpy as np
import datetime as dt
import pandas as pd
import os
import random as r
import matplotlib as mpl
import matplotlib.pyplot as plt


DATASET_PATH = './ML_dataset_restart/GIMTEC'


class Dataset:
    def __init__(self, loc, startT, endT, dispD, refD):
        self.sT = dt.datetime(startT[0], startT[1], startT[2], 0, 0, 0)
        self.eT = dt.datetime(endT[0], endT[1], endT[2], 23, 0, 0)
        self.dD = dispD
        self.rD = refD
        self.filename = loc + \
            self.sT.strftime('%Y%m%d')+'_'+self.eT.strftime('%Y%m%d')

        if not os.path.exists(DATASET_PATH+'/'+self.filename):
            print('Please create a folder with earthquake information and TEC data')
            2/'abc'

    def load_data(self, lati, long):
        EQlist = np.load(DATASET_PATH+'/'+self.filename+'/EQlist.npy')
        tec = np.load(DATASET_PATH+'/'+self.filename +
                      '/tec'+str(lati)+'_'+str(long)+'.npy')
        tec = np.reshape(tec, (int(len(tec)/24), 24))  # (num_days, 24)
        return EQlist, tec

    def calculate_each_case(self, tec, k):
        pred = np.zeros((self.dD, 24, self.rD), dtype=float)  # (30, 24, 15)
        for j in range(self.rD):
            # (0~14, 24), (1~15, 24) (14~28, 24)
            # tec.shape = (num_days, 24)
            pred[:, :, j] = tec[j:j+np.size(tec, 0)-self.rD, :]

        [q1, q2, q3] = np.percentile(pred, [25, 50, 75], axis=2)
        obs = tec[self.rD:, :]
        ub = q2 + k*(q3 - q2)
        lb = q2 - k*(q2 - q1)
        # print("q1.shape = ", q1.shape)
        pos, neg = obs - ub, obs - lb
        pos[pos < 0] = 0
        pos[pos > 0] = 1
        neg[neg > 0] = 0
        neg[neg < 0] = 1
        return obs, q2, ub, lb, pos, neg

    def process_EQ_cases(self, lati, long, k):
        eqlist, tec0 = self.load_data(lati, long)

        self.allEQ = np.zeros((eqlist.shape[0], 10, self.dD, 24), dtype=float)
        for p, i in enumerate(eqlist):
            end_date = (dt.datetime(int(i[0]), int(i[1]), int(
                i[2]))-dt.timedelta(days=1)).toordinal()
            tec1 = tec0[end_date-self.sT.toordinal()-(self.dD+self.rD-1):end_date-self.sT.toordinal()+1, :]
            # tec1.shape = (45, 24)
            obs, q2, ub, lb, pos, neg = self.calculate_each_case(tec1, k)
            self.allEQ[p, 0, :, :] = obs
            self.allEQ[p, 1, :, :] = q2
            self.allEQ[p, 2, :, :] = ub
            self.allEQ[p, 3, :, :] = lb
            self.allEQ[p, 4, :, :] = pos
            self.allEQ[p, 5, :, :] = neg
            # Below feature are time which is year, month, day
            self.allEQ[p, 6, :, :] = int(i[0])
            self.allEQ[p, 7, :, :] = int(i[1])
            self.allEQ[p, 8, :, :] = int(i[2])
            self.allEQ[p, 9, :, :] = float(i[9])
        return self.allEQ
    # 0~65, threshold:1/3, [61:65, 0:]
    # PEIAs_day = [62, 66]
    # PEIAs_hour = [0, 23]

    def filter_case(self, lati, long, allEQ, polarity, PEIAsD, PEIAsH, PEIAsL):
        eqlist, _ = self.load_data(lati, long)
        new_eqlist = np.full_like(eqlist, np.nan)
        sum_hour = (PEIAsD[1]-PEIAsD[0]+1)*(PEIAsH[1]-PEIAsH[0]+1)
        removed_cases = []

        if polarity == 'Positive':
            for p, i in enumerate(eqlist):
                if np.sum(allEQ[p, 4, PEIAsD[0]-1:PEIAsD[1], PEIAsH[0]:PEIAsH[1]+1])/sum_hour >= PEIAsL:
                    new_eqlist[p] = i
                else:
                    removed_cases.append(p)

        elif polarity == 'Negative':
            for p, i in enumerate(eqlist):
                if np.sum(allEQ[p, 5, PEIAsD[0]-1:PEIAsD[1], PEIAsH[0]:PEIAsH[1]+1])/sum_hour >= PEIAsL:
                    new_eqlist[p] = i
                else:
                    removed_cases.append(p)

        allEQ = np.delete(allEQ, removed_cases, 0)
        return allEQ, new_eqlist

    def schedule_each_case_date(self, EQlist, chaos):
        zeroArray = np.zeros(
            ((self.eT-self.sT).days+1-(self.dD+self.rD), 2), dtype=float)
        dateIndex = pd.date_range(
            start=self.sT+dt.timedelta(days=self.dD+self.rD), end=self.eT)
        df = pd.DataFrame(data=zeroArray, index=dateIndex, columns=[
                          'Magnitude', 'Binary'], dtype=float)
        all_EQ_posi = np.zeros((EQlist.shape[0], 1), dtype=int)

        for p, i in enumerate(EQlist):
            t = dt.datetime(int(i[0]), int(i[1]), int(i[2]))
            posi = np.where(df.index[:] == t)
            # print(posi)
            # print(df.index[0:5])
            # print(posi[0][0])
            df['Magnitude'][posi[0][0]] = i[-1]
            df['Binary'][posi[0][0]] = 1
            all_EQ_posi[p] = posi[0][0]
            # print(all_EQ_posi)
        # print(all_EQ_posi.shape)

        for j in all_EQ_posi:
            df['Binary'][(j[0]-30):j[0]] = 1

        for j in all_EQ_posi:
            if df['Magnitude'][j][0] >= 6.0 and df['Magnitude'][j][0] < 6.5:
                df['Binary'][j[0]+1:j[0]+1+chaos[0]] = 1

            elif df['Magnitude'][j][0] >= 6.5 and df['Magnitude'][j][0] < 7.0:
                df['Binary'][j[0]+1:j[0]+1+chaos[1]] = 1

            elif df['Magnitude'][j][0] >= 7.0 and df['Magnitude'][j][0] < 7.5:
                df['Binary'][j[0]+1:j[0]+1+chaos[2]] = 1

            elif df['Magnitude'][j][0] >= 7.5:
                df['Binary'][j[0]+1:j[0]+1+chaos[3]] = 1
        return df

    def process_No_EQ_cases(self, lati, long, k, chaos, allEQ):
        eqlist, tec0 = self.load_data(lati, long)

        self.df = self.schedule_each_case_date(eqlist, chaos)
        all_No_EQ = self.df['Binary'][self.df['Binary'] != 1]
        print("確認：", all_No_EQ.shape[0])
        # awards = list(np.arange(0, all_No_EQ.shape[0], 1))
        # selected_days = r.sample(awards, allEQ.shape[0])
        # print(all_No_EQ.index)
        self.allNoEQ = np.zeros(
            (all_No_EQ.shape[0], 10, self.dD, 24), dtype=float)
        # for p, i in enumerate(all_No_EQ.index[selected_days]):
        for p, i in enumerate(all_No_EQ.index):
            # print("TESTEST", i.year, i.month, i.day)
            end_date = (i-dt.timedelta(days=1)).toordinal()
            tec1 = tec0[end_date-self.sT.toordinal()-(self.dD+self.rD-1)                        :end_date-self.sT.toordinal()+1, :]
            obs, q2, ub, lb, pos, neg = self.calculate_each_case(tec1, k)
            self.allNoEQ[p, 0, :, :] = obs
            self.allNoEQ[p, 1, :, :] = q2
            self.allNoEQ[p, 2, :, :] = ub
            self.allNoEQ[p, 3, :, :] = lb
            self.allNoEQ[p, 4, :, :] = pos
            self.allNoEQ[p, 5, :, :] = neg
            # Below feature are time which is year, month, day
            self.allNoEQ[p, 6, :, :] = int(i.year)
            self.allNoEQ[p, 7, :, :] = int(i.month)
            self.allNoEQ[p, 8, :, :] = int(i.day)
            self.allNoEQ[p, 9, :, :] = 99999
        return self.allNoEQ


class SaveDataset:
    def __init__(self, data1, data2, loc, startT, endT, dispD, refD):
        self.data_1 = data1
        self.data_2 = data2
        self.samples = data1.shape[0]+data2.shape[0]
        self.fileName = Dataset(loc, startT, endT, dispD, refD).filename

    def pack_data(self):
        self.dataset_images = np.concatenate(
            (self.data_1, self.data_2), axis=0)
        self.dataset_labels = np.zeros((self.samples, 1), dtype=float)
        self.dataset_labels[0:self.data_1.shape[0], :] = 1
        return self.dataset_images, self.dataset_labels

    def save_them(self):
        if not os.path.exists(DATASET_PATH+'/'+self.fileName+'/test_img.npy'):
            np.save(DATASET_PATH+'/'+self.fileName +
                    '/test_img.npy', self.dataset_images)
        else:
            print('Already have the image dataset')

        if not os.path.exists(DATASET_PATH+'/'+self.fileName+'/test_labels.npy'):
            np.save(DATASET_PATH+'/'+self.fileName +
                    '/test_labels.npy', self.dataset_labels)
        else:
            print('Already have the label dataset')


def operate():
    location = 'Japan'
    latitude = 36.5                           # monitoring point
    longitude = 142

    startTime = [1999, 1, 1]                  # dataset time
    endTime = [2021, 12, 31]

    dispDays = 66                             # display days
    refDays = 15                              # reference days
    k = 1.5                                   # threshold

    polarity = 'Positive'                     # Positive or Negative
    PEIAs_day = [62, 66]
    PEIAs_hour = [0, 23]                      # data default time zone: UT
    PEIAs_limit = 1/3                         # threshold

    # prohibited days after the earthquake [6.0-6.5, 6.5-7.0, 7.0-7.5, 7.5-]
    chaosPeriod = [5, 7, 10, 15]

    d = Dataset(location, startTime, endTime, dispDays, refDays)
    allEQ = d.process_EQ_cases(latitude, longitude, k)
    key = input('Do you want to filter the cases (y/[n])? ')
    if key == 'y':
        allEQ, new_eqlist = d.filter_case(
            latitude, longitude, allEQ, polarity, PEIAs_day, PEIAs_hour, PEIAs_limit)
    else:
        print('Nothing done')
    allNoEQ = d.process_No_EQ_cases(
        latitude, longitude, k, chaosPeriod, allEQ)  # 有調整 allNoEQ 的數目
    print(allEQ.shape)
    print(allNoEQ.shape)
    s = SaveDataset(allEQ, allNoEQ, location, startTime,
                    endTime, dispDays, refDays)
    dataset_images, dataset_labels = s.pack_data()
    s.save_them()


if __name__ == '__main__':
    operate()
