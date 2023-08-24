# -*- coding: utf-8 -*-
"""
Created on Tue Feb 22 17:00:14 2022

@author: user
"""


import numpy as np
import datetime as dt
import os
import pandas as pd
import USGS_EQlist as usgs
import Remove_aftershock as rmaf


DATASET_PATH = './ML_dataset_restart/GIMTEC'
GIM_PATH = './CODEGIM_py'


class EarthquakeList:
    def __init__(self, loc, startT, endT):
        self.startT = startT
        self.endT = endT
        self.sT = dt.datetime(
            self.startT[0], self.startT[1], self.startT[2], 0, 0, 0)
        self.eT = dt.datetime(
            self.endT[0], self.endT[1], self.endT[2], 23, 0, 0)
        self.filename = loc + \
            self.sT.strftime('%Y%m%d')+'_'+self.eT.strftime('%Y%m%d')

        if not os.path.exists(DATASET_PATH+'/'+self.filename):
            os.makedirs(DATASET_PATH+'/'+self.filename)

    def load_USGS_earthquake(self, minLa, maxLa, minLo, maxLo, minMag):
        usgs_title, usgs_list = usgs.usgs_EQlist(
            self.startT, self.endT, minLa, maxLa, minLo, maxLo, minMag)
        return usgs_title, usgs_list

    def remove_case_with_Ref_days(self, dispD, refD, title, list):
        del_posi = []
        for p, i in enumerate(list):
            if dt.datetime(int(i[0]), int(i[1]), int(i[2]))-dt.timedelta(days=dispD+refD+1) < self.sT:
                del_posi.append(p)

        usgs_title_1 = np.delete(title, del_posi, 0)
        usgs_list_1 = np.delete(list, del_posi, 0)
        return usgs_title_1, usgs_list_1

    def delete_aftershock(self, title, list):
        EQlist, EQtitle = rmaf.remove_aftershock(list, title)
        return EQlist, EQtitle

    def get_earthquake(self, title, list):
        if not os.path.exists(DATASET_PATH+'/'+self.filename+'/EQlist_mag6.npy'):
            EQlist, EQtitle = self.delete_aftershock(title, list)
            np.save(DATASET_PATH+'/'+self.filename+'/EQlist_mag6.npy', EQlist)
            np.save(DATASET_PATH+'/'+self.filename +
                    '/EQtitle_mag6.npy', EQtitle)
        else:
            print('Already have the earthquake list')
            np.load(DATASET_PATH+'/'+self.filename+'/EQtitle_mag6.npy')
            np.load(DATASET_PATH+'/'+self.filename+'/EQlist_mag6.npy')
        return EQtitle, EQlist


class TimeSeriesTEC:
    def __init__(self, loc, startT, endT, lati, long):
        self.sT = dt.datetime(startT[0], startT[1], startT[2], 0, 0, 0)
        self.eT = dt.datetime(endT[0], endT[1], endT[2], 23, 0, 0)
        self.filename = loc + \
            self.sT.strftime('%Y%m%d')+'_'+self.eT.strftime('%Y%m%d')

        self.lat = round((87.5-lati)/2.5)
        self.lon = round((long+180)/5)
        dateIndex = pd.date_range(start=self.sT.strftime(
            "%Y/%m/%d %H:%M:%S"), end=self.eT.strftime("%Y/%m/%d %H:%M:%S"), freq='H')
        self.df = pd.DataFrame(index=dateIndex, columns=[
                               'tec'], dtype=np.float64)

        if not os.path.exists(DATASET_PATH+'/'+self.filename):
            os.makedirs(DATASET_PATH+'/'+self.filename)

    def process_monitoring_point(self):
        for i in range(self.eT.toordinal() - self.sT.toordinal() + 1):
            year = (self.sT+dt.timedelta(days=i)).year
            doy = (self.sT+dt.timedelta(days=i)).timetuple().tm_yday
            gim = np.load(GIM_PATH+'/GIMTEC'+str(year)+'/gim' +
                          str(year)+str('%03d' % doy)+'.npz')
            gimTEC, gimTime = gim['gim'], gim['gim_time']

            for z, j in enumerate(gimTime):
                t = dt.datetime(j[0], j[1], j[2], j[3], 0,
                                0).strftime("%Y-%m-%d %H:%M:%S")
                self.df['tec'][t] = gimTEC[self.lat+z*71, self.lon]/10

    def interpolate_TEC(self):
        self.all_TEC = self.df.interpolate(
            method='time', limit_direction='both')

    def get_TEC(self, lati, long):
        if not os.path.exists(DATASET_PATH+'/'+self.filename+'/tec'+str(lati)+'_'+str(long)+'.npy'):
            self.process_monitoring_point()
            self.interpolate_TEC()
            np.save(DATASET_PATH+'/'+self.filename+'/tec' +
                    str(lati)+'_'+str(long)+'.npy', self.all_TEC)
        else:
            print('Already have the time series TEC')
            np.load(DATASET_PATH+'/'+self.filename +
                    '/tec'+str(lati)+'_'+str(long)+'.npy')
        return self.all_TEC


def operate():
    location = 'Japan'
    latitude = 36.5                # monitoring point
    longitude = 142

    startTime = [1999, 1, 1]       # dataset time
    endTime = [2021, 12, 31]

    minLat, maxLat = 30, 45        # earthquakes zone
    minLon, maxLon = 135, 145
    min_magnitude = 6.0            # minimum magnitude

    dispDays = 66                  # display days
    refDays = 15                   # reference days

    e = EarthquakeList(location, startTime, endTime)
    usgs_title, usgs_list = e.load_USGS_earthquake(
        minLat, maxLat, minLon, maxLon, min_magnitude)
    usgs_title_1, usgs_list_1 = e.remove_case_with_Ref_days(
        dispDays, refDays, usgs_title, usgs_list)
    EQtitle, EQlist = e.get_earthquake(usgs_title_1, usgs_list_1)

    t = TimeSeriesTEC(location, startTime, endTime, latitude, longitude)
    tec = t.get_TEC(latitude, longitude)


if __name__ == '__main__':
    operate()
