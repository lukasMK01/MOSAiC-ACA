# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 18:00:01 2021

@author: Lukas Monrad-Krohn
"""

import pandas as pd
import numpy as np
import simplekml
import datetime
import glob

#%% get dropsonde files

folder = "C:/Users/Lukas Monrad-Krohn/Desktop/uni/shk/Mosaic ACA/dropsonden/"
ds_files = sorted(glob.glob(folder+"*.dat"))




#%% D20200902_083725 did not work (no coordinates)

path = "C:/Users/Lukas Monrad-Krohn/Desktop/uni/shk/Mosaic ACA/dropsonden/ds_no_position_recorded/D20200902_083725.dat"

ds = pd.read_fwf(path, skiprows=14, header=None)
ds.columns = ['Time','UTC','Press','Temp','Dewpt','RH','Uwind','Vwind','Wspd','Dir','dZ','GeoPoAlt','Lon','Lat','GPSAlt']
ds = ds.drop(columns=['Press', 'Temp', 'Dewpt', 'RH', 'Uwind', 'Vwind', 'Wspd', 'Dir', 'dZ', 'GeoPoAlt'], axis=1)
ds.replace(-999.00,np.nan, inplace=True)
ds.dropna(axis=0, inplace=True)

date = '2020-09-02'
time = '083725'

kml = simplekml.Kml()
coords = (11.233333, 81.333333,  2881.00)

pnt = kml.newpoint(name="DS_MOSAiC-ACA_"+date+'_'+time, coords=[coords])
pnt.style.iconstyle.icon.href = 'http://www.uni-leipzig.de/~sorpic/ge/dropsonde2.png'
pnt.style.iconstyle.scale = 1
pnt.altitudemode = 'absolute'

kml.save('DS_'+date+'_'+time+'.kml',format=True)
print('process finished')

#%% define function (to create kml for every ds)

def ds_to_kml(folder):
    #folder = "C:/Users/Lukas Monrad-Krohn/Desktop/uni/shk/Mosaic ACA/dropsonden/"
    ds_files = sorted(glob.glob(folder+"*.dat"))
    
    for i in range(len(ds_files)):
        
        ds = pd.read_fwf(ds_files[i], skiprows=14, header=None)
        ds.columns = ['Time','UTC','Press','Temp','Dewpt','RH','Uwind','Vwind','Wspd','Dir','dZ','GeoPoAlt','Lon','Lat','GPSAlt']
        ds = ds.drop(columns=['Press', 'Temp', 'Dewpt', 'RH', 'Uwind', 'Vwind', 'Wspd', 'Dir', 'dZ', 'GeoPoAlt'], axis=1)
        ds.replace(-999.00,np.nan, inplace=True)
        ds.dropna(axis=0, inplace=True)
        #print(ds)
        
        date = ds_files[i][-19:-15]+"-"+ds_files[i][-15:-13]+"-"+ds_files[i][-13:-11]
        time = ds_files[i][-10:-4]
        
        kml = simplekml.Kml()
        subset = ds[['Lon', 'Lat', 'GPSAlt']]
        tuples = [tuple(x) for x in subset.to_numpy()]
        lin = kml.newlinestring(name="DS_MOSAiC-ACA_"+date+"_"+time,
                                coords=tuples)
        
        #print(date, time, tuples[0:10])
        lin.style.linestyle.width=5
        lin.style.linestyle.color = simplekml.Color.red
        lin.altitudemode=simplekml.AltitudeMode.absolute
        
        pnt = kml.newpoint(name="DS_MOSAiC-ACA_"+date+'_'+time, coords=[tuples[0]])
        pnt.style.iconstyle.icon.href = 'http://www.uni-leipzig.de/~sorpic/ge/dropsonde2.png'
        pnt.style.iconstyle.scale = 1
        pnt.altitudemode = 'absolute'
        
        
        kml.save('DS_'+date+'_'+time+'.kml',format=True)
        #print(date, time, 'finished')
        
        
#%% run func

ds_to_kml("C:/Users/Lukas Monrad-Krohn/Desktop/uni/shk/Mosaic ACA/dropsonden/")