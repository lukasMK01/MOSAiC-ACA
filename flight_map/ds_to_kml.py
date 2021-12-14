# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 18:00:01 2021

@author: Lukas Monrad-Krohn (lm73code@studserv.uni-leipzig.de / lukas@monrad-krohn.com)

This is a program, which makes an .kml-file of Dropsondedata(Location, Altitude and Time)
from a .dat-file.
"""

import pandas as pd
import numpy as np
import simplekml
import datetime
import glob

# get dropsonde files

ds_folder = "C:/Users/Lukas Monrad-Krohn/Desktop/uni/shk/Mosaic ACA/dropsonden/"


# define function (to create kml for every ds)
def ds_to_kml(folder):
    ds_files = sorted(glob.glob(folder+"*.dat"))
    
    for i in range(len(ds_files)):
        
        # read dropsondedata
        ds = pd.read_fwf(ds_files[i], skiprows=14, header=None)
        ds.columns = ['Time','UTC','Press','Temp','Dewpt','RH','Uwind','Vwind','Wspd','Dir','dZ','GeoPoAlt','Lon','Lat','GPSAlt']
        ds = ds.drop(columns=['Press', 'Temp', 'Dewpt', 'RH', 'Uwind', 'Vwind', 'Wspd', 'Dir', 'dZ', 'GeoPoAlt'], axis=1)
        ds.replace(-999.00,np.nan, inplace=True)
        ds.dropna(axis=0, inplace=True)
        #print(ds)
        
        # get times
        date = ds_files[i][-19:-15]+"-"+ds_files[i][-15:-13]+"-"+ds_files[i][-13:-11]
        time = ds_files[i][-10:-4]
        
        # create kml with simplekml
        kml = simplekml.Kml()
        
        # create tuples for the coords and join them as linestring
        subset = ds[['Lon', 'Lat', 'GPSAlt']]
        tuples = [tuple(x) for x in subset.to_numpy()]
        lin = kml.newlinestring(name="DS_MOSAiC-ACA_"+date+"_"+time,
                                coords=tuples)
        lin.style.linestyle.width=5
        lin.style.linestyle.color = simplekml.Color.red
        lin.altitudemode=simplekml.AltitudeMode.absolute
        
        # add point, which shows icon
        pnt = kml.newpoint(name="DS_MOSAiC-ACA_"+date+'_'+time, coords=[tuples[0]])
        pnt.style.iconstyle.icon.href = 'http://www.uni-leipzig.de/~sorpic/ge/dropsonde2.png'
        pnt.style.iconstyle.scale = 1
        pnt.altitudemode = 'absolute'
        
        # save kml (kmz is possible as well)
        kml.save('DS_'+date+'_'+time+'.kml',format=True)
        #print(date, time, 'finished')
        
        

# run func
ds_to_kml(ds_folder)



#%% D20200902_083725 did not work (no coordinates)
# for this case GPS-data wasn't available, so I got the coordinates by checking the aircraftposition with launchtime from
# flightreport

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