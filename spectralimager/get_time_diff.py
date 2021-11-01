# -*- coding: utf-8 -*-
"""
Created on Sun Apr 25 12:01:47 2021

@author: Lukas Monrad-Krohn
"""
from spectral import *
import numpy as np
from tqdm import tqdm_notebook as tqdm
import matplotlib.pyplot as plt
from datetime import datetime
from datetime import timedelta
import glob
import matplotlib.dates as mdates

#list_of_files = sorted(glob.glob(input_folder+"*.hdr"))
list_of_files = ['C:/Users/Lukas Monrad-Krohn/Desktop/uni/shk/spectral_imager/example/MOSAiC_ACA_Flight_20200910a_0910-1014_radiance.hdr',
                 'C:/Users/Lukas Monrad-Krohn/Desktop/uni/shk/spectral_imager/example/MOSAiC_ACA_Flight_20200910a_0910-1015-1_radiance.hdr']
#1200 first, 640 second



img_640 = open_image(list_of_files[1])
img_1200 = open_image(list_of_files[0])

b_640 = np.asarray(img_640.bands.centers)
b_1200 = np.asarray(img_1200.bands.centers)
b = [b_640, b_1200]

arr_640= img_640.asarray()
arr_1200= img_1200.asarray()


#Get time
date_640 = img_640.metadata['acquisition date'][-10:]
date_640 = date_640[-4:]+"-"+date_640[-7:-5]+"-"+date_640[-10:-8]
start_time_640 = img_640.metadata['gps start time'][-13:-1]

date_1200 = img_1200.metadata['acquisition date'][-10:]
date_1200 = date_1200[-4:]+"-"+date_1200[-7:-5]+"-"+date_1200[-10:-8]
start_time_1200 = img_1200.metadata['gps start time'][-13:-1]


starttime_640 = datetime.fromisoformat(date_640+" "+start_time_640)
time_ls_640  = []
for i in range(0,len(arr_640)):
    time_ls_640.append(starttime_640 + i* timedelta(seconds=0.05))
time_640 = np.asarray(time_ls_640)

starttime_1200 = datetime.fromisoformat(date_1200+" "+start_time_1200)
time_ls_1200  = []
for i in range(0,len(arr_1200)):
    time_ls_1200.append(starttime_1200 + i* timedelta(seconds=0.05))
time_1200 = np.asarray(time_ls_1200)

print(time_640[0], time_640[-1])
print(time_1200[0], time_1200[-1])

start_640 = time_640[0].strftime("%m/%d/%Y, %H:%M:%S.%f")
end_640 = time_640[-1].strftime("%m/%d/%Y, %H:%M:%S.%f")
start_1200 = time_1200[0].strftime("%m/%d/%Y, %H:%M:%S.%f")
end_1200 = time_1200[-1].strftime("%m/%d/%Y, %H:%M:%S.%f")


k = int(np.round((float(start_640[-9:]) - float(start_1200[-9:])) / 0.05 * (-1)))

z = int(np.round((float(end_640[-9:]) - float(end_1200[-9:])) / 0.05 * (-1)))

print(k, z)
#trial and error
arr_640 = arr_640[k:z,:,:]


print(len(arr_640), len(arr_1200))

