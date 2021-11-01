# -*- coding: utf-8 -*-
"""
Created on Sat Jul 24 20:58:09 2021

@author: Lukas Monrad-Krohn
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from datetime import timedelta
import matplotlib.dates as mdates
import netCDF4
from netCDF4 import Dataset
from netCDF4 import num2date


runtime_start = datetime.now()

filepath = 'C:/Users/Lukas Monrad-Krohn/Desktop/uni/shk/spectral_imager/example/Flight_20200910_EagleHawk_CenterPixels_Radiances3.nc'

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

ncfile = Dataset(filepath, mode = 'r')

rad = ncfile.variables['rad']
time = ncfile.variables['time']
dtime = ncfile.variables['dtime']
wvl = ncfile.variables['wvl']

#time2 = num2date(time[:], units = time.units, calendar = time.calendar)
#timex = [str(i) for i in time2]

time_diff = time[:]-time[0]

# find wavelengths
idx = [find_nearest(wvl, i) for i in [1650, 2100]]

# choose times in seconds
times = [5, 100, 300]

#colors for multiple plots (maximum: 5)
color_ls1 = ['violet', 'darkblue', 'blue', 'lightblue', 'teal']
color_ls2 = ['darkgreen', 'forestgreen', 'limegreen', 'lightgreen', 'yellow']

# max 3 with running mean
color_ls3 = ['yellow', 'orange', 'red']

# number of steps for running mean
N = 10
allow_running_mean = False

fig, (ax1, ax2) = plt.subplots(2, 1, figsize = (8,10))

ax1.set_title('Radiance for certain wavelengths over time')
ax1.set_ylabel('Radiance  $ [10^{-3}\,W\,m^{-2}\,sr^{-1}\,nm^{-1}]$')
ax1.set_xlabel('time [s]') #gegebenenfalls einheit hinzufügen
for i in range(len(idx)):
    ax1.plot(time_diff, rad[idx[i],:], color=color_ls1[i], label = 'wvl = %.1f nm' %wvl[idx[i]])
    if allow_running_mean == True:
        run_mean = np.convolve(rad[idx[i],:], np.ones(N)/N, mode='valid')
        ax1.plot(time_diff[:-(N-1)], run_mean, label = 'running mean, wvl = %.1f nm' %wvl[idx[i]], 
                 color = color_ls3[i], linestyle = '--')
#ax1.set_xticks([timex[0::1200]])
#ax1.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))

ax1.legend()

ax2.set_title('Radiance for certain time over all wavelengths')
ax2.set_ylabel('Radiance  $ [10^{-3}\,W\,m^{-2}\,sr^{-1}\,nm^{-1}]$')
ax2.set_xlabel('wavelength [nm]')
for i in range(len(times)):
    ax2.plot(wvl, rad[:,20*times[i]], color=color_ls2[i], label = 'time = %.1f s' %time_diff[20*times[i]])
ax2.legend()

#plt.savefig('nc_test.png', dpi = 300, bbox_inches = 'tight')
plt.show()


#ncfile.close()

               
print(datetime.now()-runtime_start)