# -*- coding: utf-8 -*-
"""
Created on Mon Sep  6 15:52:18 2021

@author: Lukas Monrad-Krohn
"""

import numpy as np
import matplotlib.pyplot as plt
import netCDF4
from netCDF4 import Dataset
import datetime



file = 'Flight_20200910_1015_EagleHawk_3Pixelrows_Radiances_interp2.nc'
fpath = 'C:/Users/Lukas Monrad-Krohn/Desktop/uni/shk/spectral_imager/example/'

outputfolder = 'C:/Users/Lukas Monrad-Krohn/Desktop/uni/shk/spectral_imager/example/outputfolder_vergl/'

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

wavel = [600, 1000, 1600]



ncfile = Dataset(fpath+file, mode = 'r')

rad = ncfile.variables['rad'] #center
rad1 = ncfile.variables['rad1'] #top
rad2 = ncfile.variables['rad2'] #bottom
time = ncfile.variables['time']
wvl = ncfile.variables['wvl']

idx = [find_nearest(wavel, wavel[i]) for i in range(len(wavel))]

time = [time[k] - time[0] for k in range(len(time))]


fig, (ax1, ax2, ax3) = plt.subplots(3,1, sharex=True, sharey=True, figsize=(8,15))
plt.rcParams['axes.grid'] = True
plt.suptitle('Vergleich der oberen und unteren nc')
plt.xlabel('time [s]')
plt.ylabel('Radiance $ [10^{-3}\,W\,m^{-2}\,sr^{-1}\,nm^{-1}]$')

ax1.set_title(str(wavel[0])+' nm')
ax1.plot(time, rad1[idx[0],:], label = 'upper bit', color = 'darkred', alpha = 0.7)
ax1.plot(time, rad2[idx[0],:], label = 'lower bit', color = 'red', alpha = 0.7)
ax1.plot(time, (rad1[idx[0],:] + rad2[idx[0],:]) / 2, 
         label = 'mean of upper and lower', color = 'orange', alpha = 0.7)
ax1.legend(loc = 'lower right')
print('11:', np.where(rad1[idx[0],:] == max(rad1[idx[0],:])), 
      '12:', np.where(rad2[idx[0],:] == max(rad2[idx[0],:])), 
      '13:',  np.where((rad1[idx[0],:] + rad2[idx[0],:]) / 2 == max((rad1[idx[0],:] + rad2[idx[0],:]) / 2)))
print(r'\n')

ax2.set_title(str(wavel[1])+' nm')
ax2.plot(time, rad1[idx[1],:], label = 'upper bit', color = 'darkgreen', alpha = 0.7)
ax2.plot(time, rad2[idx[1],:], label = 'lower bit', color = 'green', alpha = 0.7)
ax2.plot(time, (rad1[idx[1],:] + rad2[idx[1],:]) / 2, 
         label = 'mean of upper and lower', color = 'lime', alpha = 0.7)
ax2.legend(loc = 'lower right')
print('21:', np.where(rad1[idx[1],:] == max(rad1[idx[1],:])), 
      '22:', np.where(rad2[idx[1],:] == max(rad2[idx[1],:])), 
      '23:',  np.where((rad1[idx[1],:] + rad2[idx[1],:]) / 2 == max((rad1[idx[1],:] + rad2[idx[1],:]) / 2)))
print(r'\n')

ax3.set_title(str(wavel[2])+' nm')
ax3.plot(time, rad1[idx[2],:], label = 'upper bit', color = 'darkblue', alpha = 0.7)
ax3.plot(time, rad2[idx[2],:], label = 'lower bit', color = 'blue', alpha = 0.7)
ax3.plot(time, (rad1[idx[2],:] + rad2[idx[2],:]) / 2, 
         label = 'mean of upper and lower', color = 'skyblue', alpha = 0.7)
ax3.legend(loc = 'lower right')
print('31:', np.where(rad1[idx[2],:] == max(rad1[idx[2],:])), 
      '32:', np.where(rad2[idx[2],:] == max(rad2[idx[2],:])), 
      '33:',  np.where((rad1[idx[2],:] + rad2[idx[2],:]) / 2 == max((rad1[idx[2],:] + rad2[idx[2],:]) / 2)))
print(r'\n')


#ax1.legend()
#ax2.legend()
#ax3.legend()


#plt.savefig(outputfolder+'testtest.png', dpi = 300, bbox_inches = 'tight')
plt.show()


fig2, (ax1, ax2, ax3) = plt.subplots(3,1, sharex=True, sharey=True, figsize=(8,15))
plt.rcParams['axes.grid'] = True
plt.suptitle('Vergleich der Center und Mittelwerte')
plt.xlabel('time [s]')
plt.ylabel('Radiance $ [10^{-3}\,W\,m^{-2}\,sr^{-1}\,nm^{-1}]$')

ax1.set_title(str(wavel[0])+ ' nm')
ax1.plot(time, rad[idx[0],:], label = 'center', color ='darkblue', alpha = 0.7)
ax1.plot(time, (rad1[idx[0],:] + rad2[idx[0],:]) / 2, label = 'mean', color = 'orange', alpha = 0.7)
ax1.legend()

ax2.set_title(str(wavel[1])+' nm')
ax2.plot(time, rad[idx[1],:], label = 'center', color = 'darkblue', alpha = 0.7)
ax2.plot(time, (rad1[idx[1],:] + rad2[idx[1],:]) / 2, label = 'mean', color = 'orange', alpha = 0.7)
ax2.legend()

ax3.set_title(str(wavel[2])+' nm')
ax3.plot(time, rad[idx[2],:], label = 'center', color = 'darkblue', alpha = 0.7)
ax3.plot(time, (rad1[idx[2],:] + rad2[idx[2],:]) / 2, label = 'mean', color = 'orange', alpha = 0.7)
ax3.legend()

#plt.savefig(outputfolder+ 'center_mean.png', dpi = 300, bbox_inches = 'tight')
plt.show()





fig3 = plt.figure(figsize=(8,4))
plt.title('Abweichung der Mittelwerte vom Center, Verschiebung: +2')
plt.xlabel('time [s]')
plt.ylabel('deviation')
plt.rcParams['axes.grid'] = True

plt.plot(time[2:], ((rad1[idx[0],2:] + rad2[idx[0],2:]) / 2) / rad[idx[0],:-2], color = 'green', 
         label = str(wavel[0])+' nm', linestyle = '-.', zorder = 5, alpha = 0.9)
plt.plot(time[2:], ((rad1[idx[1],2:] + rad2[idx[1],2:]) / 2) / rad[idx[1],:-2], color = 'red', 
         label = str(wavel[1])+' nm', linestyle = '--', zorder = 5, alpha = 0.7)
plt.plot(time[2:], ((rad1[idx[2],2:] + rad2[idx[2],2:]) / 2) / rad[idx[2],:-2], color = 'blue', 
         label = str(wavel[2])+' nm', linestyle = ':', zorder = 5, alpha = 0.5)

plt.ylim(0.8, 1.25)
plt.legend()
plt.savefig(outputfolder+ 'dev_center_mean_interp+2.png', dpi = 300, bbox_inches = 'tight')
plt.show()

#%%

import numpy as np
import matplotlib.pyplot as plt
import netCDF4
from netCDF4 import Dataset
import datetime



file = 'Flight_20200910_1015_EagleHawk_3Pixelrows_Radiances_interp2.nc'
fpath = 'C:/Users/Lukas Monrad-Krohn/Desktop/uni/shk/spectral_imager/example/'

outputfolder = 'C:/Users/Lukas Monrad-Krohn/Desktop/uni/shk/spectral_imager/example/outputfolder_vergl/'

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

wavel = [990, 1010, 400,1200]



ncfile = Dataset(fpath+file, mode = 'r')

rad = ncfile.variables['rad'] #center
rad1 = ncfile.variables['rad1'] #top
rad2 = ncfile.variables['rad2'] #bottom
time = ncfile.variables['time']
wvl = ncfile.variables['wvl']

idx = [find_nearest(wvl, wavel[i]) for i in range(len(wavel))]

time = [time[k] - time[0] for k in range(len(time))]

fig4, (ax1,ax2) = plt.subplots(2,1,figsize=(8,12))
ax1.set_title('Vergleich Hawk Eagle nach Zeit')
ax1.set_xlabel('time[s]')
ax1.set_ylabel('Radiance $ [10^{-3}\,W\,m^{-2}\,sr^{-1}\,nm^{-1}]$')
plt.rcParams['axes.grid'] = True



ax1.plot(time[:-10], rad[idx[0],:-10], color = 'darkred', label = str(wvl[idx[0]])+' nm')
ax1.plot(time[:-10], rad[idx[1],10:], color = 'orange', label = str(wvl[idx[1]])+' nm')
ax1.plot(time[:-10], rad[idx[2],:-10], color = 'green', label = str(wvl[idx[2]])+' nm')
ax1.plot(time[:-10], rad[idx[3],10:], color = 'blue', label = str(wvl[idx[3]])+' nm')
ax1.legend()

ax2.set_title('Abweichung Eagle Hawk')
ax2.set_xlabel('time[s]')
ax2.set_ylabel('Radiance $ [10^{-3}\,W\,m^{-2}\,sr^{-1}\,nm^{-1}]$')
ax2.set_ylim(0.8, 1.8)

ax2.plot(time[:-10], rad[idx[0],:-10] / rad[idx[1],10:], color='blue', label=str(wvl[idx[0]])+' und '+str(wvl[idx[1]])+' nm')
ax2.plot(time[:-10], rad[idx[2],:-10] / rad[idx[1],10:], color='green', label=str(wvl[idx[2]])+' und '+str(wvl[idx[1]])+' nm')
ax2.legend()

plt.savefig(outputfolder+ 'zeitvergleich-10.png', dpi = 300, bbox_inches = 'tight')
plt.show()



