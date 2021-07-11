# -*- coding: utf-8 -*-
"""
Created on Sun May 16 12:45:20 2021

@author: Lukas Monrad-Krohn
"""

from spectral import *
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from datetime import timedelta
import glob
import matplotlib.dates as mdates
import pandas as pd
import xarray as xr
import netCDF4
from netCDF4 import Dataset


runtime_start = datetime.now()

#Input folder
input_folder = "C:/Users/Lukas Monrad-Krohn/Desktop/uni/shk/spectral_imager/example/"

#wavelength(s)
wvl1 = [640] # between 400 und 993nm
wvl2 = [1200] #between 931 und 2544nm
wvlx = [640, 1200]

#Output folder
output_folder= 'C:/Users/Lukas Monrad-Krohn/Desktop/uni/shk/spectral_imager/example/test'
#"test_result/"

times_to_plot = [5, 100, 300] # in seconds since start
wvl_to_plot = [1650, 2100] # in nm

create_csv = False
create_nc = True

first_run_of_day = True
if first_run_of_day == True:
    ab = 'a'
else:
    ab = 'b'
    

# which pixel row to calculate
# center or bottom quarter or top quarter or bottom and top
centerrow = False
bottomrow = True
toprow = True

#-------------------------------------------
# initialize plotting function
def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def create_plots(data_arr, time_arr, wavel_arr, times, wavel, timeplot=True, wvlplot=True, allow_running_mean = False):
        
    # find wavelengths
    idx = [find_nearest(wavel_arr, i) for i in wavel]
    
    # choose times in seconds
    #times = [5, 100, 300]
    
    #colors for multiple plots (maximum: 5)
    color_ls1 = ['violet', 'darkblue', 'blue', 'lightblue', 'teal']
    color_ls2 = ['darkgreen', 'forestgreen', 'limegreen', 'lightgreen', 'yellow']
    
    # max 3 with running mean
    color_ls3 = ['yellow', 'orange', 'red']
    
    # number of steps for running mean
    N = 10
    #allow_running_mean = False
    
    
    # start plotting ----------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize = (8,10))
    
    
    ax1.set_title('Radiance for certain wavelengths over time')
    ax1.set_ylabel('Radiance  $ [10^{-3}\,W\,m^{-2}\,sr^{-1}\,nm^{-1}]$')
    ax1.set_xlabel('time [s]')
    for i in range(len(idx)):
        ax1.plot(time_arr, data_arr[idx[i],:], color=color_ls1[i], label = 'wvl = %.1f nm' %wavel_arr[idx[i]])
        if allow_running_mean == True:
            run_mean = np.convolve(data_arr[idx[i],:], np.ones(N)/N, mode='valid')
            ax1.plot(time_arr[:-(N-1)], run_mean, label = 'running mean, wvl = %.1f nm' %wavel_arr[idx[i]], 
                     color = color_ls3[i], linestyle = '--')
    ax1.legend()
    
    ax2.set_title('Radiance for certain time over all wavelengths')
    ax2.set_ylabel('Radiance  $ [10^{-3}\,W\,m^{-2}\,sr^{-1}\,nm^{-1}]$')
    ax2.set_xlabel('wavelength [nm]')
    for i in range(len(times)):
        ax2.plot(wavel_arr, data_arr[:,20*times[i]], color=color_ls2[i], label = 'time = %.1f s' %time_arr[20*times[i]])
    ax2.legend()
    
    #plt.savefig('nc_test.png', dpi = 300, bbox_inches = 'tight')
    plt.show()
    
    
#------------------------------------------------------------------------------
# start calculations


list_of_files = sorted(glob.glob(input_folder+"*.hdr"))
#list_of_files = ['C:/Users/Lukas Monrad-Krohn/Desktop/uni/shk/spectral_imager/example/MOSAiC_ACA_Flight_20200910a_0910-1014_radiance.hdr',
#                 'C:/Users/Lukas Monrad-Krohn/Desktop/uni/shk/spectral_imager/example/MOSAiC_ACA_Flight_20200910a_0910-1015-1_radiance.hdr']
#1200 first, 640 second
print(list_of_files)




# get data

filename_1200 = list_of_files[1][-52:-4]
filename_640 = list_of_files[0][-50:-4]
print(filename_1200, filename_640)

img_640 = open_image(list_of_files[0])
img_1200 = open_image(list_of_files[1])
b_640 = np.asarray(img_640.bands.centers)
b_1200 = np.asarray(img_1200.bands.centers)
b = [b_640, b_1200]

arr_640= img_640.asarray()
arr_1200= img_1200.asarray()

#---------------------------------------
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


#----------------------------------
#clip data

start_640 = time_640[0].strftime("%m/%d/%Y, %H:%M:%S.%f")
end_640 = time_640[-1].strftime("%m/%d/%Y, %H:%M:%S.%f")
start_1200 = time_1200[0].strftime("%m/%d/%Y, %H:%M:%S.%f")
end_1200 = time_1200[-1].strftime("%m/%d/%Y, %H:%M:%S.%f")


k = int(np.round((float(start_640[-9:]) - float(start_1200[-9:])) / 0.05 * (-1)))

z = int(np.round((float(end_640[-9:]) - float(end_1200[-9:])) / 0.05 * (-1)))

print(k, z)
#trial and error
arr_640 = arr_640[k:z,:,:]

# only work if arr_640 is longer on both ends
# watch out for problems here !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

#-------------------------------------
#get time_arr
starttime = datetime.fromisoformat(date_1200+" "+start_time_1200)
print(starttime)

time_ls  = []

for i in range(0,len(arr_1200)):
    time_ls.append(starttime + i* timedelta(seconds=0.05))
time_arr = np.asarray(time_ls)
    
print('step 1')
#-------------------------------------
#sort data
col_name_ls = ['time']
b2 = np.concatenate([b_640, b_1200[11:]])

for i in range(len(b2)):
    col_name_ls.append(str(b2[i]))

#--------------------------------------
'''
for centerpixels use 183:201 for 1200 and 488:536 for 640

for 1/4 pixels use 87:105 for 1200 and 232:280 for 640

for 3/4 pixels use 279:297 for 1200 and 744:792 fro 640
'''

# center 
if centerrow == True:
    center_arr_1200 = [[0 for _ in range(len(b_1200[11:]))] for _ in range(len(arr_1200))]
    for j in range(len(b_1200[11:])): # wavelength
        for i in range(len(arr_1200)): # timesteps
            center_arr_1200[i][j] = np.mean(arr_1200[i, 183:201, 11+j])
    center_arr_1200 = np.asarray(center_arr_1200)
    
    
    center_arr_640 = [[0 for _ in range(len(b_640))] for _ in range(len(arr_640))]
    for j in range(len(b_640)): # wavelength
        for i in range(len(arr_640)): # timesteps
            center_arr_640[i][j] = np.mean(arr_640[i, 488:536, j])
    center_arr_640 = np.asarray(center_arr_640)

    print('center calcing finished')

# botton quarter
if bottomrow == True:
    bottom_arr_1200 = [[0 for _ in range(len(b_1200[11:]))] for _ in range(len(arr_1200))]
    for j in range(len(b_1200[11:])): # wavelength
        for i in range(len(arr_1200)): # timesteps
            bottom_arr_1200[i][j] = np.mean(arr_1200[i, 87:105, 11+j])
    bottom_arr_1200 = np.asarray(bottom_arr_1200)
    
    
    bottom_arr_640 = [[0 for _ in range(len(b_640))] for _ in range(len(arr_640))]
    for j in range(len(b_640)): # wavelength
        for i in range(len(arr_640)): # timesteps
            bottom_arr_640[i][j] = np.mean(arr_640[i, 232:280, j])
    bottom_arr_640 = np.asarray(bottom_arr_640)

    print('bottom calcing finished')
    
# top quarter
if toprow == True:
    top_arr_1200 = [[0 for _ in range(len(b_1200[11:]))] for _ in range(len(arr_1200))]
    for j in range(len(b_1200[11:])): # wavelength
        for i in range(len(arr_1200)): # timesteps
            top_arr_1200[i][j] = np.mean(arr_1200[i, 279:297, 11+j])
    top_arr_1200 = np.asarray(top_arr_1200)
    
    
    top_arr_640 = [[0 for _ in range(len(b_640))] for _ in range(len(arr_640))]
    for j in range(len(b_640)): # wavelength
        for i in range(len(arr_640)): # timesteps
            top_arr_640[i][j] = np.mean(arr_640[i, 744:792, j])
    top_arr_640 = np.asarray(top_arr_640)

    print('top calcing finished')

# all timedeltas
#x = time_arr[:] - time_arr[0]
#x = np.asarray([i.total_seconds() for i in x])
#print(x, x.shape)

# all radiances
#p_all = np.concatenate((p_arr_640, p_arr_1200[:, 11:]), axis=1)

    

#---------------------------------------
#create dataframe
# currently only available for centerrow
create_csv = False
create_nc = True

if create_csv == True:
    csv_filename = 'Eagle_Hawk_'+date_1200+'_'+start_time_1200[:2]+start_time_1200[3:5]+'hallo.csv'
    
    
    dicti = {}
    dicti['time'] = time_arr
    for i in range(len(b_640)):
        dicti[str(b_640[i])] = np.around(center_arr_640[:, i],2)
        
    for i in range(len(b_1200[11:])):
        dicti[str(b_1200[11+i])] = np.around(center_arr_1200[:, i],2)
    
    
    df = pd.DataFrame(dicti)
    df.to_csv(csv_filename, index=False)
    

    print('csv finished')

#----------------------------------------
#create netcdf
datee = date_1200[:4]+date_1200[5:7]+date_1200[8:10]

date_datetime = datetime.strptime(date_1200+' 00:00:00.00', '%Y-%m-%d %H:%M:%S.%f')

# consider first_run_of_day

if create_nc == True:
    #'+date_1200+'_'+start_time_1200[:2]+start_time_1200[3:5]+'
    ncfile = netCDF4.Dataset('Flight_'+ datee +'_EagleHawk_2Pixels_Radiances.nc', mode='w', format='NETCDF4')
    #centerPixels or someting else
    
    today = datetime.today()
    #create attributes
    ncfile.title = 'Combination of Aisa Eagle and Hawk spectrum'
    ncfile.subtitle = 'Radiance spectra of 18, respectivly 48, averaged center pixels along time'
    ncfile.mission = 'MOSAiC-ACA'
    ncfile.platform = 'Polar 5'
    ncfile.instrument = 'Aisa Eagle and Aisa Hawk'
    ncfile.flight_id = datee + ab
    ncfile.sourcefile = str(filename_640)+ ', ' +str(filename_1200)
    ncfile.date_last_revised = '...'
    ncfile.featureType = '...'
    ncfile.Conventions = '...'
    ncfile.version = '1.0'
    ncfile.history = 'acquired by MOSAiC-ACA as .raw-Files, processed by Michael Schäfer, formatted to netcdf by Lukas Monrad-Krohn'
    ncfile.file_created = 'File created by L. Monrad-Krohn (email: lm73code@studserv.uni-leipzig.de) [supervised  by M. Klingebiel (email: marcus.klingebiel@uni-leipzig.de)] on '+today.strftime('%B %d, %Y')
    ncfile.institute = 'Leipzig Institute for Meteorology (LIM), Leipzig, Germany'
    
    #initialize dimensions
    time_dim = ncfile.createDimension('time', len(time_arr))
    wvl_dim = ncfile.createDimension('wvl', len(b2))
    
    #initialize variables
    time = ncfile.createVariable('time', np.float64, ('time',))
    time.units = 'seconds since 1970-01-01 00:00'#'+ date_1200 #1970-01-01 00:00'  # seconds since str(time_arr[0])
    time.standard_name = 'time'
    time.long_name = 'Time in seconds since 1970-01-01 00:00' # str(time_arr[0])
    time.calendar = 'standard'
    time.axis = 'T'
    
    #dtime = ncfile.createVariable('dtime', 'S1', ('time',))
    #dtime.units = '%Y-%m-%d %H:%M:%S.%f'
    #dtime.standart_name = 'datetime format'
    #dtime.axis = 'T'
    
    wvl = ncfile.createVariable('wvl', np.float32, ('wvl',))
    wvl.units = 'nm (10^{-9} m)'
    wvl.standard_name = 'radiation_wavelengths'
    wvl.long_name = 'Wavelengths of the spectral channels'
    wvl.axis = 'L'
    
    rad = ncfile.createVariable('rad', np.float32, ('wvl', 'time', ))
    rad.units = '10^{-3} W m^{-2} sr^{-1} nm^{-1}'
    rad.standard_name = 'Radiance' #upwelling_radiance_per_unit_wavelength_in_air
    rad.long_name = 'spectral uppward Radiance measured inflight by Aisa Eagle and Hawk'
    
    if toprow == True:
        rad2 = ncfile.createVariable('rad2', np.float32, ('wvl', 'time', ))
        rad2.units = '10^{-3} W m^{-2} sr^{-1} nm^{-1}'
        rad2.standard_name = 'Radiance'
        rad2.long_name = 'spectral uppward Radiance measured inflight by Aisa Eagle and Hawk'
        
    
    #time_delta = time_arr[:] - datetime(1970, 1, 1, 0, 0, 0, 0)#date_datetime #datetime(1970, 1, 1, 0, 0, 0, 0) # time_arr[0] for seconds since flightstart
    time_delta = [i.timestamp() for i in time_arr[:]]
    #time_delta = np.asarray([i.total_seconds() for i in time_delta])
    #time_delta = np.float32(time_delta)
    
    #dtime = [str(i) for i in time_arr]
    #dtime = np.array(dtime, dtype='object')
    #dtime_str = netCDF4.stringtochar(np.array(dtime, 'S26'))
    
    if centerrow == True:
        p_all_c = np.concatenate((center_arr_640, center_arr_1200[:,:]), axis=1)
        p_all_c = np.around(p_all_c, 2)
        p_all_c = np.float32(p_all_c)
    if bottomrow == True:
        p_all_b = np.concatenate((bottom_arr_640, bottom_arr_1200[:,:]), axis=1)
        p_all_b = np.around(p_all_b, 2)
        p_all_b = np.float32(p_all_b)
    if toprow == True:
        p_all_t = np.concatenate((top_arr_640, top_arr_1200[:,:]), axis=1)
        p_all_t = np.around(p_all_t, 2)
        p_all_t = np.float32(p_all_t)
    
    
    b2 = np.float32(b2)
    
    
    # write data to variables
    time[:] = time_delta
    #dtime[:] = dtime
    wvl[:] = b2
    if centerrow == True:
        rad[:] == np.transpose(p_all_c)
    if bottomrow == True:
        rad[:] = np.transpose(p_all_b)
    if toprow == True:
        rad2[:] = np.transpose(p_all_t)
    
    # plot stuff
    create_plots(data_arr = rad, time_arr = time[:]-time[0], wavel_arr = wvl, times = times_to_plot, wavel = wvl_to_plot)
    
    ncfile.close()
    print('nc finished')

print(datetime.now()-runtime_start)
#%%
ncfile.close()
#%%


import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from datetime import timedelta
import matplotlib.dates as mdates
import pandas as pd
import netCDF4
from netCDF4 import Dataset


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx
    
def open_nc(nc_path):
    ncfile = Dataset(nc_path, mode='r')
    
    rad = ncfile.variables['rad']
    time = ncfile.variables['time']
    wvl = ncfile.variables['wvl']
    
    return( rad, time, wvl)
#color_ls = ['darkorange', 'darkred', 'limegreen', 'darkgreen', 'navy', 'darkblue']
#axis1_arr should be time and 2 wavelength
def create_plots(data_arr, time_arr, wavel_arr, times, wavel, timeplot=True, wvlplot=True, allow_running_mean = False):
        
    # find wavelengths
    idx = [find_nearest(wavel_arr, i) for i in wavel]
    
    # choose times in seconds
    #times = [5, 100, 300]
    
    #colors for multiple plots (maximum: 5)
    color_ls1 = ['violet', 'darkblue', 'blue', 'lightblue', 'teal']
    color_ls2 = ['darkgreen', 'forestgreen', 'limegreen', 'lightgreen', 'yellow']
    
    # max 3 with running mean
    color_ls3 = ['yellow', 'orange', 'red']
    
    # number of steps for running mean
    N = 10
    #allow_running_mean = False
    
    
    # start plotting ----------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize = (8,10))
    
    
    ax1.set_title('Radiance for certain wavelengths over time')
    ax1.set_ylabel('Radiance  $ [10^{-3}\,W\,m^{-2}\,sr^{-1}\,nm^{-1}]$')
    ax1.set_xlabel('time [s]')
    for i in range(len(idx)):
        ax1.plot(time_arr, data_arr[idx[i],:], color=color_ls1[i], label = 'wvl = %.1f nm' %wavel_arr[idx[i]])
        if allow_running_mean == True:
            run_mean = np.convolve(data_arr[idx[i],:], np.ones(N)/N, mode='valid')
            ax1.plot(time_arr[:-(N-1)], run_mean, label = 'running mean, wvl = %.1f nm' %wavel_arr[idx[i]], 
                     color = color_ls3[i], linestyle = '--')
    ax1.legend()
    
    ax2.set_title('Radiance for certain time over all wavelengths')
    ax2.set_ylabel('Radiance  $ [10^{-3}\,W\,m^{-2}\,sr^{-1}\,nm^{-1}]$')
    ax2.set_xlabel('wavelength [nm]')
    for i in range(len(times)):
        ax2.plot(wavel_arr, data_arr[:,20*times[i]], color=color_ls2[i], label = 'time = %.1f s' %time_arr[20*times[i]])
    ax2.legend()
    
    #plt.savefig('nc_test.png', dpi = 300, bbox_inches = 'tight')
    plt.show()
    
nc_path = 'C:/Users/Lukas Monrad-Krohn/Desktop/uni/shk/spectral_imager/example/Flight_20200910_EagleHawk_CenterPixels_Radiances3.nc'
    

rad, time, wvl = open_nc(nc_path)

create_plots(data_arr = rad, time_arr = time[:]-time[0], wavel_arr = wvl, times = [5, 100, 300], wavel = [1650, 2100])

#print(datetime.now()-runtime_start)