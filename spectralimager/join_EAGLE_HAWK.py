# -*- coding: utf-8 -*-
"""
Created on Sun May 16 12:45:20 2021

@author: Lukas Monrad-Krohn (lm73code@studserv.uni-leipzig.de / lukas@monrad-krohn.com)

This python script will combine the dataarrays of AisaEAGLE and AiseHAWK to one netcdf file. Deteils
can be found in the powerpoint presentation.
Around line 150 it gets really complicated. Please make sure, to understand this well. There is still the need
to correct some incompatibilities manually afterwards.
"""

# imports
from spectral import *
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from datetime import timedelta
import glob
import matplotlib.dates as mdates
import pandas as pd
import netCDF4
from netCDF4 import Dataset
import sys
from tqdm import tqdm


runtime_start = datetime.now()

# organizing stuff #################################################################
#Input folder
input_folder_eagle = "/projekt_agmwend/data/MOSAiC_ACA_S/Flight_20200910a/AisaEAGLE/"
input_folder_hawk = "/projekt_agmwend/data/MOSAiC_ACA_S/Flight_20200910a/AisaHAWK/"

#wavelength(s)
wvl1 = [640] # between 400 und 993nm
wvl2 = [1200] #between 931 und 2544nm
wvlx = [640, 1200]

#Output folder
output_folder_lm= '/home/lmkrohn/spec_img/' #used for testing only, please use your own
output_folder= '/projekt_agmwend/data/MOSAiC_ACA_S/Flight_20200910a/AisaEAGLE_HAWK_combined/'


times_to_plot = [5, 100, 300] # in seconds since start
wvl_to_plot = [1650, 2100] # in nm

# decisions
run_for_all = True


createplot = False
saveplot = False

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

################################################################################
# initialize plotting function
def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def create_plots(data_arr, time_arr, wavel_arr, times, wavel, date_of_flight, time_of_flight, timeplot=True, wvlplot=True, allow_running_mean = False):
        
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
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize = (8,11))
    
    
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
    
    if saveplot == True:
        plt.savefig('Flight_'+date_of_flight+'_'+time_of_flight+ '_EagleHawk_2Pixelrows_Radiances.png', dpi = 300, bbox_inches = 'tight')
    plt.show()
    
    
#------------------------------------------------------------------------------
# start calculations
# running for all or just testing for a single fiel?
if run_for_all == True:
    eagle = sorted(glob.glob(input_folder_eagle+"*.hdr"))
    hawk = sorted(glob.glob(input_folder_hawk+"*.hdr"))

else:
    list_of_files =  ['/projekt_agmwend/data/MOSAiC_ACA_S/Flight_20200910a/AisaEAGLE/MOSAiC_ACA_Flight_20200910a_0910-1407_radiance.hdr', '/projekt_agmwend/data/MOSAiC_ACA_S/Flight_20200910a/AisaHAWK/MOSAiC_ACA_Flight_20200910a_0910-1407-1_radiance.hdr']

    # ['/projekt_agmwend/data/MOSAiC_ACA_S/Flight_20200910a/AisaEAGLE/MOSAiC_ACA_Flight_20200910a_0910-1014_radiance.hdr', '/projekt_agmwend/data/MOSAiC_ACA_S/Flight_20200910a/AisaHAWK/MOSAiC_ACA_Flight_20200910a_0910-1015-1_radiance.hdr']

    eagle = []
    hawk = []
    for i in range(len(list_of_files)):
        if list_of_files[i][-15:] == '-1_radiance.hdr':
            hawk.append(list_of_files[i])
        else:
            eagle.append(list_of_files[i])
        

# process to check for irregularities
print(len(eagle))
print(len(hawk))


list_of_problems = []
if len(hawk)!=len(eagle):
    for j in range(len(hawk)-abs(len(hawk) - len(eagle))):
        if hawk[j][-19:-15] != eagle[j][-17:-13]:
            list_of_problems.append(j)
            print(j)
            print(hawk[j][-19:-15], eagle[j][-17:-13])
            #print('\n')
    #print(list_of_problems)

# correct errors manually (first step)
# if you run it and get an error this is probably the first piece to check and manually correct
# it with the information printed above!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
del eagle[6]
del hawk[-2:]

print(len(eagle), len(hawk), 'eagle then hawk')
for i in range(len(hawk) - abs(len(hawk) - len(eagle))):
    print(i)
    print(hawk[i][-19:-15], eagle[i][-17:-13], 'hawk then eagle')

if len(hawk) != len(eagle):
    sys.exit('ERROR: hawk and eagle still don\'t have the same length')
    
# now you have the final information to check, if all the eagle and hawk datasets for each part of the flight 
# match each other
# --> start running it.

llist = [0, 44]
for num in tqdm(range(len(eagle))):#44 is maxlen range(eagle) or use range([5,6]) to just do one

    # get data
    
    filename_1200 = hawk[num][-52:-4]
    filename_640 = eagle[num][-50:-4]
    print(filename_1200, filename_640)
    
    img_640 = open_image(eagle[num])
    img_1200 = open_image(hawk[num])
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

    print(arr_640.shape, arr_1200.shape)
    
    
    #----------------------------------
    #clip data
    # we assume, that first Eagle is started, then Hawk and Hawk is stopped before Eagle
    # watch out for differences, or if it doesn't happen in the 'same' minute
    
    start_640 = time_640[0].strftime("%m/%d/%Y, %H:%M:%S.%f")
    end_640 = time_640[-1].strftime("%m/%d/%Y, %H:%M:%S.%f")
    start_1200 = time_1200[0].strftime("%m/%d/%Y, %H:%M:%S.%f")
    end_1200 = time_1200[-1].strftime("%m/%d/%Y, %H:%M:%S.%f")
    
    # k and z are parameters used to cut the dataarrays to the same starting and ending time
    k = int(np.round((float(start_640[-9:]) - float(start_1200[-9:])) / 0.05 * (-1)))
    
    z = int(np.round((float(end_640[-9:]) - float(end_1200[-9:])) / 0.05 * (-1)))
    
    #improv !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # below are different corrections to cope with variable problems, if Eagle and Hawk started in different minutes
    # or they were not always started first one then the other but mixed up.
    # If you get an error saying, that the dimensions of the array don't fit, I choose to correct it manually,
    # because it was getting too complicated in the correction part below. Good Luck! The manual correction can be done with the
    # information from the last 3 print statements. 
    #k = 140
    #z = -108

    

    print('k,z = ', k, z)

    k_1200 = False
    z_1200 = False


    if k < 0:
        if float(start_640[-9:]) < float(start_1200[-9:]):
            k = int(np.round((float(start_640[-9:]) - float(start_1200[-9:]) -60) / 0.05 * (-1)))
            k_1200 = False
            #arr_640 = arr_640[k:,:,:]
            print('k = ',k)
        if float(start_640[-9:]) > float(start_1200[-9:]):
            k = False
            k_1200 = int(np.round((float(start_1200[-9:]) - float(start_640[-9:])) / 0.05 * (-1)))
            #arr_1200 = arr_1200[k_1200:,:,:]
            print('k_1200 = ', k_1200)
                         
    if z > 0:
        if float(end_640[-9:]) > float(end_1200[-9:]):
            z = int(np.round((float(end_640[-9:]) - float(end_1200[-9:]) +60) / 0.05 * (-1)))
                
            z_1200 = False
            #arr_640 = arr_640[:z,:,:]
            print('z = ', z)
        if float(end_640[-9:]) < float(end_1200[-9:]):
            z = False
            z_1200 = int(np.round((float(end_1200[-9:]) - float(end_640[-9:])) / 0.05 * (-1)))
            #arr_1200 = arr_1200[:z_1200,:,:]
            print('z_1200 = ', z_1200)

    if k != False:
        arr_640 = arr_640[k:,:,:]
    if z != False:
        arr_640 = arr_640[:z,:,:]

    if k_1200 != False:
        arr_1200 = arr_1200[k_1200:,:,:]
    if z_1200 != False:
        arr_1200 = arr_1200[:z_1200,:,:]

    #to correct mistakes at rounding off:

    dim640 = arr_640.shape
    dim1200 = arr_1200.shape
    print(dim640, dim1200)

    if dim640[0] < dim1200[0]:
        arr_1200 = arr_1200[:-1,:,:]
    if dim640[0] > dim1200[0]:
        arr_640 = arr_640[:-1,:,:]
    
    #arr_640 = arr_640[k:z,:,:]
    
    # does not work if the succession is wrong and die difference is across the minute
    # watch out for problems here !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    
    #-------------------------------------
    #get time_arr
    starttime = datetime.fromisoformat(date_1200+" "+start_time_1200)
    print(starttime)
    
    time_ls  = []
    time_ls_old = []
    
    for i in range(0,len(arr_1200)):
        time_ls.append(starttime + i* timedelta(seconds=0.05))
    time_arr = np.asarray(time_ls)


    for i in range(0,len(arr_640)):
        time_ls_old.append(datetime.fromisoformat(date_640+' '+start_time_640) + (k+i) * timedelta(seconds=0.5))
    time_arr_old = np.asarray(time_ls_old)

    time_delta = [i.timestamp() for i in time_arr[:]]
    time_delta_old = [i.timestamp() for i in time_arr_old[:]]
    
        
    print('time calcing finished')
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

    
    def interpolate(arr, new_time=time_delta, old_time=time_delta_old):
        for i in range(len(arr[0,:])):
            arr[:,i] = np.interp(new_time, old_time, arr[:,i])

        return arr
    
    
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

        center_arr_640 = interpolate(center_arr_640)
        print(center_arr_640.shape)
    
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

        bottom_arr_640 = interpolate(bottom_arr_640)
        print(bottom_arr_640.shape)
    
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

        top_arr_640 = interpolate(top_arr_640)
        print(top_arr_640.shape)
    
        print('top calcing finished')
    

    #---------------------------------------
    #create dataframe to convert to csv
    # currently only available for centerrow
    create_csv = False
    create_nc = True # just making sure it really is off for me. Beware if you want to create a csv.
    
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
    timee = start_time_1200[0:2]+start_time_1200[3:5]
    
    date_datetime = datetime.strptime(date_1200+' 00:00:00.00', '%Y-%m-%d %H:%M:%S.%f')
    
    # consider first_run_of_day
    
    if create_nc == True:
        
        ncfile = netCDF4.Dataset(output_folder_lm+'MOSAiC_ACA_Flight_'+ datee +ab+'_'+timee+'_EagleHawk_2Pixelrows_Radiances.nc', mode='w', format='NETCDF4')
        
        
        today = datetime.today()
        #create attributes
        ncfile.title = 'Combination of Aisa Eagle and Hawk spectrum'
        ncfile.subtitle = 'Radiance spectra along time averaged over pixels 232-280 for Eagle and 87-105 for Hawk, respectively 744-792 and 279-297'
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

        if centerrow == True:
            rad = ncfile.createVariable('rad', np.float32, ('wvl', 'time', ))
            rad.units = '10^{-3} W m^{-2} sr^{-1} nm^{-1}'
            rad.standard_name = 'Radiance' #upwelling_radiance_per_unit_wavelength_in_air
            rad.long_name = 'spectral uppward Radiance measured inflight by Aisa Eagle (pixels 488-536) and Hawk (pixels 183-201)'

        if toprow == True:
            rad1 = ncfile.createVariable('rad1', np.float32, ('wvl', 'time', ))
            rad1.units = '10^{-3} W m^{-2} sr^{-1} nm^{-1}'
            rad1.standard_name = 'Radiance' #upwelling_radiance_per_unit_wavelength_in_air
            rad1.long_name = 'spectral uppward Radiance measured inflight by Aisa Eagle (pixels 232-280) and Hawk (pixels 87-105)'
            
        if bottomrow == True:
            rad2 = ncfile.createVariable('rad2', np.float32, ('wvl', 'time', ))
            rad2.units = '10^{-3} W m^{-2} sr^{-1} nm^{-1}'
            rad2.standard_name = 'Radiance'
            rad2.long_name = 'spectral uppward Radiance measured inflight by Aisa Eagle (pixels 744-792) and Hawk (pixels 279-297)'
            
        
        #time_delta = time_arr[:] - datetime(1970, 1, 1, 0, 0, 0, 0)#date_datetime #datetime(1970, 1, 1, 0, 0, 0, 0) # time_arr[0] for seconds since flightstart
        time_delta = [i.timestamp() for i in time_arr[:]]
        #time_delta = np.asarray([i.total_seconds() for i in time_delta])
        #time_delta = np.float32(time_delta)
        
        #dtime = [str(i) for i in time_arr]
        #dtime = np.array(dtime, dtype='object')
        #dtime_str = netCDF4.stringtochar(np.array(dtime, 'S26'))

        print(bottom_arr_640.shape, bottom_arr_1200.shape)
        print(top_arr_640.shape, top_arr_1200.shape)
        
        # concatenat arrays from Eagle and Hawk (real joining part)
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
        
        # wavelengths
        b2 = np.float32(b2)
        
        
        # write data to variables
        time[:] = time_delta
        #dtime[:] = dtime
        wvl[:] = b2
        if centerrow == True:
            rad[:] == np.transpose(p_all_c)
        if bottomrow == True:
            rad2[:] = np.transpose(p_all_b)
        if toprow == True:
            rad1[:] = np.transpose(p_all_t)
        
        # plot stuff to check if it is correct 
        if createplot == True:
            create_plots(data_arr = rad, time_arr = time[:]-time[0], wavel_arr = wvl, times = times_to_plot, wavel = wvl_to_plot, date_of_flight=datee, time_of_flight=timee)
            create_plots(data_arr = rad, time_arr = time[:]-time[0], wavel_arr = wvl, times = times_to_plot, wavel = wvl_to_plot, date_of_flight=datee, time_of_flight=timee)
            create_plots(data_arr = rad, time_arr = time[:]-time[0], wavel_arr = wvl, times = times_to_plot, wavel = wvl_to_plot, date_of_flight=datee, time_of_flight=timee)

            
        ncfile.close()
        print('nc finished'+ str(num))
        
        

print(datetime.now()-runtime_start) # total runtime


