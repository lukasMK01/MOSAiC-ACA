# -*- coding: utf-8 -*-
"""
Created on Sun Apr 18 11:54:00 2021

@author: Lukas Monrad-Krohn (lm73code@studserv.uni-leipzig.de / lukas@monrad-krohn.com)

This python script can be used to plot different aspects of AisaEAGLE and AisaHAWK spectral
imager data.
It is based on a script by Marcus Klinebiel (marcus.klingebiel@uni-leipzig.de) and is the easier version 
of the join_EAGLE_HAWK script.
"""
from spectral import *
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from datetime import timedelta
import glob
import matplotlib.dates as mdates



#Input folder
#input_folder = "C:/Users/Lukas Monrad-Krohn/Desktop/uni/shk/spectral_imager/example/"

#wavelength(s)
wvl1 = [640] # between 400 und 993nm
wvl2 = [1200] #between 931 und 2544nm
wvl = [640, 1200]

#Output folder
output_folder= '/home/lmkrohn/spec_img/'



#Plots settings
PlotImage = False
PlotArray = True
PlotSpec  = True
#-------------------------------------------

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

#--------------------------------------------



# because I mostly used this for visualization and never used it for a lot of different dataset
list_of_files = ['/projekt_agmwend/data/MOSAiC_ACA_S/Flight_20200910a/AisaEAGLE/MOSAiC_ACA_Flight_20200910a_0910-1014_radiance.hdr',
                 '/projekt_agmwend/data/MOSAiC_ACA_S/Flight_20200910a/AisaHAWK/MOSAiC_ACA_Flight_20200910a_0910-1015-1_radiance.hdr']
#1200 first, 640 second
print(list_of_files)

#--------------------------------------


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

# k and z are used to cut both arrays to the same start and end time
k = int(np.round((float(start_640[-9:]) - float(start_1200[-9:])) / 0.05 * (-1)))

z = int(np.round((float(end_640[-9:]) - float(end_1200[-9:])) / 0.05 * (-1)))

print(k, z)
#trial and error
arr_640 = arr_640[k:z,:,:]

# only work if arr_640 is longer on both ends
# watch out for problems here ! might need manual correction or add the stuff from join_EAGLE_HAWK.py

# with trial and error, find out which start and end time to use, then put it here 
# and in arr above


# time formatting for the plots
if PlotArray == True:
    starttime = datetime.fromisoformat(date_1200+" "+start_time_1200)
    print(starttime)

    time_ls  = []

    for i in range(0,len(arr_1200)):
        time_ls.append(starttime + i* timedelta(seconds=0.05))
    time = np.asarray(time_ls)
    
    
    
idx = []   
call_image = [img_640, img_1200]
#Plot picture for all wavelengths
for j in range(len(wvl)):
    #find nearest wavelength
    idx.append(find_nearest(b[j],wvl[j]))
    
    print(b[j][idx[j]], idx[j])
    
    # image in grayscale or color
    if PlotImage == True:
        save_rgb(output_folder+filename_640+"_both"+'.jpg', call_image[j]
                 , [idx[j], idx[j] , idx[j]])
  

c_list = ['lightcoral', 'darkorange', 'seagreen', 'teal', 'orchid', 'sandybrown',
          'olive', 'navy', 'olive']
#Plot array (array = means the Raciances for one wavelength over time and all across-track-pixels)
if PlotArray == True:
    print(len(time),np.shape(arr_640), np.shape(arr_1200))
    _,ax = plt.subplots(2,1, figsize=(12,4), sharex=True)#round(len(time)/384)
    c2 = ax[1].contourf(time,np.arange(0,384),arr_1200[:,:,idx[1]].T,25,cmap="Greys_r")
    c1 = ax[0].contourf(time,np.arange(0,1024),arr_640[:,:,idx[0]].T,25,cmap="Greys_r")
    # cmap is reversed Greys so that, high values (e.g. clouds, ice) are white and low values dark

    #find points for spectra (these are the locations for which the spectra will be made)
    if PlotSpec == True:
        round(len(arr_1200)/5)
        ls_spec_pts = []
        for i in range(1,5):
            ls_spec_pts.append((i*round(len(arr_640)/5)))

        for i in range(len(ls_spec_pts)):
            ax[1].scatter(time[ls_spec_pts[i]],384/2, color = c_list[i])
            ax[0].scatter(time[ls_spec_pts[i]],1024/2, color = c_list[i])
    ax[1].set_xlabel("Time (UTC)")
    ax[1].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
    #ax.tick_params(axis='x', rotation=45)
    ax[0].title.set_text('AISA Eagle (640 nm)')
    ax[1].title.set_text('AISA Hawk (1200 nm)')
    ax[1].set_yticks(np.arange(0,384,384/4))
    ax[1].set_ylabel("across-track-pixel")
    ax[0].set_yticks(np.arange(0,1024,1024/4))
    ax[0].set_ylabel("across-track-pixel")
    
    #cbar = fig.colorbar(c1)
    #cbar.ax.set_ylabel('Radiance')
    plt.savefig(output_folder+filename_640+'_both_arr.png',dpi=300,bbox_inches="tight")
    plt.show()
    plt.close()


#Plot spectrum (Radiances over Wavelength)
if PlotSpec == True:
    fig = plt.figure(figsize=(10,7))
    for i in range(len(ls_spec_pts)):
        plt.plot(b[1],arr_1200[ls_spec_pts[i],192,:]/100*0.01, color = c_list[i])
        plt.plot(b[0],arr_640[ls_spec_pts[i],512,:]/100*0.01, color = c_list[i])
    plt.grid(linestyle=":")
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Radiance ($W~m^{-2}~sr^{-1}~nm^{-1}$)")
    plt.savefig(output_folder+filename_640+'_both_spec.png',dpi=300,bbox_inches="tight")
    plt.show()
    plt.close()


# in others of the python files there are also ways to plot the Radiances for certain wavelenths over time