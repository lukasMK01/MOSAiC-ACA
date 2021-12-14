# -*- coding: utf-8 -*-
"""
Created on Sat Jul 31 11:27:28 2021

@author: Lukas Monrad-Krohn (lm73code@studserv.uni-leipzig.de / lukas@monrad-krohn.com)

Just another version of making the arrays, with having different wavelengths in mind.
Or add lines showing the areas used for the mean to join Eagle and Hawk.
"""

from spectral import *
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from datetime import timedelta
import glob
import matplotlib.dates as mdates

runtime_start = datetime.now()

#Input folder
input_folder = "/projekt_agmwend/data/MOSAiC_ACA_S/Flight_20200910a/AisaHAWK/"

#wavelength(s)
wvl = [1100] 

#Output folder
output_folder= '/home/lmkrohn/spec_img/test_arr/'

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


# because 1650 nm is in Hawk
list_of_files = sorted(glob.glob(input_folder+"*.hdr"))
#list_of_files = [list_of_files[1]]
#list_of_files.append('stop')

for i in range(len(list_of_files)):
    filename = list_of_files[i][-52:-4]
    print(filename)
    img = open_image(list_of_files[i])
    bands = np.asarray(img.bands.centers)
    
    arr= img.asarray()
    
    date = img.metadata['acquisition date'][-10:]
    date = date[-4:]+"-"+date[-7:-5]+"-"+date[-10:-8]
    start_time = img.metadata['gps start time'][-13:-1]
    
    starttime = datetime.fromisoformat(date+" "+start_time)
    time_ls  = []
    for i in range(0,len(arr)):
        time_ls.append(starttime + i* timedelta(seconds=0.05))
    time = np.asarray(time_ls)
    
    print(time[0], time[-1])
    
    
      
    #find nearest wavelength
    idx = find_nearest(bands,wvl)
    print(bands[idx], idx)
    
    print(len(time),np.shape(arr))
    
    _,ax = plt.subplots(1,1, figsize=(12,4))#round(len(time)/384)
    contour = ax.contourf(time,np.arange(0,384),arr[:,:,idx].T,25,cmap="Greys_r")

    ax.set_xlabel("Time (UTC)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
    #ax.tick_params(axis='x', rotation=45)
    ax.title.set_text('AISA Hawk ('+str(bands[idx])+' nm), '+str(starttime))
    ax.set_yticks(np.arange(0,384,384/4))
    ax.set_ylabel("across-track-pixel")
    ax.grid(False)
    
    # plot vertical lines showing the pixels averaged in line_plot (for top and bottom in this case)
    ax[1].plot([time[0], time[-1]], [87, 87], color = 'red', linestyle = '--', linewidth = 1)
    ax[1].plot([time[0], time[-1]], [105, 105], color = 'red', linestyle = '--', linewidth = 1)
    
    ax[0].plot([time[0], time[-1]], [232, 232], color = 'red', linestyle = '--', linewidth = 1)
    ax[0].plot([time[0], time[-1]], [280, 280], color = 'red', linestyle = '--', linewidth = 1)
    
    ax[1].plot([time[0], time[-1]], [279, 279], color = 'red', linestyle = '--', linewidth = 1)
    ax[1].plot([time[0], time[-1]], [297, 297], color = 'red', linestyle = '--', linewidth = 1)
    
    ax[0].plot([time[0], time[-1]], [744, 744], color = 'red', linestyle = '--', linewidth = 1)
    ax[0].plot([time[0], time[-1]], [792, 792], color = 'red', linestyle = '--', linewidth = 1)
    
    
    #cbar = fig.colorbar(c1)
    #cbar.ax.set_ylabel('Radiance')
    plt.savefig(output_folder+filename+'_arr_%.fnm.png' %bands[idx],dpi=300,bbox_inches="tight")
    plt.show()
    #plt.close()


print(datetime.now() - runtime_start)






