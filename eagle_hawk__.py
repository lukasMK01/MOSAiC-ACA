# -*- coding: utf-8 -*-
"""
Created on Sat Apr 17 21:41:46 2021

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
#----------------------------------------------

#Input folder
input_folder = "C:/Users/Lukas Monrad-Krohn/Desktop/uni/shk/spectral_imager/example/"

#wavelength(s)
wvl = [640] #between 400 und 993nm

#Output folder
output_folder= "test_result"


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

#list_of_files = sorted(glob.glob(input_folder+"*.hdr"))
list_of_files = ['C:/Users/Lukas Monrad-Krohn/Desktop/uni/shk/spectral_imager/example/MOSAiC_ACA_Flight_20200911a_0911-1122.hdr']
print(list_of_files)

#--------------------------------------

for i in tqdm(range(0,len(list_of_files))):
    filename = list_of_files[i][-41:-4]
    print(filename)
    img = open_image(list_of_files[i])
    b = img.bands.centers
    arr= img.asarray()
    arr = arr[159:-111,:,:]
    
    #Get time
    date = img.metadata['acquisition date'][-10:]
    date = date[-4:]+"-"+date[-7:-5]+"-"+date[-10:-8]
    start_time = '11:22:41.043'#img.metadata['gps start time'][-13:-1]
    if PlotArray == True:
        starttime = datetime.fromisoformat(date+" "+start_time)
        print(starttime)
    
        time_ls  = []

        for i in range(0,len(arr)):
            time_ls.append(starttime + i* timedelta(seconds=0.05))
        time = np.asarray(time_ls)
        
        
    #Plot picture for all wavelengths
    for j in range(len(wvl)):
        #find nearest wavelength
        idx = find_nearest(b,wvl[j])
        
        print(b[idx], idx)
        
        if PlotImage == True:
            save_rgb(output_folder+filename+"_"+str(wvl[j])+'.jpg', img, [idx, idx , idx])
      

        #Plot array
        if PlotArray == True:
            print(len(time),np.shape(arr))
            _,ax = plt.subplots(figsize=(10,1))#round(len(time)/1024)
            ax.contourf(time,np.arange(0,1024),arr[:,:,idx].T,25,cmap="Greys_r")
        
            
            #find points for spectra
            if PlotSpec == True:
                round(len(arr)/5)
                ls_spec_pts = []
                for i in range(1,5):
                    ls_spec_pts.append((i*round(len(arr)/5)))

                for i in range(len(ls_spec_pts)):
                    ax.scatter(time[ls_spec_pts[i]],1024/2)
            ax.set_xlabel("Time (UTC)")
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
            ax.tick_params(axis='x', rotation=45)
            ax.set_yticks(np.arange(0,1024,1024/4))
            ax.set_ylabel("across-track-pixel")
            plt.show()
            plt.savefig(output_folder+filename+"_"+str(wvl[j])+'_arr.png',dpi=300,bbox_inches="tight")
            plt.close()


        #Plot spectrum
        if PlotSpec == True:
            for i in range(len(ls_spec_pts)):
                plt.plot(b,arr[ls_spec_pts[i],500,:]/100*0.01)
            plt.grid(linestyle=":")
            plt.xlabel("Wavelength (nm)")
            plt.ylabel("Radiance ($W~m^{-2}~sr^{-1}~nm^{-1}$)")
            plt.show()
            plt.savefig(output_folder+filename+"_"+str(wvl[j])+'_spec.png',dpi=300,bbox_inches="tight")
            plt.close()

#%% hawk


from spectral import *
import numpy as np
from tqdm import tqdm_notebook as tqdm
import matplotlib.pyplot as plt
from datetime import datetime
from datetime import timedelta
import glob
import matplotlib.dates as mdates



#Input folder
input_folder = "C:/Users/Lukas Monrad-Krohn/Desktop/uni/shk/spectral_imager/example/"

#wavelength(s)
wvl = [1200] #between 931 und 2544nm

#Output folder
output_folder= 'test_result'
#"test_result/"


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

#list_of_files = sorted(glob.glob(input_folder+"*.hdr"))
list_of_files = ['C:/Users/Lukas Monrad-Krohn/Desktop/uni/shk/spectral_imager/example/MOSAiC_ACA_Flight_20200911a_0911-1122-1.hdr']
print(list_of_files)

#--------------------------------------

for i in tqdm(range(0,len(list_of_files))):
    filename = list_of_files[i][-43:-4]
    print(filename)
    img = open_image(list_of_files[i])
    b = img.bands.centers
    arr= img.asarray()
    
    #Get time
    date = img.metadata['acquisition date'][-10:]
    date = date[-4:]+"-"+date[-7:-5]+"-"+date[-10:-8]
    start_time = img.metadata['gps start time'][-13:-1]
    if PlotArray == True:
        starttime = datetime.fromisoformat(date+" "+start_time)
        print(starttime)
    
        time_ls  = []

        for i in range(0,len(arr)):
            time_ls.append(starttime + i* timedelta(seconds=0.05))
        time = np.asarray(time_ls)
        
        
    #Plot picture for all wavelengths
    for j in range(len(wvl)):
        #find nearest wavelength
        idx = find_nearest(b,wvl[j])
        
        print(b[idx], idx)
        
        if PlotImage == True:
            save_rgb(output_folder+filename+"_"+str(wvl[j])+'.jpg', img, [idx, idx , idx])
      

        #Plot array
        if PlotArray == True:
            print(len(time),np.shape(arr))
            _,ax = plt.subplots(figsize=(10,1))#round(len(time)/384)
            ax.contourf(time,np.arange(0,384),arr[:,:,idx].T,25,cmap="Greys_r")
        
            
            #find points for spectra
            if PlotSpec == True:
                round(len(arr)/5)
                ls_spec_pts = []
                for i in range(1,5):
                    ls_spec_pts.append((i*round(len(arr)/5)))

                for i in range(len(ls_spec_pts)):
                    ax.scatter(time[ls_spec_pts[i]],384/2)
            ax.set_xlabel("Time (UTC)")
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
            ax.tick_params(axis='x', rotation=45)
            ax.set_yticks(np.arange(0,384,384/4))
            ax.set_ylabel("across-track-pixel")
            plt.show()
            #plt.savefig(output_folder+filename+"_"+str(wvl[j])+'_arr.png',dpi=300,bbox_inches="tight")
            plt.close()


        #Plot spectrum
        if PlotSpec == True:
            for i in range(len(ls_spec_pts)):
                plt.plot(b,arr[ls_spec_pts[i],200,:]/100*0.01)
            plt.grid(linestyle=":")
            plt.xlabel("Wavelength (nm)")
            plt.ylabel("Radiance ($W~m^{-2}~sr^{-1}~nm^{-1}$)")
            plt.show()
            plt.savefig(output_folder+filename+"_"+str(wvl[j])+'_spec.png',dpi=300,bbox_inches="tight")
            plt.close()
