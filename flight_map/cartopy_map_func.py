# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 11:19:28 2021 

@author: Lukas Monrad-Krohn (lm73code@studserv.uni-leipzig.de / lukas@monrad-krohn.com)

Some code to create quicklooks for flightpaths from Longyearbyen, which contain sea ice concentration
and dropsondes.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import matplotlib.colors as colors
import cartopy
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import h5py
import xarray as xr





def mosaic_ACA_map(foldername, dataname, flightdate, flightnumber, lon_DS, lat_DS, leg_value, leg_loc='lower left'):
    
    
    #read nav-file ----------------------------------------------
    flight = pd.read_table(foldername + 'Polar5_'+ flightdate +'.nav', skiprows=3, header=None, sep="\s+")
    flight.columns = ["time", "Longitude", "Latitude", "Altitude", "Velocity", "Pitch", "Roll", 
                      "Yaw", "SZA", "SAA"]
    flight['realtime']=pd.to_timedelta(flight['time'], unit='h')
    
    #read he5 file ----------------------------------------------
    file_name = foldername + dataname
    path_sea_ice = "/HDFEOS/GRIDS/NpPolarGrid12km/Data Fields/SI_12km_NH_ICECON_DAY"
    path_lat = "/HDFEOS/GRIDS/NpPolarGrid12km/lat"
    path_lon = "/HDFEOS/GRIDS/NpPolarGrid12km/lon"
    
    with h5py.File(file_name, mode='r') as f: 
     # List available datasets.
        #print (f.keys())
        dset = f[path_sea_ice]
        data = dset[:]
    
        dset = f[path_lat]
        lat = dset[:]
        
        dset = f[path_lon]
        lon = dset[:]
        
    dataset = xr.DataArray(data, dims=["lon","lat"], coords = dict(LON=(["lon", "lat"], lon)
                                                                   , LAT=(["lon", "lat"], lat)))
    
    
    #creating map inctance--------------------------------------
    lcc_proj = ccrs.LambertConformal(central_latitude = 78, central_longitude = 15, 
                                     standard_parallels = (25, 25))
    
    
    # create figure and axis -----------------------------------
    fig = plt.figure(figsize=(10,8))
    ax = plt.axes(projection = lcc_proj)
    ax.set_extent([-7, 30, 75, 84])
    
    #plot flightpath
    ax.plot(flight['Longitude'], flight['Latitude'], transform = ccrs.PlateCarree(), 
            label=flightdate, color='orange', linewidth=2.5)
    
    #map features
    ax.coastlines('10m')
    ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, dms=True, x_inline=False, y_inline=False,
                 linewidth=1, color='black', alpha=0.7, linestyle='--')
    ax.add_feature(cfeature.NaturalEarthFeature('physical', 'land', '10m', edgecolor='face', facecolor='darkgray'))
    
    #Longyearbyen
    LYRlon, LYRlat = 15.50150, 78.24588
    plt.scatter( LYRlon, LYRlat, color='red', label='Longyearbyen', linewidth=3,
             transform =ccrs.PlateCarree(), zorder=5, marker='D')
    
    #make new colormap
    def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
        new_cmap = colors.LinearSegmentedColormap.from_list(
            'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
            cmap(np.linspace(minval, maxval, n)))
        return new_cmap
    cmap=plt.get_cmap('Blues_r')
    new_cmap = truncate_colormap(cmap, 0.5, 1)
    
    #plot sea ice data
    cs = plt.pcolormesh(lon, lat, data, transform=ccrs.PlateCarree(), cmap=new_cmap, vmin=0, vmax=100)
    
    cbar = plt.colorbar(cs, orientation='vertical', pad=0.1)
    cbar.set_label('Arctic sea ice concentration [%]')
    
    
    #plot dropsondes (dropsondes, colors and labels have to be added manually)
    plt.scatter(lon_DS, lat_DS, linewidth=2, color=['yellow', 'greenyellow', 'lawngreen', 'lime', 'green', 'darkgreen', 'darkslategray'], transform=ccrs.PlateCarree(), zorder=5)
    if 1==leg_value:
        legend1 = plt.legend(handles=[Line2D([0], [0],  marker='o',color='whitesmoke', markerfacecolor='yellow', markersize=10, label='DS1, 09:54 UTC'),
                        Line2D([0], [0],  marker='o',color='whitesmoke', markerfacecolor='greenyellow', markersize=10, label='DS2, 10:4 UTC'),
                        Line2D([0], [0],  marker='o',color='whitesmoke', markerfacecolor='lawngreen', markersize=10, label='DS3, 10:29 UTC'),
                        Line2D([0], [0],  marker='o',color='whitesmoke', markerfacecolor='lime', markersize=10, label='DS4, 10:47 UTC'),
                        Line2D([0], [0],  marker='o',color='whitesmoke', markerfacecolor='green', markersize=10, label='DS5, 11:01 UTC'),
                        Line2D([0], [0],  marker='o',color='whitesmoke', markerfacecolor='darkgreen', markersize=10, label='DS6, 12:55 UTC'),
                        Line2D([0], [0],  marker='o',color='whitesmoke', markerfacecolor='darkslategray', markersize=10, label='DS7, 14:16 UTC')],loc='upper right')
    #show both legends
    legend2 = plt.legend(loc='lower left')
    ax.add_artist(legend1)

    plt.title('MOSAiC-ACA, Flight '+flightnumber+', '+flightdate)
    plt.savefig('mosaicACA_flight'+flightnumber+'_'+flightdate, dpi=300)
    plt.show()
    
    
#%% same but without dropsondes #####################################################################################################

        #####
        #####
         ###
         ###
         ###
          #
          #
          
         ###
         ###

def mosaic_ACA_map_without_DS(foldername, dataname, flightdate, flightnumber, leg_value, leg_loc='lower left'):
    flight = pd.read_table(foldername + 'Polar5_'+ flightdate +'.nav', skiprows=3, header=None, sep="\s+")
    flight.columns = ["time", "Longitude", "Latitude", "Altitude", "Velocity", "Pitch", "Roll", 
                      "Yaw", "SZA", "SAA"]
    flight['realtime']=pd.to_timedelta(flight['time'], unit='h')
    
    
    file_name = foldername + dataname
    path_sea_ice = "/HDFEOS/GRIDS/NpPolarGrid12km/Data Fields/SI_12km_NH_ICECON_DAY"
    path_lat = "/HDFEOS/GRIDS/NpPolarGrid12km/lat"
    path_lon = "/HDFEOS/GRIDS/NpPolarGrid12km/lon"
    
    with h5py.File(file_name, mode='r') as f: 
     # List available datasets.
        #print (f.keys())
        dset = f[path_sea_ice]
        data = dset[:]
    
        dset = f[path_lat]
        lat = dset[:]
        
        dset = f[path_lon]
        lon = dset[:]
        
    dataset = xr.DataArray(data, dims=["lon","lat"], coords = dict(LON=(["lon", "lat"], lon)
                                                                   , LAT=(["lon", "lat"], lat)))
    
    
    
    lcc_proj = ccrs.LambertConformal(central_latitude = 78, central_longitude = 15, 
                                     standard_parallels = (25, 25))
    
    
    fig = plt.figure(figsize=(10,8))
    ax = plt.axes(projection = lcc_proj)
    #ax = fig.add_subplot(1, 1, 1, projection=lcc_proj)
    ax.set_extent([-7, 30, 75, 84])
    
    ax.plot(flight['Longitude'], flight['Latitude'], transform = ccrs.PlateCarree(), 
            label=flightdate, color='orange', linewidth=2.5)
    
    ax.coastlines('10m')
    ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, dms=True, x_inline=False, y_inline=False,
                 linewidth=1, color='black', alpha=0.7, linestyle='--')
    ax.add_feature(cfeature.NaturalEarthFeature('physical', 'land', '10m', edgecolor='face', facecolor='darkgray'))
    
    
    LYRlon, LYRlat = 15.50150, 78.24588
    plt.scatter( LYRlon, LYRlat, color='red', label='Longyearbyen', linewidth=3,
             transform =ccrs.PlateCarree(), zorder=5, marker='D')
    
    def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
        new_cmap = colors.LinearSegmentedColormap.from_list(
            'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
            cmap(np.linspace(minval, maxval, n)))
        return new_cmap
    cmap=plt.get_cmap('Blues_r')
    new_cmap=truncate_colormap(cmap, 0.5, 1)
    cs = plt.pcolormesh(lon, lat, data, transform=ccrs.PlateCarree(), cmap=new_cmap, vmin=0, vmax=100)
    
    cbar = plt.colorbar(cs, orientation='vertical', pad=0.1)
    cbar.set_label('Arctic sea ice concentration [%]')

    legend = plt.legend(loc='lower left')

    plt.title('MOSAiC-light '+flightnumber+', '+flightdate)
    plt.savefig('mosaicACA_flight'+flightnumber+'_'+flightdate, dpi=300)
    plt.show()


        #####
        #####
         ###
         ###
         ###
          #
          #
          
         ###
         ###
         

#%% performing function for all MOSAiC-ACA flights

foldername='C:/Users/Lukas Monrad-Krohn/Desktop/uni/shk/Mosaic ACA/Mosaic-ACA_flight02_20200830a/'
flightdate='20200830a'
flightnumber= '02'
dataname= 'AMSR_U2_L3_SeaIce12km_B04_20200830.he5'

mosaic_ACA_map_without_DS(foldername, dataname, flightdate, flightnumber, leg_value=0)

#%%
foldername='C:/Users/Lukas Monrad-Krohn/Desktop/uni/shk/Mosaic ACA/Mosaic-ACA_flight03_20200831a/'
flightdate='20200831a'
flightnumber= '03'
dataname= 'AMSR_U2_L3_SeaIce12km_B04_20200831.he5'

mosaic_ACA_map_without_DS(foldername, dataname, flightdate, flightnumber, leg_value=0)

#%%
foldername='C:/Users/Lukas Monrad-Krohn/Desktop/uni/shk/Mosaic ACA/Mosaic-ACA_flight04_20200831b/'
flightdate='20200831b'
flightnumber= '04'
dataname= 'AMSR_U2_L3_SeaIce12km_B04_20200831.he5'

mosaic_ACA_map_without_DS(foldername, dataname, flightdate, flightnumber, leg_value=0)

#%% flight 05

foldername='C:/Users/Lukas Monrad-Krohn/Desktop/uni/shk/Mosaic ACA/Mosaic-ACA_flight05_20200902/'
flightdate='20200902a'
flightnumber= '05'
dataname= 'AMSR_U2_L3_SeaIce12km_B04_20200902.he5'
lon_DS = [15.316667, 15.133333, 11.233333, 8.583333, 7.450000, 10.133333]
lat_DS = [79.816667, 81.116667, 81.333333, 81.450000, 80.900000, 79.850000]


mosaic_ACA_map(foldername, dataname, flightdate, flightnumber, lon_DS, lat_DS, leg_value=1)

#plt.legend(handles=[Line2D([0], [0],  marker='o',color='whitesmoke', markerfacecolor='yellow', markersize=10, label='DS1, 07:31 UTC'),
#                        Line2D([0], [0],  marker='o',color='whitesmoke', markerfacecolor='greenyellow', markersize=10, label='DS2, 08:20 UTC'),
#                        Line2D([0], [0],  marker='o',color='lightgray', markerfacecolor='lawngreen', markersize=10, label='DS3, 08:37 UTC'),
#                        Line2D([0], [0],  marker='o',color='lightgray', markerfacecolor='lime', markersize=10, label='DS4, 08:48 UTC'),
#                        Line2D([0], [0],  marker='o',color='lightgray', markerfacecolor='green', markersize=10, label='DS5, 11:01 UTC'),
#                        Line2D([0], [0],  marker='o',color='lightgray', markerfacecolor='darkgreen', markersize=10, label='DS6, 11:30 UTC')],loc='upper right')

#%% flight 06

foldername='C:/Users/Lukas Monrad-Krohn/Desktop/uni/shk/Mosaic ACA/Mosaic-ACA_flight06_20200904/'
flightdate='20200904a'
flightnumber= '06'
dataname= 'AMSR_U2_L3_SeaIce12km_B04_20200904.he5'
lon_DS = [20.000000, 21.283333, 22.433333, 23.916667, 20.633333]
lat_DS = [76.983333, 76.683333, 76.400000, 76.016667, 77.133333]


mosaic_ACA_map(foldername, dataname, flightdate, flightnumber, lon_DS, lat_DS, leg_value=1)

#plt.legend(handles=[Line2D([0], [0],  marker='o',color='whitesmoke', markerfacecolor='yellow', markersize=10, label='DS1, 12:54 UTC'),
#                        Line2D([0], [0],  marker='o',color='whitesmoke', markerfacecolor='greenyellow', markersize=10, label='DS2, 13:05 UTC'),
#                        Line2D([0], [0],  marker='o',color='lightgray', markerfacecolor='lawngreen', markersize=10, label='DS3, 13:16 UTC'),
#                        Line2D([0], [0],  marker='o',color='lightgray', markerfacecolor='lime', markersize=10, label='DS4, 13:30 UTC'),
#                        Line2D([0], [0],  marker='o',color='lightgray', markerfacecolor='green', markersize=10, label='DS5, 16:06 UTC')],loc='upper right')

#%% flight 07

foldername='C:/Users/Lukas Monrad-Krohn/Desktop/uni/shk/Mosaic ACA/Mosaic-ACA_flight07_20200907/'
flightdate='20200907a'
flightnumber= '07'
dataname= 'AMSR_U2_L3_SeaIce12km_B04_20200907.he5'
lon_DS = [9.012446, 9.000853, 8.997032, 9.001439, 9.009037, 0.997094, 0.999869, 1.000790, 1.001047, 0.314518, 1.000744, 1.001237, 5.135990, 9.019563]
lat_DS = [79.363109, 79.977231, 81.004137, 82.001098, 82.975490, 82.909310, 81.993068, 80.991554, 79.996393, 79.530145, 78.992187, 78.515191, 78.338665, 78.366123]


mosaic_ACA_map(foldername, dataname, flightdate, flightnumber, lon_DS, lat_DS, leg_value=1)

'''plt.legend(handles=[Line2D([0], [0],  marker='o',color='whitesmoke', markerfacecolor='yellow', markersize=10, label='DS1, 09:05 UTC'),
                        Line2D([0], [0],  marker='o',color='whitesmoke', markerfacecolor='greenyellow', markersize=10, label='DS2, 09:18 UTC'),
                        Line2D([0], [0],  marker='o',color='whitesmoke', markerfacecolor='lawngreen', markersize=10, label='DS3, 09:42 UTC'),
                        Line2D([0], [0],  marker='o',color='whitesmoke', markerfacecolor='lime', markersize=10, label='DS4, 10:04 UTC'),
                        Line2D([0], [0],  marker='o',color='whitesmoke', markerfacecolor='green', markersize=10, label='DS5, 10:26 UTC'),
                        Line2D([0], [0],  marker='o',color='whitesmoke', markerfacecolor='darkgreen', markersize=10, label='DS6, 10:51 UTC'),
                        Line2D([0], [0],  marker='o',color='whitesmoke', markerfacecolor='mediumaquamarine', markersize=10, label='DS7, 11:14 UTC'),
                        Line2D([0], [0],  marker='o',color='whitesmoke', markerfacecolor='aquamarine', markersize=10, label='DS8, 11:39 UTC'),
                        Line2D([0], [0],  marker='o',color='whitesmoke', markerfacecolor='lightseagreen', markersize=10, label='DS9, 12:04 UTC'),
                        Line2D([0], [0],  marker='o',color='whitesmoke', markerfacecolor='teal', markersize=10, label='DS10, 12:18 UTC'),
                        Line2D([0], [0],  marker='o',color='whitesmoke', markerfacecolor='darkslategray', markersize=10, label='DS11, 12:31 UTC'),
                        Line2D([0], [0],  marker='o',color='whitesmoke', markerfacecolor='darkviolet', markersize=10, label='DS12, 12:43 UTC'),
                        Line2D([0], [0],  marker='o',color='whitesmoke', markerfacecolor='purple', markersize=10, label='DS13, 13:09 UTC'),
                        Line2D([0], [0],  marker='o',color='whitesmoke', markerfacecolor='magenta', markersize=10, label='DS14, 13:27 UTC')
                        ],loc='upper right')'''

#['yellow', 'greenyellow', 'lawngreen', 'lime', 'green', 'darkgreen', 'mediumaquamarine', 'aquamarine', 'lightseagreen', 'teal', 'darkslategray', 'darkviolet', 'purple', 'magenta']


#%% flight 08
foldername='C:/Users/Lukas Monrad-Krohn/Desktop/uni/shk/Mosaic ACA/Mosaic-ACA_flight08_20200908/'
flightdate='20200908a'
flightnumber= '08'
dataname= 'AMSR_U2_L3_SeaIce12km_B04_20200908.he5'
lon_DS = [6.882757, 7.000606, 7.001099, 7.002777, 6.994147, 6.999840, 7.002271]
lat_DS = [79.635708, 80.171457, 81.005413, 82.010791, 81.286948, 80.482405, 79.994916]


mosaic_ACA_map(foldername, dataname, flightdate, flightnumber, lon_DS, lat_DS, leg_value=1)

'''plt.legend(handles=[Line2D([0], [0],  marker='o',color='whitesmoke', markerfacecolor='yellow', markersize=10, label='DS1, 08:49 UTC'),
                        Line2D([0], [0],  marker='o',color='whitesmoke', markerfacecolor='greenyellow', markersize=10, label='DS2, 09:00 UTC'),
                        Line2D([0], [0],  marker='o',color='whitesmoke', markerfacecolor='lawngreen', markersize=10, label='DS3, 09:15 UTC'),
                        Line2D([0], [0],  marker='o',color='whitesmoke', markerfacecolor='lime', markersize=10, label='DS4, 09:34 UTC'),
                        Line2D([0], [0],  marker='o',color='whitesmoke', markerfacecolor='green', markersize=10, label='DS5, 12:46 UTC'),
                        Line2D([0], [0],  marker='o',color='whitesmoke', markerfacecolor='darkgreen', markersize=10, label='DS6, 12:57 UTC'),
                        Line2D([0], [0],  marker='o',color='whitesmoke', markerfacecolor='darkslategray', markersize=10, label='DS7, 13:10 UTC')],loc='upper right')'''


#%% flight 09
foldername='C:/Users/Lukas Monrad-Krohn/Desktop/uni/shk/Mosaic ACA/Mosaic-ACA_flight09_20200910/'
flightdate='20200910a'
flightnumber= '09'
dataname= 'AMSR_U2_L3_SeaIce12km_B04_20200910.he5'
lon_DS = [7.225092, 6.028523, 4.771415, 3.979864, 2.960555, 3.510101, 8.631769, 10.055495, 9.526059, 9.135779, 8.958008]
lat_DS = [79.589973, 80.503718, 81.448945, 81.958985, 82.461939, 82.601879, 82.159505, 82.010970, 81.011372, 79.992230, 79.446489]

mosaic_ACA_map(foldername, dataname, flightdate, flightnumber, lon_DS, lat_DS, leg_value=1)

'''plt.legend(handles=[Line2D([0], [0],  marker='o',color='whitesmoke', markerfacecolor='yellow', markersize=10, label='DS1, 09:14 UTC'),
                        Line2D([0], [0],  marker='o',color='whitesmoke', markerfacecolor='greenyellow', markersize=10, label='DS2, 09:32 UTC'),
                        Line2D([0], [0],  marker='o',color='whitesmoke', markerfacecolor='lawngreen', markersize=10, label='DS3, 09:50 UTC'),
                        Line2D([0], [0],  marker='o',color='whitesmoke', markerfacecolor='lime', markersize=10, label='DS4, 09:59 UTC'),
                        Line2D([0], [0],  marker='o',color='whitesmoke', markerfacecolor='green', markersize=10, label='DS5, 10:11 UTC'),
                        Line2D([0], [0],  marker='o',color='whitesmoke', markerfacecolor='darkgreen', markersize=10, label='DS6, 11:21 UTC'),
                        Line2D([0], [0],  marker='o',color='whitesmoke', markerfacecolor='mediumaquamarine', markersize=10, label='DS7, 11:41 UTC'),
                        Line2D([0], [0],  marker='o',color='whitesmoke', markerfacecolor='aquamarine', markersize=10, label='DS8, 13:09 UTC'),
                        Line2D([0], [0],  marker='o',color='whitesmoke', markerfacecolor='lightseagreen', markersize=10, label='DS9, 13:31 UTC'),
                        Line2D([0], [0],  marker='o',color='whitesmoke', markerfacecolor='teal', markersize=10, label='DS10, 13:52 UTC'),
                        Line2D([0], [0],  marker='o',color='whitesmoke', markerfacecolor='darkslategray', markersize=10, label='DS11, 14:04 UTC')],loc='upper right')'''
    
#%% flight 10
foldername='C:/Users/Lukas Monrad-Krohn/Desktop/uni/shk/Mosaic ACA/Mosaic-ACA_flight10_20200911/'
flightdate='20200911a'
flightnumber= '10'
dataname= 'AMSR_U2_L3_SeaIce12km_B04_20200911.he5'
lon_DS = [14.488490, 13.203516, 11.293642, 9.511504, 7.180367, 4.180358, 3.562314, 10.450077, 12.608221, 14.680618]
lat_DS = [80.149077, 80.502382, 80.974504, 81.367123, 81.820998, 82.323109, 82.544556, 81.166822, 80.656324, 80.093829]

mosaic_ACA_map(foldername, dataname, flightdate, flightnumber, lon_DS, lat_DS, leg_value=1)
    
''' plt.legend(handles=[Line2D([0], [0],  marker='o',color='whitesmoke', markerfacecolor='yellow', markersize=10, label='DS1, 09:02 UTC'),
                        Line2D([0], [0],  marker='o',color='whitesmoke', markerfacecolor='greenyellow', markersize=10, label='DS2, 09:09 UTC'),
                        Line2D([0], [0],  marker='o',color='whitesmoke', markerfacecolor='lawngreen', markersize=10, label='DS3, 09:20 UTC'),
                        Line2D([0], [0],  marker='o',color='whitesmoke', markerfacecolor='lime', markersize=10, label='DS4, 09:28 UTC'),
                        Line2D([0], [0],  marker='o',color='whitesmoke', markerfacecolor='green', markersize=10, label='DS5, 09:39 UTC'),
                        Line2D([0], [0],  marker='o',color='whitesmoke', markerfacecolor='darkgreen', markersize=10, label='DS6, 09:50 UTC'),
                        Line2D([0], [0],  marker='o',color='whitesmoke', markerfacecolor='mediumaquamarine', markersize=10, label='DS7, 11:24 UTC'),
                        Line2D([0], [0],  marker='o',color='whitesmoke', markerfacecolor='aquamarine', markersize=10, label='DS8, 12:52 UTC'),
                        Line2D([0], [0],  marker='o',color='whitesmoke', markerfacecolor='lightseagreen', markersize=10, label='DS9, 13: U05TC'),
                        Line2D([0], [0],  marker='o',color='whitesmoke', markerfacecolor='teal', markersize=10, label='DS10, 13:18 UTC')],loc='upper right')'''
#%% flight 11
foldername='C:/Users/Lukas Monrad-Krohn/Desktop/uni/shk/Mosaic ACA/Mosaic-ACA_flight11_20200913/'
flightdate='20200913a'
flightnumber= '11'
dataname= 'AMSR_U2_L3_SeaIce12km_B04_20200913.he5'
lon_DS = [9.992434, 4.916209, 0.952405, -4.054259, -8.010435, -0.002965, 5.017547]
lat_DS = [79.141523, 79.498201, 79.707701, 79.901990, 79.999693, 79.789501, 79.469468]

mosaic_ACA_map(foldername, dataname, flightdate, flightnumber, lon_DS, lat_DS, leg_value=1)
    
'''plt.legend(handles=[Line2D([0], [0],  marker='o',color='whitesmoke', markerfacecolor='yellow', markersize=10, label='DS1, 09:54 UTC'),
                        Line2D([0], [0],  marker='o',color='whitesmoke', markerfacecolor='greenyellow', markersize=10, label='DS2, 10:4 UTC'),
                        Line2D([0], [0],  marker='o',color='whitesmoke', markerfacecolor='lawngreen', markersize=10, label='DS3, 10:29 UTC'),
                        Line2D([0], [0],  marker='o',color='whitesmoke', markerfacecolor='lime', markersize=10, label='DS4, 10:47 UTC'),
                        Line2D([0], [0],  marker='o',color='whitesmoke', markerfacecolor='green', markersize=10, label='DS5, 11:01 UTC'),
                        Line2D([0], [0],  marker='o',color='whitesmoke', markerfacecolor='darkgreen', markersize=10, label='DS6, 12:55 UTC'),
                        Line2D([0], [0],  marker='o',color='whitesmoke', markerfacecolor='darkslategray', markersize=10, label='DS7, 14:16 UTC')],loc='upper right')
   '''
    
    
    
    
    
    

