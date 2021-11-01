# -*- coding: utf-8 -*-
"""
Created on Sun Jan 24 15:05:49 2021

@author: Lukas Monrad-Krohn
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


flight = pd.read_table('C:/Users/Lukas Monrad-Krohn/Desktop/uni/shk/Mosaic ACA/Mosaic-ACA_flight02_20200830a/Polar5_20200830a.nav', skiprows=3, header=None, sep="\s+")
flight.columns = ["time", "Longitude", "Latitude", "Altitude", "Velocity", "Pitch", "Roll", 
                  "Yaw", "SZA", "SAA"]
flight['realtime']=pd.to_timedelta(flight['time'], unit='h')


file_name = 'C:/Users/Lukas Monrad-Krohn/Desktop/uni/shk/Mosaic ACA/Mosaic-ACA_flight02_20200830a/AMSR_U2_L3_SeaIce12km_B04_20200830.he5'
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
        label='flightpath 20200830a', color='orange', linewidth=2.5)

ax.coastlines('10m')
ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, dms=True, x_inline=False, y_inline=False,
             linewidth=1, color='black', alpha=0.7, linestyle='--')
ax.add_feature(cfeature.NaturalEarthFeature('physical', 'land', '10m', edgecolor='face', facecolor='darkgray'))


LYRlon, LYRlat = 15.50150, 78.24588
#plt.plot( LYRlon, LYRlat, marker='o', color='red', label='LYR', markersize=5,
#         transform =ccrs.Geodetic())
plt.scatter( LYRlon, LYRlat, color='red', label='LYR', linewidth=3, zorder=6,
         transform =ccrs.PlateCarree())



def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap

cmap=plt.get_cmap('Blues_r')
new_cmap = truncate_colormap(cmap, 0.5, 1)
#cs=ax.contourf(lon, lat, data, transform=ccrs.PlateCarree(), cmap=cmap, vmin=0, vmax=100)
cs = plt.pcolormesh(lon, lat, data, transform=ccrs.PlateCarree(), cmap=new_cmap, vmin=0, vmax=100)
#dataset.plot(transform=ccrs.PlateCarree(), cmap=cmap)

#cbar_ax = fig.add_axes([0.9, 0.1, 0.05, 0.8])
cbar = plt.colorbar(cs, orientation='vertical', pad=0.1)#, cax=cbar_ax 
cbar.set_label('ice concentration [%]')

lon_DS = [15.316667, 15.133333, 11.233333, 8.583333, 7.450000, 10.133333]
lat_DS = [79.816667, 81.116667, 81.333333, 81.450000, 80.900000, 79.850000]
plt.scatter(lon_DS, lat_DS, linewidth=3, color=['yellow', 'greenyellow', 'lawngreen', 'lime', 'green', 'darkgreen']
            , transform=ccrs.PlateCarree())


legend1 = plt.legend(handles=[Line2D([0], [0],  marker='o',color='w', markerfacecolor='yellow', markersize=10, label='DS1'),
                   mpatches.Patch(color='greenyellow', label='DS2'),
                   mpatches.Patch(color='lawngreen', label='DS3'),
                   mpatches.Patch(color='lime', label='DS4'),
                   mpatches.Patch(color='green', label='DS5'),
                   mpatches.Patch(color='darkgreen', label='DS6')],loc='upper right')


legend2 = plt.legend(loc='lower left')
ax.add_artist(legend1)
plt.title('Mosaic ACA, flight1_20200830a')
#plt.savefig('mosaicACA_flight1_20200830a', dpi=300)
plt.show()
