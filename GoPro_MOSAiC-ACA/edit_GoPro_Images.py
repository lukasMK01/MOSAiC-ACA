'''
This program transforms raw GoPro images from the LIM Airborne campaigns into 
images with a smaller file size (depending on the output quality). In addition, 
it changes the filename to the timestamp when the picture was taken and it adds 
the timestamp on the picture if you wish. If neccessary, the picture can be rotated.


Marcus Klingebiel
marcus.klingebiel@uni-leipzig.de

Furthermore it adds a map of the flightroute and the altitude profile of the flight
and shows the current altitude. The individual images are then written into 
a timelaps video of the flight.

Use the following linux command to join the images to a video (there is also a method
with cv2, which can be 'activated' in this script but ffmpeg is better):
ffmpeg -r 24 -pattern_type glob -i '*.jpg' -s hd1080 -vcodec libx264 name.mp4

Lukas Monrad-Krohn
lm74code@studserv.uni-leipzig.de
'''



##### Settings ##############################################################################
input_folder = "/projekt_agmwend/data_raw/MOSAiC_ACA_S_raw/Flight_20200910a/GoPro_20200910a/"
output_folder= "/projekt_agmwend/data/MOSAiC_ACA_S/Flight_20200910a/GoPro_images/"#"/home/lmkrohn/GoPro/"#not planned to save the images

output_folder_video =  '/projekt_agmwend/data/MOSAiC_ACA_S/Flight_20200910a/'
output_folder_image = '/projekt_agmwend/data/MOSAiC_ACA_S/Flight_20200910a/GoPro_map_alt_images/'
prefix = "Flight_"

sic_folder = "/home/lmkrohn/GoPro/"
sic_name = "AMSR_U2_L3_SeaIce12km_B04_20200910.he5"
flightdate = "20200910a"
flightnumber = "09"

output_quality= 30 #percent
rotate_image = 0 #degree
correct_hour = -2

add_text = True
from PIL import ImageFont
font = ImageFont.truetype(r'/home/lmkrohn/GoPro/esparac.ttf', 100)


##### Modules ##############################################################################
from PIL import Image
from PIL import ImageDraw 
import glob
from tqdm import tqdm

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
from matplotlib import gridspec
import matplotlib.dates as mdates
from datetime import datetime
from datetime import timedelta
import cv2


##### find nearest ####################################################################
def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


##### convert timestamp to datetime########################################################
def Convert_timestamp_to_datetime(ts):
    dt=[]
    for i in range(len(ts)):
        dt.append(datetime.fromtimestamp(ts[i]) - timedelta(hours=1))
    return np.asarray(dt)


##### define map function ##################################################################
def mosaic_ACA_map_without_DS(foldername, dataname, flightdate, flightnumber, pic_time, leg_value, leg_loc='lower left'):
    flight = pd.read_table(foldername + 'Polar5_'+ flightdate +'.nav', skiprows=3, header=None, sep="\s+")
    flight.columns = ["time", "Longitude", "Latitude", "Altitude", "Velocity", "Pitch", "Roll", 
                      "Yaw", "SZA", "SAA"]
    flight['realtime']=pd.to_timedelta(flight['time'], unit='h')
    flight['strtime']=flight['realtime'].astype(str)[7:]
    flight['seconds']=flight['realtime'].dt.total_seconds()
    hmsm = Convert_timestamp_to_datetime(flight['seconds'])
    print(hmsm[0])

    
    
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
    
    gs = gridspec.GridSpec(2, 1, height_ratios=[2, 1]) 
    
    fig = plt.figure(figsize=(8,8))
    #fig, (ax1, ax2) = plt.subplots(2,1, figsize=(10,12), 
    #                               gridspec_kw={'height_ratios': [2, 1], 'wspace':0.1,}, 
    #                               subplot_kw={'projection': lcc_proj})
    ax = plt.axes(projection = lcc_proj)
    #ax1 = fig.add_subplot(2, 1, 1, projection=lcc_proj)
    ax.set_extent([-7, 30, 75, 84])
    
    ax.plot(flight['Longitude'], flight['Latitude'], transform = ccrs.PlateCarree(), 
            label=flightdate, color='darkorange', linewidth=2.5)
    
    ax.coastlines('10m')
    ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, dms=True, x_inline=False, y_inline=False,
                 linewidth=1, color='black', alpha=0.7, linestyle='--')
    ax.add_feature(cfeature.NaturalEarthFeature('physical', 'land', '10m', edgecolor='face', facecolor='darkgray'))
    
    
    LYRlon, LYRlat = 15.50150, 78.24588
    ax.scatter( LYRlon, LYRlat, color='red', label='Longyearbyen', linewidth=3,
             transform =ccrs.PlateCarree(), zorder=5, marker='D')
    
    
    pic_seconds = int(capture_time[9:11])*3600 + int(capture_time[11:13])*60 + int(capture_time[13:15])
    idx_pic_time = find_nearest(flight['seconds'], pic_seconds)
    
    ax.scatter(flight['Longitude'][idx_pic_time], flight['Latitude'][idx_pic_time], color='mediumblue', zorder=5,
                label='Polar 5', linewidth=3, transform=ccrs.PlateCarree())
    
    
    def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
        new_cmap = colors.LinearSegmentedColormap.from_list(
            'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
            cmap(np.linspace(minval, maxval, n)))
        return new_cmap
    
    
    cmap=plt.get_cmap('Blues_r')
    new_cmap=truncate_colormap(cmap, 0.5, 1)
    cs = ax.pcolormesh(lon, lat, data, transform=ccrs.PlateCarree(), cmap=new_cmap, vmin=0, vmax=100)
    
    cbar = plt.colorbar(cs, ax = ax, orientation='vertical', pad=0.1)
    cbar.set_label('Arctic sea ice concentration [%]')

    legend = ax.legend(loc='lower left')

    #plt.title('MOSAiC-flight '+flightnumber+', '+flightdate)

    
    plt.savefig(output_folder_image+'MOSAiC-ACA_flight'+flightnumber+'_'+capture_time+'_map.jpg', 
                dpi=300, bbox_inches='tight', transparent=True)
    #plt.show()
    
    #ax2 = fig.add_subplot(2,1,2)
    fig2, ax2 = plt.subplots(figsize=(8,2.5))
    #plt.rcParams['axes.grid'] = True
    dateformat = mdates.DateFormatter('%H:%M')
    
    
    ax2.plot(hmsm, flight['Altitude']/1000, linewidth= 2.5, color = 'darkorange')#,
             #label = 'altitude profile')
    ax2.scatter(hmsm[idx_pic_time], flight['Altitude'][idx_pic_time]/1000, color = 'mediumblue',
                label = 'altitude: '+str(flight['Altitude'][idx_pic_time])+' m',
                linewidth = 3, zorder=5)
    
    alt = int(np.round(flight['Altitude'][idx_pic_time]))
    
    ax2.grid(b=True, axis='both', which='major')
    #ax2.set_xticks(np.arange(29000,55000, 5000))
    ax2.set_yticks(np.arange(0,4.1,1))
    ax2.xaxis.set_major_formatter(dateformat)
    ax2.set_xlabel('time')
    ax2.set_ylabel('altitude (km)')
    #ax2.legend(loc = 'lower right')
    
    plt.savefig(output_folder_image+'MOSAiC-ACA_flight'+flightnumber+'_'+capture_time+"_alt_"+str(alt)+".jpg", 
                dpi=300, bbox_inches='tight', transparent=True)
    
    return alt





##### Main program #########################################################################
list_images = sorted(glob.glob(input_folder+"*.JPG"))

#height, width, layers = list_images[0].shape

#fourcc = cv2.VideoWriter_fourcc('D', 'I', 'V', 'X')#mp4v
#video = cv2.VideoWriter(output_folder_video+'Video_GoPro_map_alt_20200910a.avi', fourcc, 30, (5400, 3000), isColor=True)

for i in tqdm(range(10326, len(list_images), 2)):
    im = Image.open(list_images[i])
    capture_time = im._getexif()[36868]
    year = capture_time[0:4]
    month= capture_time[5:7]
    day  = capture_time[8:10]
    hour = str(int(capture_time[11:13]) + correct_hour)
    if len(hour) == 1:
        hour = "0"+hour
    minute=capture_time[14:16]
    second=capture_time[17:19]

    capture_time = year+month+day+"_"+hour+minute+second

    im = im.rotate(rotate_image)
    '''
    if add_text == True:
        draw = ImageDraw.Draw(black_im)
        draw.text((4000, 0),"MOSAiC-ACA, Flight "+flightnumber+ " \n  "+year+"-"+month+"-"+day+"   "+hour+":"+minute+":"+second,font=font) # this will draw text with Blackcolor and 16 size
    '''

    black_im = Image.new(mode='RGB', size=(5400, 3000))
    black_im.paste(im, (0,0))

    ##### run map function ############################################################
    alt = mosaic_ACA_map_without_DS(sic_folder, sic_name, flightdate, flightnumber, capture_time, 0)
    
    # add map to im
    im_map = Image.open(output_folder_image+'MOSAiC-ACA_flight'+flightnumber+'_'+capture_time+"_map.jpg").convert('RGBA')
    (im_width, im_height) = (im.width, im.height)
    (new_map_width, new_map_height) = (im_map.width*2 // 3, im_map.height*2 // 3)
    im_map = im_map.resize((new_map_width, new_map_height))
    black_im.paste(im_map, (4000, 1225))#2600 for transparent
    
    # add altitude profile to im
    im_alt = Image.open(output_folder_image+'MOSAiC-ACA_flight'+flightnumber+'_'+capture_time+"_alt_"+str(alt)+".jpg").convert('RGBA')
    (new_alt_width, new_alt_height) = (im_alt.width*2 //3, im_alt.height*2 // 3)
    im_alt = im_alt.resize((new_alt_width, new_alt_height))
    black_im.paste(im_alt, (4000,2499), im_alt)#2600 und 1250 for transparent
    
    
    if add_text == True:
        draw = ImageDraw.Draw(black_im)
        draw.text((4000, 200), str(alt)+ ' m a.s.l.',font=font)
        draw.text((4000, 0),"MOSAiC-ACA, Flight "+flightnumber+ " \n"+year+"-"+month+"-"+day+"   "+hour+":"+minute+":"+second,font=font) # this will draw text with Blackcolor (took away (0,0,0)) and 16 size
    
    black_im.save(output_folder+prefix+capture_time+".jpg",optimize=True,quality=output_quality)
    
    #black_im2 = np.asarray(black_im)
    #video.write(np.flip(black_im2, axis=-1))
    #flip because opencv uses BGR and matplotlib RGB
    
    #black_im.save(output_folder+prefix+capture_time+"3.jpg",optimize=True,quality=output_quality)
    cv2.destroyAllWindows()


#cv2.destroyAllWindows()
#video.release()
print("Done")

# 
