# -*- coding: utf-8 -*-
"""
Created on Mon Mar 15 15:41:20 2021

@author: Lukas Monrad-Krohn
"""

import simplekml
from simplekml import Kml, Snippet, Types
import pandas as pd
import numpy as np
import datetime


#date als 2020-08-30
#absoluter Dateipfad
def time_track(date, flightnumber, path):

    flight=pd.read_table(path , skiprows=3, header=None, sep="\s+")
    flight.columns=["time", "Longitude", "Latitude", "Altitude", "Velocity", "Pitch", "Roll", "Yaw", "SZA", "SAA"]
    flight['datetime'] = pd.to_timedelta(flight['time'], unit='h') + pd.to_datetime(date)
    flight = flight[flight.index % 50 == 0]
    #print(flight)

    subset = flight[['Longitude', 'Latitude', 'Altitude']]
    coords = [tuple(x) for x in subset.to_numpy()]

    iso_time = [dt.strftime('%Y-%m-%dT%H:%M:%S.%f+02:00') for dt in flight['datetime']]


    # Create the KML document
    kml = Kml(name="MOSAiC-ACA_flight"+flightnumber+"_"+date+"_time", open=1)
    doc = kml.newdocument(name="MOSAiC-ACA_flight"+flightnumber+"_"+date+"_time")
    doc.lookat.gxtimespan.begin = iso_time[0]
    doc.lookat.gxtimespan.end = iso_time[-1]
    doc.lookat.longitude = 15.5
    doc.lookat.latitude = 78.24
    doc.lookat.range = 100000.0

    # Create a folder
    fol = doc.newfolder(name='Track_with_time')

    # Create a schema for extended data
    roll = flight['Roll'].values.tolist()
    yaw = flight['Yaw'].values.tolist()
    pitch = flight['Pitch'].values.tolist()
    schema = kml.newschema()
    schema.newgxsimplearrayfield(name='Roll', type=Types.float, displayname='Roll (deg)')
    schema.newgxsimplearrayfield(name='Yaw', type=Types.float, displayname='Yaw (deg)')
    schema.newgxsimplearrayfield(name='Pitch', type=Types.float, displayname='Pitch (deg)')

    # Create a new track in the folder
    trk = fol.newgxtrack(name='MOSAiC-ACA_flight'+flightnumber+'_' +date+'_time')

    # Apply the above schema to this track
    trk.extendeddata.schemadata.schemaurl = schema.id

    # Add all the information to the track
    trk.newwhen(iso_time) # Each item in the give nlist will become a new <when> tag
    trk.newgxcoord(coords)# Ditto
    trk.altitudemode=simplekml.AltitudeMode.absolute
    trk.extendeddata.schemadata.newgxsimplearraydata('Roll', roll) # Ditto
    trk.extendeddata.schemadata.newgxsimplearraydata('Yaw', yaw)
    trk.extendeddata.schemadata.newgxsimplearraydata('Pitch', pitch)

    # Styling
    trk.stylemap.normalstyle.iconstyle.icon.href = 'http://maps.google.com/mapfiles/kml/shapes/airports.png'
    trk.stylemap.normalstyle.linestyle.color = '99ffac59'
    trk.stylemap.normalstyle.linestyle.width = 6
    trk.stylemap.highlightstyle.iconstyle.icon.href = 'http://maps.google.com/mapfiles/kml/shapes/airports.png'
    trk.stylemap.highlightstyle.iconstyle.scale = 1.2
    trk.stylemap.highlightstyle.linestyle.color = '99ffac59'
    trk.stylemap.highlightstyle.linestyle.width = 8
    #http://earth.google.com/images/kml-icons/track-directional/track-0.png
    # Save the kml to file
    kml.save("MOSAiC-ACA_flight"+flightnumber+"_"+date+"_timeUTC.kml", format=True)
    print('process finished')
    

#%% test with number 02

time_track('2020-08-30', '2', 'C:/Users/Lukas Monrad-Krohn/Desktop/uni/shk/Mosaic ACA/Mosaic-ACA_flight02_20200830a/Polar5_20200830a.nav')

#%% loop
dates = ['2020-08-30', '2020-08-31', '2020-08-31', '2020-09-02', '2020-09-04', '2020-09-07',
         '2020-09-08', '2020-09-10', '2020-09-11', '2020-09-13']

paths = ['C:/Users/Lukas Monrad-Krohn/Desktop/uni/shk/Mosaic ACA/Mosaic-ACA_flight02_20200830a/Polar5_20200830a.nav', 
         'C:/Users/Lukas Monrad-Krohn/Desktop/uni/shk/Mosaic ACA/Mosaic-ACA_flight03_20200831a/Polar5_20200831a.nav', 
         'C:/Users/Lukas Monrad-Krohn/Desktop/uni/shk/Mosaic ACA/Mosaic-ACA_flight04_20200831b/Polar5_20200831b.nav', 
         'C:/Users/Lukas Monrad-Krohn/Desktop/uni/shk/Mosaic ACA/Mosaic-ACA_flight05_20200902/Polar5_20200902a.nav', 
         'C:/Users/Lukas Monrad-Krohn/Desktop/uni/shk/Mosaic ACA/Mosaic-ACA_flight06_20200904/Polar5_20200904a.nav', 
         'C:/Users/Lukas Monrad-Krohn/Desktop/uni/shk/Mosaic ACA/Mosaic-ACA_flight07_20200907/Polar5_20200907a.nav', 
         'C:/Users/Lukas Monrad-Krohn/Desktop/uni/shk/Mosaic ACA/Mosaic-ACA_flight08_20200908/Polar5_20200908a.nav', 
         'C:/Users/Lukas Monrad-Krohn/Desktop/uni/shk/Mosaic ACA/Mosaic-ACA_flight09_20200910/Polar5_20200910a.nav', 
         'C:/Users/Lukas Monrad-Krohn/Desktop/uni/shk/Mosaic ACA/Mosaic-ACA_flight10_20200911/Polar5_20200911a.nav', 
         'C:/Users/Lukas Monrad-Krohn/Desktop/uni/shk/Mosaic ACA/Mosaic-ACA_flight11_20200913/Polar5_20200913a.nav']

for i in range(10):
    time_track(dates[i], str(i+2), paths[i])




















