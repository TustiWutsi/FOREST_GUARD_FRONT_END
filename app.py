import streamlit as st
import folium
from streamlit_folium import folium_static
import ee 

service_account = 'earth-engine@wagon-bootcamp-data.iam.gserviceaccount.com'
credentials = ee.ServiceAccountCredentials(service_account, 'private_key.json')
ee.Initialize(credentials)

st.title('Forest guards')

# Display predicted target on a map
prediction_adam_crossentropy = ee.Image('users/bastidearthur/france_mont_de_marsan_2018model_adam_crossentropy')

jaxa = ee.ImageCollection('JAXA/ALOS/PALSAR/YEARLY/FNF').filterDate('2017-01-01', '2017-12-31').median()

palette_1 = ['006400','FEFF99']     

mapid_1 = jaxa.getMapId({'bands': ['fnf'],
                       'min': 1, 
                       'max': 2, 
                      'palette':palette_1
                      })

mapid_2 = prediction_adam_crossentropy.getMapId({'bands': ['fnf'],
                       'min': 0, 
                       'max': 1, 
                      'palette':palette_1
                      })


map = folium.Map(location=[45.5, 2.8], zoom_start=9)

folium.TileLayer(
    tiles=mapid_1['tile_fetcher'].url_format,
    attr='Map Data &copy; <a href="https://earthengine.google.com/">Google Earth Engine</a>',
    overlay=True,
    name='jaxa',
    color=palette_1,
    opacity = 1
  ).add_to(map)

folium.TileLayer(
    tiles=mapid_2['tile_fetcher'].url_format,
    attr='Map Data &copy; <a href="https://earthengine.google.com/">Google Earth Engine</a>',
    overlay=False,
    name='prediction adam_crossentropy',
    color=palette,
    opacity = 1
  ).add_to(map)

map.add_child(folium.LayerControl())

folium_static(map)