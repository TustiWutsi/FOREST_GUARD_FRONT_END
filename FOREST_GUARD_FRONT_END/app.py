import streamlit as st
import folium
from streamlit_folium import folium_static
import ee 
from dotenv import load_dotenv
import os
import pandas as pd

dotenv_path = os.path.join(os.path.dirname(__file__), ".env")

load_dotenv(dotenv_path)
key_data = os.getenv('EARTH_KEY_DATA')

service_account = 'earth-engine@wagon-bootcamp-data.iam.gserviceaccount.com'
credentials = ee.ServiceAccountCredentials(service_account, key_data=key_data)
ee.Initialize(credentials)

st.title('Forest guards')

# Forest selection
df_zone = pd.DataFrame({'first column': ['Select a forest', 'Vosges', 'Black Forest']})
option_zone = st.selectbox('Select a forest', df_zone['first column'])
zones = {
        'Black Forest' : [49, 7.5],
        'Vosges' : [48.5, 8.3]
        }


# Year selection
option_year = st.slider('Select a year', 2017, 2020)

OPTICAL_BANDS = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7']
THERMAL_BANDS = ['B10', 'B11']
BANDS = OPTICAL_BANDS + THERMAL_BANDS
RESPONSE = 'fnf'
FEATURES = BANDS + [RESPONSE]

MODEL_NAME = 'JP_test_model_adam_binarycrossentropy'
PROJECT = 'wagon-bootcamp-data'
REGION = 'europe-west1'

def add_ee_layer(self, ee_image_object, vis_params, name):
  map_id_dict = ee.Image(ee_image_object).getMapId(vis_params)
  folium.raster_layers.TileLayer(
      tiles=map_id_dict['tile_fetcher'].url_format,
      attr='Map Data &copy; <a href="https://earthengine.google.com/">Google Earth Engine</a>',
      name=name,
      overlay=True,
      control=True,
      show=False,
      opacity=0.5
  ).add_to(self)
folium.Map.add_ee_layer = add_ee_layer

# Use Landsat 8 surface reflectance data.
l8sr = ee.ImageCollection('LANDSAT/LC08/C01/T1_SR')

# Cloud masking function.
def maskL8sr(image):
  cloudShadowBitMask = ee.Number(2).pow(3).int()
  cloudsBitMask = ee.Number(2).pow(5).int()
  qa = image.select('pixel_qa')
  mask1 = qa.bitwiseAnd(cloudShadowBitMask).eq(0).And(
    qa.bitwiseAnd(cloudsBitMask).eq(0))
  mask2 = image.mask().reduce('min')
  mask3 = image.select(OPTICAL_BANDS).gt(0).And(
          image.select(OPTICAL_BANDS).lt(10000)).reduce('min')
  mask = mask1.And(mask2).And(mask3)
  return image.select(OPTICAL_BANDS).divide(10000).addBands(
          image.select(THERMAL_BANDS).divide(10).clamp(273.15, 373.15)
            .subtract(273.15).divide(100)).updateMask(mask)

# The image input data is a cloud-masked median composite.
image = l8sr.filterDate(
    '2018-01-01', '2020-12-31').map(maskL8sr).median().select(BANDS).float()
image20 = l8sr.filterDate(
    '2020-01-01', '2020-12-31').map(maskL8sr).median().select(BANDS).float()
image19 = l8sr.filterDate(
    '2019-01-01', '2019-12-31').map(maskL8sr).median().select(BANDS).float()
image18 = l8sr.filterDate(
    '2018-01-01', '2018-12-31').map(maskL8sr).median().select(BANDS).float()

years = {
        2018 : image18,
        2019 : image19,
        2020 : image20
        }

# Load the trained model and use it for prediction.  If you specified a region 
# other than the default (us-central1) at model creation, specify it here.
model = ee.Model.fromAiPlatformPredictor(
    projectId = PROJECT,
    modelName = MODEL_NAME,
    # modelName = "ai_platform_ee_output_format_tf2",
    #version = VERSION_NAME,
    version = "v1622563893",
    region= REGION,
    inputTileSize = [144, 144],
    inputOverlapSize = [8, 8],
    proj = ee.Projection('EPSG:4326').atScale(30),
    fixInputProj = True,
    outputBands = {'fnf': {
        'type': ee.PixelType.float()
      }
    }
)
palette = [ '#FEFF99',
           '006400',
            # '0000FF'
          ]

if option_zone != 'Select a forest':  
  # Use folium to visualize the input imagery and the predictions.
  mapid = years[option_year].getMapId({'bands': ['B4', 'B3', 'B2'], 'min': 0, 'max': 0.3})
  map = folium.Map(location=zones[option_zone], zoom_start=15)
  folium.TileLayer(
      tiles=mapid['tile_fetcher'].url_format,
      attr='Google Earth Engine',
      overlay=True,
      name='landsat',
      show=False
    ).add_to(map)

  ### Prediction
  predictions = model.predictImage(image.toArray())

  mask_pred = predictions.updateMask(predictions.gte(0.8))

  map.add_ee_layer(mask_pred,
                  {'bands': ['fnf'],
                        'min': 1, 
                        'max': 1, 
                        'palette':palette
                        }, 'predictions_masked'
                  )
  ###JAXA
  jaxa = ee.ImageCollection('JAXA/ALOS/PALSAR/YEARLY/FNF').filterDate('2017-01-01', '2017-12-31').median()

  mask_pred_jaxa = jaxa.updateMask(jaxa.lte(1.5))

  map.add_ee_layer(mask_pred_jaxa,
                  {'bands': ['fnf'],
                        'min': 1, 
                        'max': 3, 
                        'palette':palette
                        }, 'jaxa_masked'
                  )


  map.add_child(folium.LayerControl())

  folium_static(map)

else:
  pass