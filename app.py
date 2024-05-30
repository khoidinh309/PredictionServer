from flask import Flask, jsonify, request
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import tensorflow as tf
import keras
import json
import pickle

app = Flask(__name__)

# Load the model
mlp_model = tf.keras.models.load_model('./mlp_housing.keras')

linear_model = pickle.load(open('my_HaNoi_housing_linear_model.pkl', 'rb'))
preprocessing_pipeline = pickle.load(open('preprocessing_pipeline.pkl', 'rb'))

def split_date(housing):
  housing['month'] = housing['date'].str.split('/').str[0].astype(float)
  housing['day'] = housing['date'].str.split('/').str[1].astype(float)
  housing['year'] = housing['date'].str.split('/').str[2].astype(float)
  
  return housing
  
def get_season(month):
  if 1 <= month <= 3:
      return 'Winter'
  elif 4 <= month <= 6:
      return 'Spring'
  elif 7 <= month <= 9:
      return 'Summer'
  else:
      return 'Fall'
    
    
def encodeDate(data, col, max_val):
  data[col + '_sin'] = np.sin(2 * np.pi * data[col]/max_val)
  data[col + '_cos'] = np.cos(2 * np.pi * data[col]/max_val)
  return data

def utf8_encode(s):
    return s.encode('utf-8')

@app.route('/linear/predict', methods=['POST'])
def predict():
    try:
      # Get the data from the request
      data = request.get_json()
      print(data)
     
      housing = pd.DataFrame([data])
      housing = split_date(housing)
      
      housing = housing.drop('date', axis=1)
      
      housing['season'] = housing['month'].apply(get_season)
      print(housing.columns)

      transformed_data = preprocessing_pipeline.transform(housing)
      
      prediction = linear_model.predict(transformed_data)
      
      return jsonify({'Prediction': prediction[0]} ) 
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400
    
@app.route('/mlp/predict', methods=['POST'])
def predict_mlp():
    try:
      # Get the data from the request
      data = request.get_json()
     
      mlp_housing = pd.DataFrame([data])
      mlp_housing = split_date(mlp_housing)
      
      #housing = housing.drop('date', axis=1)
      
      mlp_housing['season'] = mlp_housing['month'].apply(get_season)
      
      mlp_housing = mlp_housing.drop("date", axis=1)
      
      numeric_columns = mlp_housing.select_dtypes(include=['int32', 'float64']).columns
      mlp_housing[numeric_columns] = mlp_housing[numeric_columns].astype(float)

      string_columns = mlp_housing.select_dtypes(include=['object']).columns
      mlp_housing[string_columns] = mlp_housing[string_columns].astype(str)
      
      cat_columns = ['district', 'ward', 'location_type', 'season']
      num_columns = ['number_of_bedrooms', 'area', 'year', 'month', 'day']
      
      prediction = mlp_model.predict([mlp_housing[num_columns]] + [mlp_housing[col] for col in cat_columns])
      
      print(prediction)
      
      return jsonify({'Prediction': prediction[0][0].astype(float)} ) 
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400
  
if __name__ == '__main__':
    app.run(port=5000, debug=True)