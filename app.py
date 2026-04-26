import pickle
import streamlit as st
import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# load model
model = pickle.load(open('lr_model.pkl','rb'))

# load scaler
scaler = pickle.load(open('scaler.pkl','rb'))

# title
st.title('House Price Prediction App')

# input variables
Square_Footage = st.number_input('Square Footage', min_value=503,max_value=4999,value=1000)
Num_Bedrooms = st.number_input('Number of Bedrooms', min_value=1,max_value=5,value=2)
Num_Bathrooms = st.number_input('Number of Bathrooms', min_value=1,max_value=3,value=2)
Lot_Size = st.number_input('Lot Size', min_value=1,max_value=5,value=3)
Year_Built = st.number_input('Year Built', min_value=1950,max_value=2022,value=2000)
Garage_Size = st.number_input('Garage Size', min_value=0,max_value=2,value=1)
Neighborhood_Quality = st.number_input('Neighborhood Quality', min_value=1,max_value=10,value=5)

# create dataframe
input_features = pd.DataFrame({
    'Square_Footage':[Square_Footage],
    'Num_Bedrooms':[Num_Bedrooms],
    'Num_Bathrooms':[Num_Bathrooms],
    'Year_Built':[Year_Built],
    'Lot_Size':[Lot_Size],
    'Garage_Size':[Garage_Size],
    'Neighborhood_Quality':[Neighborhood_Quality]
})

# apply scaling
input_features[['Square_Footage','Num_Bedrooms','Num_Bathrooms','Year_Built','Lot_Size','Garage_Size','Neighborhood_Quality']] = scaler.transform(input_features
              [['Square_Footage','Num_Bedrooms','Num_Bathrooms','Year_Built','Lot_Size','Garage_Size','Neighborhood_Quality']])

# predictions
if st.button('Predict'):
    prediction = model.predict(input_features)
    st.success(f'The predicted house price is ${prediction[0]:,.2f}')