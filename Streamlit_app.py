# -*- coding: utf-8  -*-

'''
Created on Fri 1 Sep 2023

@author: abudto
'''


import numpy as np
import pandas as pd
import pickle
import streamlit as st

loaded_model = pickle.load(open('Random_Forest_Regression.pkl', 'rb'))
loaded_encoder = pickle.load(open('Column_tnf.pkl', 'rb'))


def main():

    # Giving a title
    st.title('Used Car Price Prediction')
    user_input = {}
    # getting the input data from the user
    user_input['Model'] = st.selectbox('Model', ('HR-V', 'Yaris', 'Ertiga', 'others', 'Xenia', 'Baleno', 'Sigra',
                                                 'Mobilio', 'Avanza', 'Agya', 'Calya', 'Ignis', 'Xpander', 'Raize',
                                                 'Brio', 'Ayla', 'XL7', 'Terios', 'Rush', 'BR-V', 'Kijang Innova',
                                                 'Jazz', 'CR-V', 'Fortuner', 'Almaz', 'X-Trail', 'City',
                                                 'Grand Livina', 'Alphard', 'X5', 'C200', 'Vellfire', 'Civic',
                                                 'CX-5', 'Cooper', 'GLA200', 'Camry', 'Pajero Sport', 'Santa Fe',
                                                 'Creta', 'Veloz', '520i', 'Stargazer', 'X1', 'CLA200', '320i',
                                                 'Land Cruiser', 'E300', 'Kijang Innova Zenix', 'Palisade',
                                                 'IONIQ 5'))
    user_input['Year'] = st.selectbox('Year',   (2018, 2021, 2017, 2016, 2014, 2019, 2022, 2023, 2020, 2013, 2015,
                                      2012, 2009, 2002, 2011, 2005, 2004, 2007, 2008, 2000, 2010,
                                      2006))
    user_input['Mileage'] = st.text_input('Mileage')
    user_input['Brand'] = st.selectbox('Brand', ('Honda', 'Toyota', 'Suzuki', 'Nissan', 'Daihatsu', 'Mitsubishi',
                                                 'Mercedes-Benz', 'Mazda', 'Wuling', 'BMW', 'others', 'Land Rover',
                                                 'Volkswagen', 'Lexus', 'MINI', 'Chevrolet', 'Hyundai', 'Isuzu',
                                                 'Ford', 'Jeep', 'Porsche'))
    user_input['Color'] = st.selectbox('Color', ('Grey', 'Red', 'Black', 'Silver', 'White', 'Others', 'Brown',
                                       'Orange', 'Blue', 'Yellow', 'Green', 'Gold', 'Maroon', 'Purple'))
    user_input['Body Type'] = st.selectbox('Body Type',  ('SUV', 'Hatchback', 'MPV', 'Wagon', 'Trucks', 'Pick-up',
                                           'Van Wagon', 'Sedan', 'Cabriolet', 'Gran Coupe', 'Coupe', 'Van',
                                           'Convertible', 'Sportback', 'Jeep', 'MPV Minivans',
                                           'Fastback', 'Minibus', 'Others', 'Double Cabin',
                                           'Compact Car City Car', 'SUV Offroad 4WD', 'Targa'))
    user_input['Fuel Type'] = st.selectbox('Fuel Type', ('Pertamax', 'Solar', 'Electric', 'Petrol - Unleaded', 'Premium',
                                           'Diesel', 'CNG'))
    user_input['Seating Capacity'] = st.text_input('Seating Capacity')
    user_input['Seller City'] = st.selectbox('Seller City',   ('Bekasi', 'Bogor', 'Depok', 'Bandung', 'Sumedang', 'Garut',
                                             'Cikarang', 'Cianjur', 'Ciamis', 'Tasikmalaya', 'Cimahi', 'Banjar',
                                             'Cirebon', 'Purwakarta', 'Karawang', 'Sukabumi', 'Majalengka',
                                             'Subang', 'Kuningan', 'Jakarta Pusat', 'Jakarta Selatan',
                                             'Jakarta Utara', 'Jakarta Timur', 'Jakarta Barat',
                                             'Kepulauan Seribu', 'Kediri', 'Surabaya', 'Malang', 'Pamekasan',
                                             'Sidoarjo', 'Mojokerto', 'Bojonegoro', 'Jombang', 'Gresik',
                                             'Nganjuk', 'Bangkalan', 'Tuban', 'Madiun', 'Blitar', 'Lamongan',
                                             'Pasuruan', 'Situbondo', 'Sampang', 'Ponorogo', 'Magetan',
                                             'Tulungagung', 'Sumenep', 'Pacitan', 'Bondowoso', 'Probolinggo',
                                             'Ngawi', 'Trenggalek', 'Lumajang', 'Jember', 'Batu'))
    user_input['Seller Region'] = st.selectbox('Seller Region', ('Jawa Barat','Jawa Timur','DKI Jakarta'))

    # Code for Prediction
    price = ''

    # Creating a button for Prediction

    if st.button('Used Car Price Predict Result'):

        user_input_dict = {
            'Model': [user_input['Model']],
            'Year': [user_input['Year']],
            'Mileage': [user_input['Mileage']],
            'Brand': [user_input['Brand']],
            'Color': [user_input['Color']],
            'Body Type': [user_input['Body Type']],
            'Fuel Type': [user_input['Fuel Type']],
            'Seating Capacity': [user_input['Seating Capacity']],
            'Seller Address': [user_input['Seller City']],
            'Seller Region': [user_input['Seller Region']]
        }

        # Create a DataFrame from the user input dictionary
        user_input_df = pd.DataFrame(user_input_dict)

        # Feature Engineering, add Car_age and drop Year
        user_input_df['Car_age'] = 2023 - user_input_df['Year']
        user_input_df.drop(['Year'], axis=1, inplace=True)

        # Debugging: Print the shape and columns of the input data
        # print("Input Data Shape:", user_input_df.shape)
        # print("Input Data Columns:", user_input_df.columns)

        price = used_car_predict(user_input_df)

    st.success(price)


def used_car_predict(input_data):

    # transform the new data with one hot encoder and standard scaler
    new_data_test_tnf = loaded_encoder.transform(input_data)

    prediction = loaded_model.predict(new_data_test_tnf)

    prediction_ftd = float(prediction)

    return 'Price: Rp {:,.0f}'.format(prediction_ftd)


if __name__ == '__main__':
    main()