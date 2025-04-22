import streamlit as st
import pickle
import gzip 
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

#load best model yang dicompress bcs ukuran filenya ke gedean
with gzip.open('best_model_compressed.pkl.gz', 'rb') as file:
    best_model = pickle.load(file)

#loadscaler n labelencoder
with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

with open('label_encoder.pkl', 'rb') as file:
    label_encoder = pickle.load(file)

#preprocessing input data
def preprocess_input(input_data):
    input_data.fillna(input_data.select_dtypes(include=['float64', 'int64']).median(), inplace=True)
    for col in input_data.select_dtypes(include=['object']).columns:
        input_data[col].fillna(input_data[col].mode()[0], inplace=True)

    input_data['type_of_meal_plan'] = label_encoder.transform(input_data['type_of_meal_plan'])
    input_data['room_type_reserved'] = label_encoder.transform(input_data['room_type_reserved'])
    input_data['market_segment_type'] = label_encoder.transform(input_data['market_segment_type'])

    input_data = input_data.select_dtypes(include=[np.number])

    input_data_scaled = scaler.transform(input_data)
    
    return input_data_scaled

#interface streamlit
st.title('Hotel Booking Prediction')

st.write("""
### Masukkan informasi penginapan untuk memprediksi status booking.
""")

#input data dari user
no_of_adults = st.number_input('Jumlah Dewasa:', min_value=1, max_value=10, value=2)
no_of_children = st.number_input('Jumlah Anak:', min_value=0, max_value=10, value=0)
no_of_weekend_nights = st.number_input('Malam Akhir Pekan:', min_value=0, max_value=7, value=1)
no_of_week_nights = st.number_input('Malam dalam Seminggu:', min_value=0, max_value=7, value=3)
type_of_meal_plan = st.selectbox('Paket Makanan:', ['Meal Plan 1', 'Meal Plan 2', 'Not Selected'])
room_type_reserved = st.selectbox('Tipe Kamar yang Dipesan:', ['Room_Type 1', 'Room_Type 2', 'Room_Type 3'])
market_segment_type = st.selectbox('Tipe Segmen Pasar:', ['Online', 'Offline'])
lead_time = st.number_input('Lead Time (hari):', min_value=1, max_value=365, value=50)
arrival_year = st.number_input('Tahun Kedatangan:', min_value=2020, max_value=2025, value=2023)
arrival_month = st.number_input('Bulan Kedatangan:', min_value=1, max_value=12, value=5)
arrival_date = st.number_input('Tanggal Kedatangan:', min_value=1, max_value=31, value=10)
repeated_guest = st.number_input('Tamu yang Pernah Mendaftar Ulang:', min_value=0, max_value=1, value=0)
no_of_previous_cancellations = st.number_input('Jumlah Pembatalan Sebelumnya:', min_value=0, max_value=5, value=0)
no_of_previous_bookings_not_canceled = st.number_input('Jumlah Pemesanan Sebelumnya yang Tidak Dibatalkan:', min_value=0, max_value=10, value=1)
avg_price_per_room = st.number_input('Harga Rata-rata Per Kamar:', min_value=0, value=150)
no_of_special_requests = st.number_input('Jumlah Permintaan Khusus:', min_value=0, max_value=10, value=0)

#DataFrame dari inputan pengguna
input_data = pd.DataFrame({
    'no_of_adults': [no_of_adults],
    'no_of_children': [no_of_children],
    'no_of_weekend_nights': [no_of_weekend_nights],
    'no_of_week_nights': [no_of_week_nights],
    'type_of_meal_plan': [type_of_meal_plan],
    'room_type_reserved': [room_type_reserved],
    'market_segment_type': [market_segment_type],
    'lead_time': [lead_time],
    'arrival_year': [arrival_year],
    'arrival_month': [arrival_month],
    'arrival_date': [arrival_date],
    'repeated_guest': [repeated_guest],
    'no_of_previous_cancellations': [no_of_previous_cancellations],
    'no_of_previous_bookings_not_canceled': [no_of_previous_bookings_not_canceled],
    'avg_price_per_room': [avg_price_per_room],
    'no_of_special_requests': [no_of_special_requests]
})

#preprocess input data and make predictions
preprocessed_data = preprocess_input(input_data)
prediction = best_model.predict(preprocessed_data)

#result
if prediction[0] == 0:
    st.write("**Status Booking:** Dibatalkan")
else:
    st.write("**Status Booking:** Tidak Dibatalkan")
