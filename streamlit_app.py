import streamlit as st
import pickle
import gzip
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load best model yang dicompress
with gzip.open('/mnt/data/best_model_compressed.pkl.gz', 'rb') as file:
    best_model = pickle.load(file)

# Load scaler and label encoder
with open('/mnt/data/scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

with open('/mnt/data/label_encoder.pkl', 'rb') as file:
    label_encoder = pickle.load(file)

# Preprocessing input data
def preprocess_input(input_data):
    input_data.fillna(input_data.select_dtypes(include=['float64', 'int64']).median(), inplace=True)
    for col in input_data.select_dtypes(include=['object']).columns:
        input_data[col].fillna(input_data[col].mode()[0], inplace=True)

    input_data['person_gender'] = label_encoder.transform(input_data['person_gender'])
    input_data['previous_loan_defaults_on_file'] = label_encoder.transform(input_data['previous_loan_defaults_on_file'])

    education_order = ['High School', 'Associate', 'Bachelor', 'Master', 'Doctorate']
    input_data['person_education'] = pd.Categorical(input_data['person_education'], categories=education_order, ordered=True).codes

    input_data = pd.get_dummies(input_data, columns=['person_home_ownership', 'loan_intent'], drop_first=False)

    required_columns = best_model.get_booster().feature_names
    for col in required_columns:
        if col not in input_data.columns:
            input_data[col] = 0
    input_data = input_data[required_columns]

    return input_data

# Interface streamlit
st.set_page_config(page_title="hotel", page_icon="🏨", layout="centered")
st.title("Hotel Booking App")

st.markdown(""" 
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
    'person_age': [no_of_adults],  #gantisesuai data yang relevan
    'person_gender': ['male'],  #ganti sesuai input pengguna
    'person_income': [50000],  #ganti sesuai data pengguna
    'person_education': ['Bachelor'],  #ganti sesuai input pengguna
    'previous_loan_defaults_on_file': ['No'],  #ganti sesuai input
    'person_emp_exp': [5],  #ganti sesuai data pengguna
    'person_home_ownership': ['Own'],  #ganti sesuai input pengguna
    'loan_intent': ['PERSONAL'],  #ganti sesuai input pengguna
    'loan_amnt': [20000],  #ganti sesuai input pengguna
    'loan_int_rate': [5],  #ganti sesuai input
    'cb_person_cred_hist_length': [10],  #ganti sesuai input
    'credit_score': [700],  #ganti sesuai input pengguna
    'loan_percent_income': [30]  #ganti sesuai input pengguna
})

#preprocess input data and make predictions
preprocessed_data = preprocess_input(input_data)
prediction = best_model.predict(preprocessed_data)

#result
if prediction[0] == 0:
    st.write("**Status Booking:** Dibatalkan")
else:
    st.write("**Status Booking:** Tidak Dibatalkan")
