from flask import Flask, request, jsonify
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

#inisialisasi Flask app 
app = Flask(__name__)

#load model yang udah disave
with open('best_model.pkl', 'rb') as file:
    model = pickle.load(file)

#loadscaler n labelencoder
with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

with open('label_encoder.pkl', 'rb') as file:
    label_encoder = pickle.load(file)

#preprocessing data input
def preprocess_input(input_data):
    #preprocessing data input sebelum prediksi
    #handling missing values
    input_data.fillna(input_data.select_dtypes(include=['float64', 'int64']).median(), inplace=True)
    for col in input_data.select_dtypes(include=['object']).columns:
        input_data[col].fillna(input_data[col].mode()[0], inplace=True)

    #encoding buat categirucal
    input_data['type_of_meal_plan'] = label_encoder.transform(input_data['type_of_meal_plan'])
    input_data['room_type_reserved'] = label_encoder.transform(input_data['room_type_reserved'])
    input_data['market_segment_type'] = label_encoder.transform(input_data['market_segment_type'])

    #just numeric column
    input_data = input_data.select_dtypes(include=[np.number])

    #scaling data
    input_data_scaled = scaler.transform(input_data)

    return input_data_scaled

#endpoint buat prediksi
@app.route('/predict', methods=['POST'])
def predict():
    #nerima data input dan memberikan prediksi
    #dapetiin data input dari permintaan POST
    data = request.get_json()  #nerima data dalam format JSON

    #ngubah data input ke dataframe
    input_data = pd.DataFrame(data)

    #preprocessing data input
    preprocessed_data = preprocess_input(input_data)

    #prediksi menggunakan model
    prediction = model.predict(preprocessed_data)

    #ngembaliin hasil prediksi dalam format JSON
    return jsonify({"prediction": prediction[0]})

if __name__ == '__main__':
    #run flask
    app.run(debug=True)
