import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import pickle

#load data
df = pd.read_csv("Dataset_B_hotel.csv")

df.head()

class HotelBookingModel:
    def __init__(self, data_path):
        #inisialisassi dengan path file data
        self.data_path = data_path
        self.df = pd.read_csv(self.data_path)  #load data
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.best_model = None
        self.rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.xgb_model = XGBClassifier(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()

    def load_data(self):
        #load data
        self.df = pd.read_csv(self.data_path)
        print("Data loaded successfully.")
        print(self.df.head())

    def handle_missing_values(self):
        #handle missing values in dataset
        #handling missing values for numeric columns
        self.df.fillna(self.df.select_dtypes(include=['float64', 'int64']).median(), inplace=True)

        #handling missing values for categorical columns
        for col in self.df.select_dtypes(include=['object']).columns:
            self.df[col].fillna(self.df[col].mode()[0], inplace=True)

    def encode_categorical_features(self):
        #encode categorical features into numerical values
        self.df['type_of_meal_plan'] = self.label_encoder.fit_transform(self.df['type_of_meal_plan'])
        self.df['room_type_reserved'] = self.label_encoder.fit_transform(self.df['room_type_reserved'])
        self.df['market_segment_type'] = self.label_encoder.fit_transform(self.df['market_segment_type'])
        self.df['booking_status'] = self.label_encoder.fit_transform(self.df['booking_status'])

    def split_data(self):
        #split data into features (X) and target (y)
        X = self.df.drop('booking_status', axis=1)  #drop target variable
        y = self.df['booking_status']  #target variable

        #cuma pilih numeric columns for features
        X = X.select_dtypes(include=[np.number])

        #split data into training and testing sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    def scale_features(self):
        #scale numerical features
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)

    def train_models(self):
        #train Random Forest and XGBoost models
        self.rf_model.fit(self.X_train, self.y_train)
        self.xgb_model.fit(self.X_train, self.y_train)

    def evaluate_models(self):
        #evaluate both models and return accuracy scores
        #evaluate Random Forest
        rf_pred = self.rf_model.predict(self.X_test)
        rf_accuracy = accuracy_score(self.y_test, rf_pred)

        #evaluate XGBoost
        xgb_pred = self.xgb_model.predict(self.X_test)
        xgb_accuracy = accuracy_score(self.y_test, xgb_pred)

        return rf_accuracy, xgb_accuracy

    def save_best_model(self, rf_accuracy, xgb_accuracy):
        #select the best model based on accuracy and save it using pickle
        if rf_accuracy > xgb_accuracy:
            self.best_model = self.rf_model
            print("Random Forest is the best model.")
        else:
            self.best_model = self.xgb_model
            print("XGBoost is the best model.")

        #save best model ke pickle
        with open('best_model.pkl', 'wb') as file:
            pickle.dump(self.best_model, file)

        print("Best model saved successfully.")

#mengguanakan class buat menjalankan proses
if __name__ == "__main__":

    data_path = 'Dataset_B_hotel.csv'

    #buat object model
    model = HotelBookingModel(data_path)

    #menjalankan seluruh proses secara terstruktur
    model.load_data()  #load data
    model.handle_missing_values()  #ini missing values
    model.encode_categorical_features()  #lakuin encoding pada categorical fitur
    model.split_data()  #ngebagi data menjadi training dan testing set
    model.scale_features()  #scaling pada data numeric
    model.train_models()  #train model Random Forest dan XGBoost

    #evaluasi model n save model terbaik
    rf_accuracy, xgb_accuracy = model.evaluate_models()
    model.save_best_model(rf_accuracy, xgb_accuracy)  #save model terbaik
