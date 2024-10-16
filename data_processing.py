# data_processing.py

import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

def load_data():
    df = pd.read_csv('poker_data.csv')
    return df

def preprocess_data(df):
    X = df.drop(columns=['CLASS'])
    y = df['CLASS']
    return X, y

def build_model(input_shape):
    model = Sequential()
    model.add(Dense(64, activation='relu', input_shape=(input_shape,)))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(10, activation='softmax'))  # Assuming 10 classes
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
