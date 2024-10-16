import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import load_model
import os

# Load dataset
def load_data():
    df = pd.read_csv('poker_data.csv')  # Ensure your dataset file name matches
    return df

# Preprocess the dataset
def preprocess_data(df):
    X = df.drop(columns=['CLASS'])
    y = df['CLASS']
    return X, y

# Build the neural network model
def build_model(input_shape):
    model = Sequential()
    model.add(Dense(64, activation='relu', input_shape=(input_shape,)))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(10, activation='softmax'))  # 10 classes for poker hands
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

if __name__ == "__main__":
    # Load data
    df = load_data()
    print("Initial dataset shape:", df.shape)
    
    # Check if the dataset is loaded
    if df.empty:
        print("DataFrame is empty. Please check your dataset file.")
        exit()

    # Preprocess data
    X, y = preprocess_data(df)
    print("Features shape:", X.shape)
    print("Labels shape:", y.shape)

    # Split dataset into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Check class distribution before resampling
    print("Class distribution before resampling:")
    print(y_train.value_counts())

    # Resample with SMOTE
    smote = SMOTE(k_neighbors=3)  # Adjust k_neighbors to avoid errors
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

    # Check class distribution after resampling
    print("Class distribution after resampling:")
    print(pd.Series(y_resampled).value_counts())

    # Build model
    model = build_model(X_resampled.shape[1])

    # Train model
    model.fit(X_resampled, y_resampled, epochs=10, batch_size=32)

    # Save the model
    model.save('poker_model.h5')
    print("Model saved as poker_model.h5")
