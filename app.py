from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the trained model
model = load_model('poker_model.h5')

# Define a route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Define a route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Get the input data from the form
    data = request.form.to_dict()
    features = [float(data[f'S{i}']) for i in range(1, 6)] + \
               [float(data[f'C{i}']) for i in range(1, 6)]
    features = np.array(features).reshape(1, -1)

    # Make a prediction
    prediction = model.predict(features)
    class_prediction = np.argmax(prediction, axis=1)[0]

    return jsonify({'predicted_class': int(class_prediction)})

if __name__ == '__main__':
    app.run(debug=True)
