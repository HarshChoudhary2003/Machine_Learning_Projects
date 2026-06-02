from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load the model and scaler
try:
    model = joblib.load('model.pkl')
    scaler = joblib.load('scaler.pkl')
except FileNotFoundError:
    model = None
    scaler = None
    print("Model or scaler not found. Make sure to run train.py first.")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None or scaler is None:
        return jsonify({'error': 'Model not loaded'}), 500
        
    try:
        data = request.json
        
        # Extract features in the correct order
        # ['Daily Time Spent on Site', 'Age', 'Area Income', 'Daily Internet Usage', 'Male']
        features = [
            float(data['daily_time']),
            float(data['age']),
            float(data['area_income']),
            float(data['daily_internet']),
            float(data['male'])
        ]
        
        # Scale the features
        features_scaled = scaler.transform([features])
        
        # Predict
        prediction = model.predict(features_scaled)[0]
        probability = model.predict_proba(features_scaled)[0][1]
        
        result = {
            'prediction': int(prediction),
            'probability': float(probability),
            'message': 'User is likely to click on the ad' if prediction == 1 else 'User is unlikely to click on the ad'
        }
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5000)
