from flask import Flask, render_template, request
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
import pickle 
import joblib

app = Flask(__name__)

# Load and prepare the data (this can be adjusted based on the dataset path)
# For simplicity, we'll assume the model is already trained and serialized

# Sample training data logic (replace with actual saved model)
model = joblib.load('c:/Deepak_Chauhan/Machine Learning/heart-disease-prediction/heart_disease_model.pkl')


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Getting form data
        age = int(request.form['age'])
        sex = int(request.form['sex'])
        cp = int(request.form['cp'])
        trestbps = int(request.form['trestbps'])
        chol = int(request.form['chol'])
        fbs = int(request.form['fbs'])
        restecg = int(request.form['restecg'])
        thalach = int(request.form['thalach'])
        exang = int(request.form['exang'])
        oldpeak = float(request.form['oldpeak'])
        slope = int(request.form['slope'])
        ca = int(request.form['ca'])
        thal = int(request.form['thal'])
        
        # Model prediction
        features = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])
        prediction = model.predict(features)
        
        # Mapping result to readable text
        result = 'Heart Disease Detected' if prediction == 1 else 'No Heart Disease'
        
        return render_template('result.html', prediction=result)

if __name__ == "__main__":
    app.run(debug=True)
