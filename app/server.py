from flask import Flask, jsonify, request, render_template
import joblib
import numpy as np
import pandas as pd

from app import app

model = joblib.load('models/tourist_flow_model.pkl')

def make_predictions(year, month, hotel_id):
    predictions = []
    for day in range(1, 32):
        day_of_week = pd.Timestamp(f'{year}-{month}-{day}').dayofweek
        features = np.array([[year, month, day, day_of_week, hotel_id]])
        prediction = model.predict(features)[0]
        predictions.append({"date": f'{year}-{month}-{day}', "roomsOccupied": prediction})
    return predictions

@app.route('/predict', methods=['GET'])
def predict():
    try:
        year = request.args.get('year', default=2025, type=int)
        month = request.args.get('month', default=1, type=int)
        hotel_id = request.args.get('hotel_id', default=1, type=int)
        predictions = make_predictions(year, month, hotel_id)
        return jsonify(predictions)
    except Exception as e:
        print(f"Error in /predict: {str(e)}")
        return jsonify({"error": str(e)}), 500
    

@app.route('/historical', methods=['GET'])
def historical():
    try:
        df = pd.read_csv('data/processed_occupancy_data.csv')
        years = df['year'].unique()
        historical_data = []
        
        for year in years:
            year_data = df[df['year'] == year]
            total_rooms = year_data['roomsOccupied'].sum()
            total_available = year_data['quantityHabs'].sum() 
            historical_data.append({
                'year': int(year),
                'total_rooms': int(total_rooms),
                'total_available': int(total_available)
            })
        
        return jsonify(historical_data)
    except Exception as e:
        print(f"Error in /historical: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route('/')
def index():
    try:
        print("Rendering index.html")
        return render_template('index.html')
    except Exception as e:
        print(f"Error in rendering index.html: {str(e)}")
        return f"An error occurred: {str(e)}", 500


