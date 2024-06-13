import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import joblib
import numpy as np

def train_model():
    # Cargar los datos preprocesados
    df = pd.read_csv('data/processed_occupancy_data.csv')

    # Seleccionar las características (features) y la variable objetivo (target)
    X = df[['year', 'month', 'day', 'day_of_week', 'hotel_id']]
    y = df['roomsOccupied']
    
    # Dividir los datos en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Entrenar el modelo de regresión
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluar el modelo
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f'Mean Squared Error: {mse}')
    
    # Guardar el modelo entrenado
    joblib.dump(model, 'models/tourist_flow_model.pkl')
    print("Modelo entrenado y guardado correctamente.")

def predict_occupancy(year):
    # Cargar el modelo entrenado
    model = joblib.load('models/tourist_flow_model.pkl')
    
    predictions = []
    for hotel_id in range(1, 6):  # 5 hoteles
        for month in [1]:  # Enero
            for day in range(1, 32):  # Días de Enero
                day_of_week = pd.Timestamp(f'{year}-{month}-{day}').dayofweek
                features = np.array([[year, month, day, day_of_week, hotel_id]])
                predicted_occupancy = model.predict(features)[0]
                predictions.append({
                    'hotel_id': hotel_id,
                    'date': f'{year}-{month:02}-{day:02}',
                    'predicted_occupancy': predicted_occupancy
                })
    return predictions

def main():
    train_model()
    predictions = predict_occupancy(2025)
    
    # Convertir las predicciones a un DataFrame de pandas
    predictions_df = pd.DataFrame(predictions)
    
    # Guardar las predicciones en un archivo CSV
    predictions_df.to_csv('data/predicted_occupancy_2025.csv', index=False)
    print("Predicciones para el año 2025 guardadas correctamente.")


