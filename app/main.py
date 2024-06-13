import requests
import pandas as pd
import joblib

def main():
    # Descargar los datos del endpoint
    url = "http://localhost:8080/hakaton/v1/export-data/ocupancy"
    response = requests.get(url)
    
    # Verificar si la solicitud fue exitosa
    if response.status_code == 200:
        data = response.json()
        
        # Convertir los datos a un DataFrame de pandas
        df = pd.DataFrame(data)
        
        # Convertir la columna de fecha a formato datetime
        df['date'] = pd.to_datetime(df['date'])
        
        # Extraer características de fecha
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['day'] = df['date'].dt.day
        df['day_of_week'] = df['date'].dt.dayofweek
        
        # Extraer hotel_id desde el diccionario de texto
        df['hotel_id'] = df['hotel'].apply(lambda x: x['id'])
        
        # Eliminar la columna 'hotel' ya que no es necesaria
        df.drop(columns=['hotel'], inplace=True)
        
        # Imprimir las primeras filas del DataFrame para verificación
        print(df.head())
        
        # Guardar el DataFrame procesado
        df.to_csv('data/processed_occupancy_data.csv', index=False)
        
        print("Datos preprocesados y guardados correctamente.")
    else:
        print(f"Error al obtener los datos: {response.status_code}")


