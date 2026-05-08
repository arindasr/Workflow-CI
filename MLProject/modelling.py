import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import os

def train():
    print("Memulai proses otomatis re-training WeMovies AI di GitHub Actions...")
    
    df = pd.read_csv('data_bersih.csv')
    df = df.sample(n=1000, random_state=42) 
    
    X = df[['userId', 'movieId']]
    y = df['rating']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    with mlflow.start_run():
        rf = RandomForestRegressor(n_estimators=50, random_state=42)
        rf.fit(X_train, y_train)
        
        # Simpan model secara lokal di robot GitHub
        mlflow.sklearn.log_model(rf, "model")
        print("Model berhasil dilatih dan disimpan!")

if __name__ == "__main__":
    train()