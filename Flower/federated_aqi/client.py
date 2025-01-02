# client.py

import hydra
from omegaconf import DictConfig
import flwr as fl
import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
import requests

def build_model(input_dim: int = 10):
    """Build the same architecture as the server."""
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Reshape((1, input_dim), input_shape=(input_dim,)))
    model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)))
    model.add(tf.keras.layers.GRU(32))
    model.add(tf.keras.layers.Dense(1))
    model.compile(optimizer="adam", loss="mean_squared_error", metrics=["mae"])
    return model

def load_and_preprocess_data(path: str):
    """Load and preprocess the air quality dataset."""
    df = pd.read_csv(path)
    df.dropna(inplace=True)

    # Suppose your dataset columns are: [PM2.5, PM10, SO2, NO2, CO, O3, TEMP, PRES, DEWP, WSPM, etc...]
    # For demonstration, let's pick 10 features: PM2.5, PM10, SO2, NO2, CO, O3, TEMP, PRES, DEWP, WSPM
    # Then define a simple 'AQI' target as a linear combo or any function.
    # Adjust columns as your dataset demands.
    features = ["pm2.5", "pm10", "so2", "no2", "co", "o3", "temp", "pres", "dew", "wspd"]
    # If your dataset has different names, rename them accordingly:
    df.columns = [col.lower() for col in df.columns]

    # Filter columns, handle any missing or rename if needed
    df = df.dropna(subset=features)

    X = df[features].values

    # Example: Simple linear combination as your "AQI" (very simplistic):
    # In reality, you'd compute official AQI based on breakpoints, etc.
    df["AQI"] = (
        0.4 * df["pm2.5"] +
        0.3 * df["pm10"] +
        0.2 * df["so2"] +
        0.1 * df["no2"]  # Just an example
    )
    y = df["AQI"].values

    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    return X_train, X_val, y_train, y_val

class AQIClient(fl.client.NumPyClient):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.model = build_model(input_dim=10)
        # Load data
        self.X_train, self.X_val, self.y_train, self.y_val = load_and_preprocess_data(cfg.dataset.path)

    def get_parameters(self):
        """Return current weights of the local model."""
        return self.model.get_weights()

    def fit(self, parameters, config):
        """Train locally on the client's dataset."""
        self.model.set_weights(parameters)

        # Train
        self.model.fit(
            self.X_train,
            self.y_train,
            epochs=self.cfg.training.epochs,
            batch_size=self.cfg.training.batch_size,
            validation_split=self.cfg.training.validation_split,
            verbose=1,
        )
        return self.model.get_weights(), len(self.X_train), {}

    def evaluate(self, parameters, config):
        """Evaluate the model on the local validation set."""
        self.model.set_weights(parameters)
        loss, mae = self.model.evaluate(self.X_val, self.y_val, verbose=0)
        # Let's define 'accuracy' as 1 - mae for demonstration
        # This is not a typical definition, but shows how Flower aggregator can interpret it.
        accuracy = 1.0 - mae
        return float(loss), len(self.X_val), {"accuracy": accuracy}


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    # Start Flower client
    fl.client.start_numpy_client(
        server_address=cfg.server.server_address,
        client=AQIClient(cfg),
    )

if __name__ == "__main__":
    main()
