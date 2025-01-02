import argparse
import yaml
import numpy as np
import flwr as fl
from flwr.client import NumPyClient
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import mean_squared_error

from prepare_data import load_and_split_data

def build_model(input_dim: int, hidden_units: int, learning_rate: float) -> keras.Model:
    """
    Build a simple GRU model for a regression task on time series.
    """
    model = keras.Sequential()
    model.add(layers.Input(shape=(1, input_dim)))
    model.add(layers.GRU(hidden_units))
    model.add(layers.Dense(1))
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mse"
    )
    return model

class TimeSeriesClient(NumPyClient):
    def __init__(self, 
                 X_train, y_train, 
                 X_val, y_val, 
                 input_dim,
                 hidden_units,
                 learning_rate,
                 epochs,
                 batch_size):
        super().__init__()
        self.epochs = epochs
        self.batch_size = batch_size
        
        self.model = build_model(input_dim, hidden_units, learning_rate)
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val

    def get_parameters(self, config):
        return self.model.get_weights()

    def set_parameters(self, parameters, config):
        self.model.set_weights(parameters)

    def fit(self, parameters, config):
        self.set_parameters(parameters, config)

        X_train_reshaped = np.expand_dims(self.X_train, axis=1)
        self.model.fit(
            X_train_reshaped,
            self.y_train,
            epochs=self.epochs,
            batch_size=self.batch_size,
            verbose=0
        )

        return self.get_parameters(config), len(self.X_train), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters, config)

        X_val_reshaped = np.expand_dims(self.X_val, axis=1)
        predictions = self.model.predict(X_val_reshaped, verbose=0)
        mse = mean_squared_error(self.y_val, predictions)

        deviation = predictions.flatten() - self.y_val
        min_dev = float(np.min(deviation))
        max_dev = float(np.max(deviation))

        return float(mse), len(self.X_val), {
            "min_deviation": min_dev,
            "max_deviation": max_dev
        }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--idx", type=int, default=0, help="Client index")
    parser.add_argument("--clients", type=int, default=3, help="Number of total clients")
    parser.add_argument("--config", type=str, default="client_config.yaml", help="Path to YAML config")
    args = parser.parse_args()

    # 1) Load config from YAML
    with open(args.config, "r") as f:
        config_data = yaml.safe_load(f)

    model_config = config_data["model"]
    # e.g. hidden_units=16, learning_rate=0.001, epochs=5, batch_size=32
    hidden_units = model_config.get("hidden_units", 16)
    learning_rate = model_config.get("learning_rate", 0.001)
    epochs = model_config.get("epochs", 3)
    batch_size = model_config.get("batch_size", 16)

    data_config = config_data["data"]
    csv_path = data_config.get("csv_path", "new_delhi_aqi.csv")

    # 2) Load data for all clients
    client_data, input_dim = load_and_split_data(
        csv_path=csv_path,
        n_clients=args.clients
    )
    X_train, y_train, X_val, y_val = client_data[args.idx]

    # 3) Create FL client
    fl_client = TimeSeriesClient(
        X_train, y_train,
        X_val, y_val,
        input_dim,
        hidden_units,
        learning_rate,
        epochs,
        batch_size
    )

    # 4) Start the client
    fl.client.start_client(
        server_address="127.0.0.1:8080",
        client=fl_client.to_client(),
    )

if __name__ == "__main__":
    main()
