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
    def __init__(
        self, 
        X_train, y_train, 
        X_val, y_val, 
        input_dim,
        hidden_units,
        learning_rate,
        epochs,
        batch_size
    ):
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
        self.model.save("client_model2.h5")

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
    # City argument: "delhi" or "covai"
    parser.add_argument("--city", type=str, default="delhi", help="City for dataset (delhi or covai)")
    # Client index within that city (0 or 1, since each city has 2 clients)
    parser.add_argument("--idx", type=int, default=0, help="Client index (0 or 1)")
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

    # For convenience, store possible CSV paths in the config
    # Example structure in YAML:
    #
    # data:
    #   csv_paths:
    #       delhi: "new_delhi_aqi.csv"
    #       covai: "covai_aqi.csv"
    #
    data_config = config_data["data"]
    csv_paths = data_config.get("csv_paths", {})
    
    # 2) Select CSV path based on city
    city_lower = args.city.lower()
    if city_lower not in csv_paths:
        raise ValueError(f"Unsupported city '{args.city}'. Please specify 'delhi' or 'covai'.")
    csv_path = csv_paths[city_lower]

    # 3) Load data for this city. We assume each city has 2 clients => n_clients=2
    client_data, input_dim = load_and_split_data(
        csv_path=csv_path,
        n_clients=2  # Each city is split into 2 clients
    )
    
    # Check that the client index is valid
    if args.idx not in [0, 1]:
        raise ValueError("Client index must be 0 or 1 for each city.")
    
    X_train, y_train, X_val, y_val = client_data[args.idx]

    # 4) Create FL client
    fl_client = TimeSeriesClient(
        X_train, y_train,
        X_val, y_val,
        input_dim,
        hidden_units,
        learning_rate,
        epochs,
        batch_size
    )

    # 5) Start the client
    fl.client.start_client(
        server_address="127.0.0.1:8080",
        client=fl_client.to_client(),
    )

if __name__ == "__main__":
    main()
