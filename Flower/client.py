import argparse
import yaml
import numpy as np
import pandas as pd
import flwr as fl
from flwr.client import NumPyClient
import tensorflow as tf
import os
from sklearn.preprocessing import MinMaxScaler


def parse_dataset_name(filename):
    """
    Extracts the interval, year, and place from the dataset filename.
    """
    parts = filename.split('_')
    interval = parts[2]
    year = parts[3]
    place = ' '.join(parts[7:-2])
    return {'interval': interval, 'year': year, 'place': place}


def build_hybrid_mlp_lstm(input_dim, learning_rate):
    """
    Build the hybrid MLP-LSTM model.
    """
    inputs = tf.keras.Input(shape=(1, input_dim))
    # MLP Block
    mlp = tf.keras.layers.Dense(10, activation="tanh")(inputs)
    mlp = tf.keras.layers.Dense(5, activation="tanh")(mlp)
    mlp = tf.keras.layers.Dense(1)(mlp)
    mlp_output = tf.keras.layers.Reshape((1, 1))(mlp)

    # LSTM Block
    lstm = tf.keras.layers.LSTM(128, return_sequences=False)(inputs)
    lstm = tf.keras.layers.Dense(100, activation="tanh")(lstm)
    lstm_output = tf.keras.layers.Dense(1)(lstm)

    # Combine
    combined = tf.keras.layers.Concatenate(axis=-1)([mlp_output, tf.keras.layers.Reshape((1, 1))(lstm_output)])
    combined_output = tf.keras.layers.Dense(1)(combined)

    model = tf.keras.Model(inputs=inputs, outputs=combined_output)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss="mse", metrics=["mae"])
    return model


class TimeSeriesClient(NumPyClient):
    def __init__(self, data, input_dim, learning_rate, epochs, batch_size, output_dir, dataset_path):
        super().__init__()
        self.epochs = epochs
        self.batch_size = batch_size
        self.output_dir = output_dir
        self.data = data
        self.input_dim = input_dim
        self.dataset_path = dataset_path

        # Build the model
        self.model = build_hybrid_mlp_lstm(input_dim=input_dim, learning_rate=learning_rate)
        os.makedirs(self.output_dir, exist_ok=True)

    def get_parameters(self, config):
        return self.model.get_weights()

    def set_parameters(self, parameters, config):
        self.model.set_weights(parameters)

    def fit(self, parameters, config):
        self.set_parameters(parameters, config)
        X_train, y_train = self.data["train"]
        self.model.fit(X_train, y_train, epochs=self.epochs, batch_size=self.batch_size, verbose=0)
        return self.get_parameters(config), len(X_train), {}
    def evaluate(self, parameters, config):
        # Set model parameters
        self.set_parameters(parameters, config)

        # Get test data
        X_test, y_test = self.data["test"]

        # Evaluate the model
        loss, mae = self.model.evaluate(X_test, y_test, verbose=0)

        # Get predictions
        predictions = self.model.predict(X_test).flatten()

        # Handle shape mismatch
        if len(predictions.shape) == 1 and len(y_test.shape) == 2:
            y_test = y_test[:, 0]  # Use only the first column of y_test

        # Ensure shapes match
        if len(predictions) != len(y_test):
            min_len = min(len(predictions), len(y_test))
            predictions = predictions[:min_len]
            y_test = y_test[:min_len]

        # Assuming pollutants is a list of the columns you're predicting
        pollutants = ["PM2.5 (µg/m³)", "PM10 (µg/m³)", "NO (µg/m³)", "NO2 (µg/m³)", "SO2 (µg/m³)", "CO (mg/m³)", "Ozone (µg/m³)"]

        # Prepare the results dictionary
        results = {
            "Loss": loss,
            "MAE": mae,
        }

        # Add columns for predicted, actual, and MSE for each pollutant
        for i, pollutant in enumerate(pollutants):
            if i < len(predictions):
                results[f"{pollutant} Predicted"] = predictions[i]
                results[f"{pollutant} Actual"] = y_test[i]
                results[f"{pollutant} MSE"] = np.square(y_test[i] - predictions[i])

        # Create DataFrame for logging
        results_df = pd.DataFrame(results, index=[0])

        # Extract place from the dataset name
        dataset_name = os.path.basename(self.dataset_path)
        dataset_info = parse_dataset_name(dataset_name)
        place = dataset_info['place']

        # Save results to a CSV file for each place
        csv_filename = f"{place}_evaluation_results.csv"
        csv_path = os.path.join(self.output_dir, csv_filename)
        results_df.to_csv(csv_path, index=False)

        print(f"Evaluation results for {place} saved at {csv_path}")

        return float(loss), len(X_test), {"mae": float(mae)}

def preprocess_data(dataset_path):
    df = pd.read_csv(dataset_path)
    pollutants = ["PM2.5 (µg/m³)", "PM10 (µg/m³)", "NO (µg/m³)", "NO2 (µg/m³)", "SO2 (µg/m³)", "CO (mg/m³)", "Ozone (µg/m³)"]
    df = df[["Timestamp"] + pollutants].dropna()
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    df = df.set_index("Timestamp")
    values = df.values

    # Normalize the data
    scaler = MinMaxScaler()
    values_scaled = scaler.fit_transform(values)

    # Prepare train/test data
    train_size = int(0.85 * len(values_scaled))
    train_data, test_data = values_scaled[:train_size], values_scaled[train_size:]
    X_train, y_train = train_data[:-1], train_data[1:]
    X_test, y_test = test_data[:-1], test_data[1:]

    # Reshape for LSTM input
    X_train = np.expand_dims(X_train, axis=1)
    X_test = np.expand_dims(X_test, axis=1)
    return {"train": (X_train, y_train), "test": (X_test, y_test)}, X_train.shape[-1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="Path to the dataset.")
    parser.add_argument("--config", type=str, default="client_config.yaml", help="Path to YAML config.")
    args = parser.parse_args()

    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    learning_rate = config["model"]["learning_rate"]
    epochs = config["model"]["epochs"]
    batch_size = config["model"]["batch_size"]
    output_dir = config["output_dir"]

    # Preprocess dataset
    data, input_dim = preprocess_data(args.dataset)

    # Initialize and start the client
    client = TimeSeriesClient(data=data, input_dim=input_dim, learning_rate=learning_rate, epochs=epochs, batch_size=batch_size, output_dir=output_dir, dataset_path=args.dataset)
    fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=client)


if __name__ == "__main__":
    main()
