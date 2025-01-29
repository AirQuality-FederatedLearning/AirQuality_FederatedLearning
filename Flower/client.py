import argparse
import yaml
import numpy as np
import pandas as pd
import flwr as fl
from flwr.client import NumPyClient
import tensorflow as tf
import os
from sklearn.preprocessing import MinMaxScaler

##############################################################################
# 1) Utilities: parse_dataset_name, model builder, etc.                     #
##############################################################################

def parse_dataset_name(filename):
    """
    Extract some metadata from the dataset name. 
    Adjust to suit your naming convention as needed.
    Example filename structure: 
      'Air_Quality_interval_1_year_2022_something_something_dataset.csv'
    """
    parts = filename.split('_')
    interval = parts[2]  # e.g. '1'
    year = parts[3]      # e.g. '2022'
    # Just an example of how to parse location info
    # You may need to adjust indexing based on your file naming pattern.
    place = ' '.join(parts[7:-2])  # e.g. 'somewhere else'
    return {'interval': interval, 'year': year, 'place': place}


def build_hybrid_mlp_lstm(input_dim, learning_rate):
    """
    Build a hybrid MLP+LSTM model that outputs 7 units 
    (one per pollutant).
    """
    inputs = tf.keras.Input(shape=(1, input_dim))
    
    # MLP branch
    mlp = tf.keras.layers.Dense(10, activation="tanh")(inputs)
    mlp = tf.keras.layers.Dense(5, activation="tanh")(mlp)
    mlp = tf.keras.layers.Dense(1)(mlp)
    mlp_output = tf.keras.layers.Reshape((1, 1))(mlp)

    # LSTM branch
    lstm = tf.keras.layers.LSTM(128, return_sequences=False)(inputs)
    lstm = tf.keras.layers.Dense(100, activation="tanh")(lstm)
    lstm_output = tf.keras.layers.Dense(1)(lstm)
    lstm_output = tf.keras.layers.Reshape((1, 1))(lstm_output)

    # Combine branches
    combined = tf.keras.layers.Concatenate(axis=-1)([mlp_output, lstm_output])
    
    # Final Dense to produce 7 outputs (for 7 pollutants)
    combined_output = tf.keras.layers.Dense(7)(combined)

    model = tf.keras.Model(inputs=inputs, outputs=combined_output)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mse",
        metrics=["mae"]
    )
    return model


##############################################################################
# 2) Client Class Definition                                                 #
##############################################################################
class TimeSeriesClient(NumPyClient):
    def __init__(
        self,
        data,
        input_dim,
        learning_rate,
        epochs,
        batch_size,
        output_dir,
        dataset_path
    ):
        super().__init__()
        self.epochs = epochs
        self.batch_size = batch_size
        self.output_dir = output_dir
        self.data = data
        self.input_dim = input_dim
        self.dataset_path = dataset_path

        # Build the same model architecture
        self.model = build_hybrid_mlp_lstm(
            input_dim=input_dim,
            learning_rate=learning_rate
        )

        os.makedirs(self.output_dir, exist_ok=True)

    def get_parameters(self, config):
        return self.model.get_weights()

    def set_parameters(self, parameters, config):
        self.model.set_weights(parameters)

    def fit(self, parameters, config):
        self.set_parameters(parameters, config)
        X_train, y_train = self.data["train"]
        self.model.fit(
            X_train, y_train,
            epochs=self.epochs,
            batch_size=self.batch_size,
            verbose=0
        )
        return self.get_parameters(config), len(X_train), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters, config)
        X_test, y_test = self.data["test"]
        loss, mae = self.model.evaluate(X_test, y_test, verbose=0)

        # Make predictions
        predictions = self.model.predict(X_test)

        # predictions shape: (num_samples, 1, 7) or (num_samples, 7)
        # After the final Dense(7), Keras typically returns (num_samples, 7).
        # But because of the Reshape layers, you might see an additional dimension.
        # Let’s handle that carefully:
        if len(predictions.shape) == 3:
            # e.g., (num_samples, 1, 7)
            predictions = predictions.reshape(predictions.shape[0], predictions.shape[2])
        # Now predictions is (num_samples, 7).

        # y_test is (num_samples, 7). Just confirm shape:
        if len(y_test.shape) == 3:
            y_test = y_test.reshape(y_test.shape[0], y_test.shape[2])

        # Inverse scale
        scaler = self.data["scaler"]
        predictions_original = scaler.inverse_transform(predictions)
        y_test_original = scaler.inverse_transform(y_test)

        # Pollutants list
        pollutants = [
            "PM2.5 (µg/m³)", 
            "PM10 (µg/m³)", 
            "NO (µg/m³)", 
            "NO2 (µg/m³)", 
            "SO2 (µg/m³)", 
            "CO (mg/m³)", 
            "Ozone (µg/m³)"
        ]

        # Parse dataset name to differentiate output files
        parsed_info = parse_dataset_name(os.path.basename(self.dataset_path))
        place = parsed_info['place']  # Or incorporate interval/year if desired

        # Save separate CSV for each pollutant
        rows_per_file = 15  # or modify as needed

        for j, pollutant in enumerate(pollutants):
            data_rows = []
            # We'll limit to rows_per_file just for demonstration. 
            # If you want all rows, replace min(...) with the entire length.
            for i in range(min(rows_per_file, len(predictions_original))):
                pred_val = predictions_original[i][j]
                actual_val = y_test_original[i][j]
                mse_val = (actual_val - pred_val) ** 2
                data_rows.append({
                    f"{pollutant} Predicted": pred_val,
                    f"{pollutant} Actual": actual_val,
                    f"{pollutant} MSE": mse_val
                })

            df = pd.DataFrame(data_rows)
            # Example file naming: "<place>_<pollutant_cleaned>.csv"
            pollutant_cleaned = pollutant.replace(' ', '_').replace('/', '_')
            csv_path = os.path.join(
                self.output_dir, 
                f"{place}_{pollutant_cleaned}.csv"
            )
            df.to_csv(csv_path, index=False)
            print(f"Saved {pollutant} data to {csv_path}")

        # Return the final metrics from evaluate
        return float(loss), len(X_test), {"mae": float(mae)}


##############################################################################
# 3) Preprocessing Function                                                  #
##############################################################################
def preprocess_data(dataset_path):
    """
    1. Load the CSV
    2. Keep only the necessary columns (Timestamp + 7 pollutant columns)
    3. Drop NaNs
    4. Scale data
    5. Split into train/test
    6. Reshape for input to (batch, 1, 7) so LSTM has time dimension = 1
    7. Return dictionary with train/test/scaler
    """
    df = pd.read_csv(dataset_path)
    pollutants = [
        "PM2.5 (µg/m³)", 
        "PM10 (µg/m³)", 
        "NO (µg/m³)", 
        "NO2 (µg/m³)", 
        "SO2 (µg/m³)", 
        "CO (mg/m³)", 
        "Ozone (µg/m³)"
    ]

    # Keep only relevant columns
    df = df[["Timestamp"] + pollutants].dropna()

    # Convert Timestamp to datetime, set as index (optional but typical for timeseries)
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    df = df.set_index("Timestamp")

    values = df.values  # shape (num_samples, 7)

    # Scale data
    scaler = MinMaxScaler()
    values_scaled = scaler.fit_transform(values)

    # Train/test split, e.g., 85% train
    train_size = int(0.85 * len(values_scaled))
    train_data = values_scaled[:train_size]
    test_data = values_scaled[train_size:]

    # We predict the next step from the current step
    X_train, y_train = train_data[:-1], train_data[1:]
    X_test, y_test = test_data[:-1], test_data[1:]

    # Reshape for LSTM (batch_size, time_steps=1, input_dim=7)
    X_train = np.expand_dims(X_train, axis=1)
    X_test = np.expand_dims(X_test, axis=1)

    # y_train and y_test remain (n_samples, 7)
    return {
        "train": (X_train, y_train),
        "test": (X_test, y_test),
        "scaler": scaler
    }, X_train.shape[-1]


##############################################################################
# 4) Main: Start the FL Client                                              #
##############################################################################
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to the dataset."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="client_config.yaml",
        help="Path to YAML config."
    )
    args = parser.parse_args()

    # Load hyperparams from YAML
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    learning_rate = config["model"]["learning_rate"]
    epochs = config["model"]["epochs"]
    batch_size = config["model"]["batch_size"]
    output_dir = config["output_dir"]

    # Preprocess local dataset
    data, input_dim = preprocess_data(args.dataset)

    # Create the FL client
    client = TimeSeriesClient(
        data=data,
        input_dim=input_dim,
        learning_rate=learning_rate,
        epochs=epochs,
        batch_size=batch_size,
        output_dir=output_dir,
        dataset_path=args.dataset
    )

    # Start the Flower client
    fl.client.start_numpy_client(
        server_address="127.0.0.1:8080",
        client=client
    )


if __name__ == "__main__":
    main()
