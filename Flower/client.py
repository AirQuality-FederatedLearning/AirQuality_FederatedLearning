import argparse
import yaml
import numpy as np
import pandas as pd
import flwr as fl
from flwr.client import NumPyClient
import tensorflow as tf
import os

from prepare_data import (
    load_dataset_and_split_by_month,
    parse_dataset_name,
)

def build_multioutput_model(input_dim: int, hidden_units: int, learning_rate: float, num_outputs: int) -> tf.keras.Model:
    """
    Build a multi-output GRU model for predicting multiple pollutants.
    """
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=(1, input_dim)))
    model.add(tf.keras.layers.GRU(hidden_units))
    model.add(tf.keras.layers.Dense(num_outputs))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mse",
    )
    return model

class TimeSeriesClient(NumPyClient):
    def __init__(
        self,
        monthly_data,        # Dict[month -> (X, y)]
        input_dim,
        hidden_units,
        learning_rate,
        num_outputs,
        place_name,
        epochs,
        batch_size,
        output_dir
    ):
        super().__init__()
        self.epochs = epochs
        self.batch_size = batch_size
        self.place_name = place_name

        # Build the model initially
        self.model = build_multioutput_model(
            input_dim, hidden_units, learning_rate, num_outputs
        )

        # monthly_data is a dict: {1: (X1, y1), 2: (X2, y2), ...}
        # which we’ll slice based on the current round number
        self.monthly_data = monthly_data
        self.months_sorted = sorted(self.monthly_data.keys())  # e.g. [1,2,...,12]
        self.num_months = len(self.months_sorted)
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def get_parameters(self, config):
        return self.model.get_weights()

    def set_parameters(self, parameters, config):
        self.model.set_weights(parameters)

    def fit(self, parameters, config):
        """Train on data for the round’s month, then save predictions in an Excel sheet."""
        self.set_parameters(parameters, config)

        # Identify which round we are in
        current_round = config.get("current_round", 1)
        # Or sometimes it's just "round"
        if "round" in config:
            current_round = config["round"]

        # Determine which month to use based on current_round
        # If we exceed the available months, we can cycle or just pick last month
        if current_round <= self.num_months:
            month_key = self.months_sorted[current_round - 1]
        else:
            # For indefinite or large rounds, you can cycle through months
            idx = (current_round - 1) % self.num_months
            month_key = self.months_sorted[idx]

        X_month, y_month = self.monthly_data[month_key]

        # Reshape for GRU: (samples, 1, features)
        X_month_reshaped = np.expand_dims(X_month, axis=1)

        # Fit
        self.model.fit(
            X_month_reshaped,
            y_month,
            epochs=self.epochs,
            batch_size=self.batch_size,
            verbose=0
        )

        # Create predictions for logging
        preds = self.model.predict(X_month_reshaped, verbose=0)

        # Save a local Excel/CSV with predicted vs. actual
        self.save_predictions(month_key, X_month, y_month, preds, current_round)

        # Save the updated local model with place suffix
        local_model_path = os.path.join(self.output_dir, f"client_{self.place_name}_round_{current_round}.h5")
        self.model.save(local_model_path)

        # Return updated weights
        return self.get_parameters(config), len(X_month), {}

    def evaluate(self, parameters, config):
        """In this example, we’ll just do a trivial evaluate (or we can do an out-of-sample)."""
        self.set_parameters(parameters, config)
        # Optionally, pick some data for evaluation
        # For simplicity, evaluate on the last month
        last_month_key = self.months_sorted[-1]
        X_eval, y_eval = self.monthly_data[last_month_key]
        X_eval_reshaped = np.expand_dims(X_eval, axis=1)

        preds = self.model.predict(X_eval_reshaped, verbose=0)
        mse = np.mean((preds - y_eval)**2)
        return float(mse), len(X_eval), {"mse": float(mse)}

    def save_predictions(self, month_key, X, y_true, y_pred, current_round):
        """
        Save an Excel sheet with columns in this order:
        Pollutant_1_actual, Pollutant_1_pred,
        Pollutant_2_actual, Pollutant_2_pred,
        ...
        Pollutant_n_actual, Pollutant_n_pred,
        Pollutant_1_error, Pollutant_2_error, ..., Pollutant_n_error
        """
        import pandas as pd
        import os
        errors = y_pred - y_true

        # We build a dictionary of columns in the exact sequence we want.
        num_pollutants = y_true.shape[1]

        # Ordered dictionary of columns (Python 3.7+ preserves insertion order)
        cols = {}

        # First add Actual and Predicted columns side by side.
        for i in range(num_pollutants):
            cols[f"Pollutant_{i+1}_actual"] = y_true[:, i]
            cols[f"Pollutant_{i+1}_pred"]   = y_pred[:, i]

        # Then add the error columns at the end.
        for i in range(num_pollutants):
            cols[f"Pollutant_{i+1}_error"] = errors[:, i]

        # Create DataFrame with the desired column order
        df_result = pd.DataFrame(cols)

        # Construct file path for output
        file_path = os.path.join(
            self.output_dir, 
            f"predictions_{self.place_name}_month_{month_key}_round_{current_round}.xlsx"
        )
        df_result.to_excel(file_path, index=False)
        print(f"[Client {self.place_name}] Saved predictions to {file_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="client_config.yaml", help="Path to YAML config.")
    parser.add_argument("--dataset", type=str, required=True, help="Path to the CSV/Excel dataset.")
    parser.add_argument("--idx", type=int, default=0, help="Client index (if you want multiple clients per dataset).")
    parser.add_argument("--place", type=str, default=None, help="Optional place override.")
    args = parser.parse_args()

    # 1) Load config from YAML
    with open(args.config, "r") as f:
        config_data = yaml.safe_load(f)

    model_config = config_data["model"]
    hidden_units = model_config.get("hidden_units", 16)
    learning_rate = model_config.get("learning_rate", 0.001)
    epochs = model_config.get("epochs", 1)
    batch_size = model_config.get("batch_size", 16)
    num_outputs = model_config.get("num_outputs", 15)  # e.g. 15 pollutants

    output_dir = config_data.get("output_dir", "./client_outputs")

    # 2) Parse dataset name to get place if not provided
    if args.place is None:
        name_parts = parse_dataset_name(os.path.basename(args.dataset))
        # parse_dataset_name returns { interval, year, place }, etc.
        place_name = name_parts["place"]
    else:
        place_name = args.place

    # 3) Load monthly-split data
    monthly_data, input_dim = load_dataset_and_split_by_month(args.dataset)

    # 4) Create FL client
    fl_client = TimeSeriesClient(
        monthly_data=monthly_data,
        input_dim=input_dim,
        hidden_units=hidden_units,
        learning_rate=learning_rate,
        num_outputs=num_outputs,
        place_name=place_name,
        epochs=epochs,
        batch_size=batch_size,
        output_dir=output_dir,
    )

    # 5) Start the client
    fl.client.start_client(
        server_address="127.0.0.1:8080",
        client=fl_client,
    )


if __name__ == "__main__":
    main()
