# prepare_data.py

import pandas as pd
import numpy as np

def load_and_split_data(
    csv_path: str,
    n_clients: int = 3,
):
    """
    Loads time-series data from CSV, sorts by date, splits among n_clients.
    Each client gets a chunk of data. 
    We'll forcibly keep the last 20 rows from each chunk for test/validation.
    
    Returns:
        client_datasets: List of tuples (X_train, y_train, X_val, y_val) for each client
        features_dim: how many features (for the GRU input shape)
    """
    # 1) Load dataset
    df = pd.read_csv(csv_path)

    # 2) Parse the 'date' column. Adjust the column name if different.
    if "date" not in df.columns:
        raise ValueError("CSV must have a 'date' column for time series.")
    df["date"] = pd.to_datetime(df["date"])
    
    # 3) Sort by date ascending
    df.sort_values(by="date", inplace=True)

    # Example: we treat 'PM2.5' as the target for forecasting.
    # You can adapt for multi-step or multi-feature forecasting.
    if "pm2_5" not in df.columns:
        raise ValueError("Dataset must contain a 'PM2.5' column as the target.")

    # Let's drop 'date' from features (or transform it to numeric if you want).
    # We'll do a simple numeric transform example here:
    # year, month, day, hour to get some time-based features.
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day
    df["hour"] = df["date"].dt.hour

    # Drop the original date if you prefer
    df.drop(columns=["date"], inplace=True)

    # 4) For demonstration, the label is 'PM2.5'; features are all other columns.
    features = df.drop(columns=["pm2_5"]).values
    labels = df["pm2_5"].values

    total_len = len(df)
    chunk_size = total_len // n_clients

    client_datasets = []
    start_idx = 0
    for i in range(n_clients):
        end_idx = start_idx + chunk_size
        if i == n_clients - 1:
            end_idx = total_len  # last chunk takes the remainder

        X_chunk = features[start_idx:end_idx]
        y_chunk = labels[start_idx:end_idx]

        # We'll keep the last 20 rows in each chunk as "val/test" for that client
        if len(X_chunk) < 20:
            # If a chunk is smaller than 20, adapt or skip
            raise ValueError("Chunk size too small, not enough data for last 20 rows")

        split_point = len(X_chunk) - 20
        X_train, X_val = X_chunk[:split_point], X_chunk[split_point:]
        y_train, y_val = y_chunk[:split_point], y_chunk[split_point:]

        client_datasets.append((X_train, y_train, X_val, y_val))
        start_idx = end_idx

    # We'll assume all feature columns are numeric at this point
    features_dim = features.shape[1]
    
    return client_datasets, features_dim

def main():
    # Quick test
    client_data, feat_dim = load_and_split_data("new_delhi_aqi.csv", n_clients=3)
    print(f"Got {len(client_data)} clients' data. Feature dimension: {feat_dim}")
    for i, (X_train, y_train, X_val, y_val) in enumerate(client_data):
        print(f"Client {i}: Train size={len(X_train)}, Val size={len(X_val)}")

if __name__ == "__main__":
    main()
