import pandas as pd
import numpy as np

def load_and_split_data(
    csv_path: str,
    n_clients: int = 1,
):
    """
    Loads time-series data from Excel/CSV, sorts by date, splits among n_clients.
    Each client gets a chunk of data.
    We'll forcibly keep the last 20 rows from each chunk for test/validation.

    Returns:
        client_datasets: List of tuples (X_train, y_train, X_val, y_val) for each client
        features_dim: how many features (for the GRU input shape)
    """

    # 1) Load dataset
    df = pd.read_excel(csv_path)

    # 2) Rename and check required columns
    rename_map = {
        "From Date": "from_date",
        "To Date": "to_date",
        "PM2.5": "pm2_5",
        "PM10": "pm10",
        "NO2": "no2",
        "SO2": "so2",
        "CO": "co",
        "Ozone": "ozone",
    }
    df.rename(columns=rename_map, inplace=True)

    required_columns = ["from_date", "pm2_5"]
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Dataset must contain a '{col}' column.")

    # 3) Parse "from_date" as the primary datetime column
    df["from_date"] = pd.to_datetime(df["from_date"], errors="coerce")

    # Drop rows with invalid dates
    df = df.dropna(subset=["from_date"])

    # 4) Sort by date ascending
    df.sort_values(by="from_date", inplace=True)

    # 5) Fill missing values
    df.fillna(method="ffill", inplace=True)  # Forward-fill missing values
    df.fillna(method="bfill", inplace=True)  # Backward-fill if needed

    # 6) Extract time features
    df["year"] = df["from_date"].dt.year
    df["month"] = df["from_date"].dt.month
    df["day"] = df["from_date"].dt.day
    df["hour"] = df["from_date"].dt.hour

    # Drop unnecessary columns
    df.drop(columns=["from_date", "to_date"], inplace=True, errors="ignore")

    # 7) Separate features from the target (pm2_5)
    features = df.drop(columns=["pm2_5"]).values
    labels = df["pm2_5"].values

    total_len = len(df)
    if total_len < n_clients * 20:
        raise ValueError(
            "Not enough data to allocate 20 validation rows per client."
        )

    chunk_size = total_len // n_clients

    client_datasets = []
    start_idx = 0
    for i in range(n_clients):
        end_idx = start_idx + chunk_size
        if i == n_clients - 1:
            # last chunk takes the remainder
            end_idx = total_len

        X_chunk = features[start_idx:end_idx]
        y_chunk = labels[start_idx:end_idx]

        # We'll keep the last 20 rows in each chunk as "val/test" for that client
        if len(X_chunk) < 20:
            raise ValueError(
                f"Chunk {i} size too small ({len(X_chunk)}), not enough data for the last 20 rows."
            )

        split_point = len(X_chunk) - 20
        X_train, X_val = X_chunk[:split_point], X_chunk[split_point:]
        y_train, y_val = y_chunk[:split_point], y_chunk[split_point:]

        client_datasets.append((X_train, y_train, X_val, y_val))
        start_idx = end_idx

    features_dim = features.shape[1]

    return client_datasets, features_dim

def main():
    # Quick test for Covai and Delhi datasets
    covai_data, covai_feat_dim = load_and_split_data("cpcb_covai_aqi.xlsx", n_clients=2)
    print(f"Covai Data: Feature dimension: {covai_feat_dim}")
    for i, (X_train, y_train, X_val, y_val) in enumerate(covai_data):
        print(f"Client {i} (Covai): Train size={len(X_train)}, Val size={len(X_val)}")

    delhi_data, delhi_feat_dim = load_and_split_data("cpcb_delhi_aqi.xlsx", n_clients=2)
    print(f"Delhi Data: Feature dimension: {delhi_feat_dim}")
    for i, (X_train, y_train, X_val, y_val) in enumerate(delhi_data):
        print(f"Client {i} (Delhi): Train size={len(X_train)}, Val size={len(X_val)}")

if __name__ == "__main__":
    main()
