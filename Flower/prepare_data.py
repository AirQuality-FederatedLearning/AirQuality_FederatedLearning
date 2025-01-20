import pandas as pd
import numpy as np
import os

def parse_dataset_name(filename):
    """
    Example function to parse dataset name.
    Adjust this logic as needed based on your naming convention.
    """
    parts = filename.split('_')
    # You mentioned an interval in parts[2], year in parts[3], place in parts[7:-2], etc.
    # Safely handle short arrays:
    if len(parts) < 8:
        return {"interval": None, "year": None, "place": filename}

    interval = parts[2]
    year = parts[3]
    # The place is ' '.join(...) of the portion you want:
    place = ' '.join(parts[7:-2])
    return {
        'interval': interval,
        'year': year,
        'place': place
    }

def load_dataset_and_split_by_month(csv_or_excel_path):
    """
    Load the dataset (CSV or Excel) with columns:
      Timestamp, PM2.5 (µg/m³), PM10 (µg/m³), NO (µg/m³), ...
      ...
    Then split the data by month.

    Returns:
        monthly_data (dict): {month_number: (X, Y)}
        input_dim (int): Number of features used
    """

    ext = os.path.splitext(csv_or_excel_path)[1].lower()
    if ext in [".csv"]:
        df = pd.read_csv(csv_or_excel_path)
    else:
        df = pd.read_excel(csv_or_excel_path)

    # Parse the timestamp
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
    df.dropna(subset=["Timestamp"], inplace=True)

    df.sort_values("Timestamp", inplace=True)

    # Fill NA with zeros or other strategy
    df = df.fillna(0)

    # Extract the month (1..12) for splitting
    df["Month"] = df["Timestamp"].dt.month

    # Example: we consider everything except pollutant columns as "features."
    # Or pick specific columns as you like:
    target_columns = [
        "PM2.5 (µg/m³)", "PM10 (µg/m³)",
        "NO (µg/m³)", "NO2 (µg/m³)", "NOx (ppb)", "NH3 (µg/m³)",
        "SO2 (µg/m³)", "CO (mg/m³)", "Ozone (µg/m³)",
        "Benzene (µg/m³)", "Toluene (µg/m³)",
        "Xylene (µg/m³)", "O Xylene (µg/m³)",
        "Eth-Benzene (µg/m³)", "MP-Xylene (µg/m³)",
    ]

    # Example set of features: (You can also choose to include pollutant columns as features if you prefer a forecasting approach.)
    # Here we assume meteorological + time-based columns are the input.
    feature_columns = [
        "AT (°C)", "RH (%)", "WS (m/s)", "WD (deg)",
        "RF (mm)", "TOT-RF (mm)", "SR (W/mt2)",
        "BP (mmHg)", "VWS (m/s)"
    ]

    # Make sure these columns exist (some might not exist in your dataset). 
    # For demonstration, let's do a quick intersection:
    feature_columns = [col for col in feature_columns if col in df.columns]

    # Build X and Y arrays
    X_all = df[feature_columns].values.astype(float)
    Y_all = df[target_columns].values.astype(float)

    # We might also add day/hour as numeric features:
    df["Day"] = df["Timestamp"].dt.day
    df["Hour"] = df["Timestamp"].dt.hour
    # Append them:
    X_all = np.concatenate([X_all, df[["Day", "Hour"]].values], axis=1)

    # Now we have a final X of dimension = len(feature_columns) + 2 for day/hour
    input_dim = X_all.shape[1]

    # Split by month
    monthly_data = {}
    for m in sorted(df["Month"].unique()):
        indices = df["Month"] == m
        X_m = X_all[indices]
        Y_m = Y_all[indices]
        monthly_data[m] = (X_m, Y_m)

    return monthly_data, input_dim
