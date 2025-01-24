import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dropout, Dense

def create_dataset(data, look_back=10):
    X, y = [], []
    for i in range(len(data) - look_back):
        X.append(data[i:(i + look_back)])
        y.append(data[i + look_back])
    return np.array(X), np.array(y)

# Define GRU model
def build_gru_model(input_shape):
    model = Sequential([
        GRU(64, activation='relu', return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        GRU(64, activation='relu'),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

def preprocess_data(file_path, look_back=10):
    # Load the dataset
    df = pd.read_csv(file_path)
    
    # Handle missing values by filling them with the mean of the column
    df.fillna(df.mean(), inplace=True)
    
    # Normalize the dataset
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df.values)
    
    # Create the dataset
    X, y = create_dataset(scaled_data, look_back)
    
    return X, y, scaler

def forecast(file_path, parameter, city):
    look_back = 10
    X, y, scaler = preprocess_data(file_path, look_back)
    
    # Split the data into training and testing sets
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # Reshape input to be 3D [samples, time steps, features]
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
    
    # Build and train model
    model = build_gru_model((look_back, 1))
    model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=1)
    
    # Save the model
    model_save_path = f"C:/Users/DELL/Documents/Amrita/4th year/AirQuality_FederatedLearning/Flower/flower_demo/models/{city}_{parameter}_timeseries_model.h5"
    model.save(model_save_path)
    print(f"Model saved to {model_save_path}")
    
    # Forecast
    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(predictions)
    y_test = scaler.inverse_transform(y_test.reshape(-1, 1))
    
    # Calculate MSE
    mse = mean_squared_error(y_test, predictions)
    
    # Prepare results
    forecasted_values = predictions.flatten()
    actual_values = y_test.flatten()
    
    return forecasted_values, actual_values, mse

def train_and_test_all_variables():
    results = {}
    # Assuming you have a list of file paths, parameters, and cities
    file_paths = ["path/to/data1.csv", "path/to/data2.csv"]
    parameters = ["PM2.5", "PM10"]
    cities = ["Coimbatore", "Delhi"]
    
    for file_path, parameter, city in zip(file_paths, parameters, cities):
        forecasted_values, actual_values, mse = forecast(file_path, parameter, city)
        if forecasted_values is None or actual_values is None or mse is None:
            print(f"Skipping {parameter} in {city} due to NaN values.")
            continue
        if city not in results:
            results[city] = {}
        results[city][parameter] = {
            "forecasted_values": forecasted_values,
            "actual_values": actual_values,
            "mse": mse
        }
    
    return results

def custom_test(results):
    print("\nAvailable cities: Coimbatore, Delhi")
    city = input("Enter the city: ")
    print("Available parameters: PM2.5, PM10, NO2, SO2, CO, Ozone")
    parameter = input("Enter the parameter to test: ")

    if city in results and parameter in results[city]:
        print(f"Forecasted values for {parameter} in {city}: {results[city][parameter]['forecasted_values']}")
        print(f"Actual values: {results[city][parameter]['actual_values']}")
        print(f"Mean Squared Error: {results[city][parameter]['mse']}")
    else:
        print("Invalid city or parameter.")

if __name__ == "__main__":
    # Train and test on all datasets and parameters
    results = train_and_test_all_variables()

    # Custom testing
    custom_test(results)