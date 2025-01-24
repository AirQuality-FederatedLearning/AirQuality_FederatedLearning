import numpy as np
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError

def main():
    # Load the model with custom objects
    custom_objects = {'mse': MeanSquaredError()}
    client_model = load_model("Coimbatore_PM2.5_timeseries_model.h5", custom_objects=custom_objects)
    print("Model loaded successfully")
    
    # Get the input data from user
    input_data = input("Enter the input data: ")
    
    # Convert the input string to a list of floats
    input_data = input_data.split(",")
    input_data = [float(x) for x in input_data]
    
    # Reshape the input data to match the expected input shape
    input_data = np.array(input_data).reshape(1, len(input_data), 1)
    
    # Perform the prediction
    prediction = client_model.predict(input_data)
    print(f"Prediction: {prediction}")

if __name__ == "__main__":
    main()