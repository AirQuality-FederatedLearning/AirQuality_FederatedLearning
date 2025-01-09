import tensorflow as tf
import numpy as np

# Load the saved model with custom objects
try:
    model_path = "client_model.h5"  # Ensure this file exists in the current directory
    client_model = tf.keras.models.load_model(model_path, custom_objects={"mse": tf.keras.losses.MeanSquaredError()})
    print(f"Model loaded successfully from {model_path}.")
except OSError:
    print(f"Error: Could not load the model from {model_path}. Ensure the file exists.")
    exit(1)

# Accept user input for prediction
try:
    input_data = input("Enter comma-separated input values for prediction: ")
    input_data = np.array([float(x) for x in input_data.split(",")])  # Convert input to a NumPy array

    # Ensure the input shape matches the model's expected input shape
    # For a GRU model, we need to reshape to (batch_size, time_steps, features)
    input_data = np.expand_dims(input_data, axis=0)  # Add batch dimension
    input_data = np.expand_dims(input_data, axis=1)  # Add time_steps dimension

    # Perform the prediction
    prediction = client_model.predict(input_data)

    print(f"Prediction: {prediction.flatten()[0]}")
except Exception as e:
    print(f"Error during prediction: {e}")
