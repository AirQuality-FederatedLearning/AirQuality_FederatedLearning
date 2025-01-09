import flwr as fl
from flwr.server.strategy import FedProx
import tensorflow as tf
import os

class SaveModelStrategy(FedProx):
    def __init__(self, proximal_mu, model_save_path):
        super().__init__(proximal_mu=proximal_mu)
        self.model_save_path = model_save_path
        self.global_model = None

    def on_fit_end(self, server_round, weights, config):
        # Save the global model after every round
        print(f"Round {server_round}: Saving model to {self.model_save_path}")
        save_path = os.path.join(self.model_save_path, f"global_model_round_{server_round}.h5")
        self.global_model = self.create_model_from_weights(weights)
        self.global_model.save(save_path)
        print(f"Global model saved to {save_path} at round {server_round}.")
        return super().on_fit_end(server_round, weights, config)

    def create_model_from_weights(self, weights):
        # Create a new model and set the weights
        model = self.build_model()
        model.set_weights(weights)
        return model

    @staticmethod
    def build_model():
        # Build the same model architecture used by the clients
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(1, 6)),  # Example input shape
            tf.keras.layers.GRU(16),
            tf.keras.layers.Dense(1),
        ])
        model.compile(optimizer="adam", loss="mse")
        return model

def main():
    # Directory to save the global model
    model_save_dir = "C:/Users/DELL/Documents/Amrita/4th year/AirQuality_FederatedLearning/Flower/flower_demo/models"
    os.makedirs(model_save_dir, exist_ok=True)  # Ensure the directory exists

    # Create strategy with model saving capabilities
    strategy = SaveModelStrategy(proximal_mu=0.1, model_save_path=model_save_dir)

    # Start the Flower server
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        strategy=strategy,
        config=fl.server.ServerConfig(num_rounds=10),  # e.g., 10 rounds
    )

if __name__ == "__main__":
    main()
