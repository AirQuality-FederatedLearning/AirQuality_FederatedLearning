import os
import tensorflow as tf
import flwr as fl

# Base strategies from Flower
from flwr.server.strategy import FedAvg
from flwr.server.strategy import FedProx
from flwr.server.strategy import FedAdam  # or FedYogi, FedAdagrad, etc.

# --------------------------------------------------------------------
# 1) Common SaveModelMixin to save the global model each round
# --------------------------------------------------------------------
class SaveModelMixin:
    """A mixin providing shared functionality to save the global model after each round."""

    def on_fit_end(self, server_round, parameters, config):
        """Called by Flower after aggregation each round."""
        print(f"[Server] Completed round {server_round}. Saving global model...")

        # Build local model with the aggregated weights
        model = self.create_model_from_weights(parameters)
        save_path = os.path.join(self.model_save_path, f"global_model_round_{server_round}.h5")
        model.save(save_path)

        print(f"[Server] Global model saved to {save_path}")
        # Proceed with the base strategy's on_fit_end
        return super().on_fit_end(server_round, parameters, config)

    def create_model_from_weights(self, weights):
        """Create a fresh model instance and set the aggregated weights."""
        model = self.build_model()
        model.set_weights(weights)
        return model

    @staticmethod
    def build_model():
        """
        Build the same multi-output model used by the clients.
        Adjust the input shape (1, 25) and output size (15) as appropriate.
        """
        num_outputs = 15  # e.g., 15 pollutant targets
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(1, 25)),  # Example input dimension
            tf.keras.layers.GRU(16),
            tf.keras.layers.Dense(num_outputs),
        ])
        model.compile(optimizer="adam", loss="mse")
        return model


# --------------------------------------------------------------------
# 2) SaveModelFedAvg
# --------------------------------------------------------------------
class SaveModelFedAvg(SaveModelMixin, FedAvg):
    """FedAvg strategy that saves the global model after each round."""

    def __init__(self, model_save_path):
        # Initialize FedAvg with default or custom arguments
        super().__init__()
        self.model_save_path = model_save_path


# --------------------------------------------------------------------
# 3) SaveModelFedProx
# --------------------------------------------------------------------
class SaveModelFedProx(SaveModelMixin, FedProx):
    """FedProx strategy that saves the global model after each round."""

    def __init__(self, model_save_path, proximal_mu=0.1):
        super().__init__(proximal_mu=proximal_mu)  # FedProx constructor
        self.model_save_path = model_save_path


# --------------------------------------------------------------------
# 4) SaveModelFedAdam
# --------------------------------------------------------------------
class SaveModelFedAdam(SaveModelMixin, FedAdam):
    """FedAdam strategy that saves the global model after each round."""

    def __init__(self, model_save_path, eta=1e-2, beta_1=0.9, beta_2=0.999, tau=1e-9):
        # FedAdam constructor parameters:
        # - eta: server-side learning rate
        # - beta_1, beta_2: Adam momentum/hyperparameters
        # - tau: server-side momentum dampening
        super().__init__(eta=eta, beta_1=beta_1, beta_2=beta_2, tau=tau)
        self.model_save_path = model_save_path


# --------------------------------------------------------------------
# 5) Main server code
# --------------------------------------------------------------------
def main():
    # Directory to save the global model
    model_save_dir = "./saved_models"
    os.makedirs(model_save_dir, exist_ok=True)

    # ----------------------------------------------------------------
    # Pick ONE of the following strategies by uncommenting the line:
    # ----------------------------------------------------------------

    # 5a) FedAvg with saving
    strategy = SaveModelFedAvg(model_save_path=model_save_dir)

    # 5b) FedProx with mu=0.1
    # strategy = SaveModelFedProx(model_save_path=model_save_dir, proximal_mu=0.1)

    # 5c) FedAdam with default parameters
    # strategy = SaveModelFedAdam(model_save_path=model_save_dir, eta=0.001)

    # ----------------------------------------------------------------
    # Start the FL server, specifying the number of rounds (or a large number for indefinite).
    # ----------------------------------------------------------------
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        strategy=strategy,
        config=fl.server.ServerConfig(num_rounds=30),  # or a very large integer for near-indefinite
    )


if __name__ == "__main__":
    main()
