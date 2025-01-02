# server.py

import hydra
from omegaconf import DictConfig
import flwr as fl
import tensorflow as tf
import ray

# Optional: If you want to enable Ray to speed up aggregator
# ray.init(ignore_reinit_error=True)

def build_model(input_dim: int = 10):
    """Build a sample model (GRU/BiLSTM or MLP)."""
    model = tf.keras.Sequential()
    # Example with GRU + BiLSTM:
    model.add(tf.keras.layers.Reshape((1, input_dim), input_shape=(input_dim,)))
    model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)))
    model.add(tf.keras.layers.GRU(32))
    model.add(tf.keras.layers.Dense(1))
    model.compile(optimizer="adam", loss="mean_squared_error", metrics=["mae"])
    return model

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    # 1. Create and compile model (this will serve as the global model)
    global_model = build_model(input_dim=10)

    # 2. Define Flower strategy
    # You can override the default FedAvg if you want a Ray-based aggregator, weighted aggregator, etc.
    # For demonstration, we’ll just do standard FedAvg.
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=cfg.client.fraction_fit,
        fraction_eval=cfg.client.fraction_eval,
        min_fit_clients=cfg.client.min_fit_clients,
        min_eval_clients=cfg.client.min_eval_clients,
        min_available_clients=cfg.client.min_available_clients,
        # Optionally, provide the initial model weights:
        initial_parameters=fl.common.weights_to_parameters(global_model.get_weights()),
    )

    # 3. Start Flower server
    fl.server.start_server(
        server_address=cfg.server.server_address,
        strategy=strategy,
        config={"num_rounds": cfg.server.rounds},
    )


if __name__ == "__main__":
    main()
