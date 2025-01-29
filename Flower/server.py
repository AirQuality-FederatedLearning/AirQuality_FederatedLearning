import os
import flwr as fl
import tensorflow as tf
from flwr.server.strategy import FedAvg

def build_hybrid_mlp_lstm(input_dim, learning_rate):
    inputs = tf.keras.Input(shape=(1, input_dim))
    mlp = tf.keras.layers.Dense(10, activation="tanh")(inputs)
    mlp = tf.keras.layers.Dense(5, activation="tanh")(mlp)
    mlp = tf.keras.layers.Dense(1)(mlp)
    mlp_output = tf.keras.layers.Reshape((1, 1))(mlp)

    lstm = tf.keras.layers.LSTM(128, return_sequences=False)(inputs)
    lstm = tf.keras.layers.Dense(100, activation="tanh")(lstm)
    lstm_output = tf.keras.layers.Dense(1)(lstm)
    lstm_output = tf.keras.layers.Reshape((1, 1))(lstm_output)

    combined = tf.keras.layers.Concatenate(axis=-1)([mlp_output, lstm_output])
    combined_output = tf.keras.layers.Dense(7)(combined)
    model = tf.keras.Model(inputs=inputs, outputs=combined_output)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mse",
        metrics=["mae"],
    )
    return model

class SaveModelFedAvg(FedAvg):
    def __init__(self, model_save_path: str):
        super().__init__()
        self.model_save_path = model_save_path

    def on_fit_end(self, server_round, parameters, config):
        model = self.create_model_from_weights(parameters)
        model_path = os.path.join(self.model_save_path, f"global_model_round_{server_round}.h5")
        model.save(model_path)
        print(f"Global model saved to {model_path}")
        return super().on_fit_end(server_round, parameters, config)

    @staticmethod
    def create_model_from_weights(weights):
        model = build_hybrid_mlp_lstm(input_dim=7, learning_rate=0.001)
        model.set_weights(weights)
        return model

def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=10, help="Number of federated rounds")
    parser.add_argument("--save_dir", type=str, default="./saved_models", help="Directory to save global models")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    strategy = SaveModelFedAvg(model_save_path=args.save_dir)

    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=args.rounds),
        strategy=strategy,
    )

if __name__ == "__main__":
    main()
