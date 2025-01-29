import os
import flwr as fl
import tensorflow as tf
from flwr.server.strategy import FedAvg

###############################################################################
# 1) Model Builder (Server side) - Must have the same architecture as clients #
###############################################################################
def build_hybrid_mlp_lstm(input_dim, learning_rate):
    """
    Build a hybrid MLP+LSTM model that outputs 7 units 
    (one per pollutant).
    """
    inputs = tf.keras.Input(shape=(1, input_dim))
    
    # MLP branch
    mlp = tf.keras.layers.Dense(10, activation="tanh")(inputs)
    mlp = tf.keras.layers.Dense(5, activation="tanh")(mlp)
    mlp = tf.keras.layers.Dense(1)(mlp)
    mlp_output = tf.keras.layers.Reshape((1, 1))(mlp)

    # LSTM branch
    lstm = tf.keras.layers.LSTM(128, return_sequences=False)(inputs)
    lstm = tf.keras.layers.Dense(100, activation="tanh")(lstm)
    lstm_output = tf.keras.layers.Dense(1)(lstm)
    lstm_output = tf.keras.layers.Reshape((1, 1))(lstm_output)

    # Combine branches
    combined = tf.keras.layers.Concatenate(axis=-1)([mlp_output, lstm_output])
    
    # Final Dense to produce 7 outputs (for 7 pollutants)
    combined_output = tf.keras.layers.Dense(7)(combined)

    model = tf.keras.Model(inputs=inputs, outputs=combined_output)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mse",
        metrics=["mae"]
    )
    return model


###############################################################################
# 2) Custom FedAvg Strategy to Save Global Model at Each Round               #
###############################################################################
class SaveModelFedAvg(FedAvg):
    def __init__(self, model_save_path: str):
        super().__init__()
        self.model_save_path = model_save_path

    def on_fit_end(self, server_round, parameters, config):
        """
        Called by the Flower server at the end of each training round.
        We convert the parameters to a Keras model and save it.
        """
        # Rebuild the model with the same architecture
        model = self.create_model_from_weights(parameters)
        
        # Build the path and save the model
        model_path = os.path.join(self.model_save_path, f"global_model_round_{server_round}.h5")
        model.save(model_path)
        print(f"Global model saved to {model_path}")

        # Call the super method to finish
        return super().on_fit_end(server_round, parameters, config)

    @staticmethod
    def create_model_from_weights(weights):
        # Must match the architecture exactly with 7 outputs
        model = build_hybrid_mlp_lstm(input_dim=7, learning_rate=0.001)
        model.set_weights(weights)
        return model


###############################################################################
# 3) Main: Start the FL Server                                               #
###############################################################################
def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rounds",
        type=int,
        default=5,
        help="Number of federated rounds (default: 10)"
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./saved_models",
        help="Directory to save global models"
    )
    args = parser.parse_args()

    # Create the model save directory if it doesn't exist
    os.makedirs(args.save_dir, exist_ok=True)

    # Create our custom strategy
    strategy = SaveModelFedAvg(model_save_path=args.save_dir)

    # Start Flower server
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=args.rounds),
        strategy=strategy
    )


if __name__ == "__main__":
    main()
