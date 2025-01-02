# server.py

import flwr as fl
from flwr.server.strategy import FedAvg , FedOpt ,FedProx,FedAvg

def main():
    # Create strategy (FedAvg, or anything else)
    strategy =FedProx(proximal_mu=0.1)    
    # strategy = FedAvg()

    # Start the Flower server
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        strategy=strategy,
        config=fl.server.ServerConfig(num_rounds=25),  # e.g., 3 rounds
    )

if __name__ == "__main__":
    main()
