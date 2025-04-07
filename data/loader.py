# data/loader.py
import pickle
import torch
from torch_geometric.data import Data
import os
local_path = os.path.join(os.path.dirname(os.path.abspath(__file__)))

def load_traffic_data(name_file):
    """
    Load traffic data from a pickle file and prepare a PyTorch Geometric Data object.
    
    Args:
        filepath (str): Path to the pickle file containing traffic data.
    
    Returns:
        Data: A PyTorch Geometric Data object with traffic network information.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    export_dir = os.path.join(local_path, name_file)
    # Load data from pickle file
    with open(export_dir, "rb") as f:
        traffic = pickle.load(f)

    # Extract node features (generated/attracted demand)
    x = torch.tensor(traffic["nodes_features"], dtype=torch.float)

    # Construct edge_index (shape [2, num_edges])
    links = traffic["links_topology"]  # numpy array (num_edges, 2)
    edge_index = torch.tensor(links.T, dtype=torch.long)

    # Labels (targets) = estimated flow for each link
    df = traffic["results_df"]
    y = torch.tensor(df["estimated_flow"].values, dtype=torch.float)

    # Edge features = observed flow (0 where no measurement)
    obs = df["observed_flow"].fillna(0).values
    edge_attr = torch.tensor(obs, dtype=torch.float).unsqueeze(1)

    # Training mask: True only where flow is observed
    train_mask = torch.tensor(~df["observed_flow"].isna().values)

    # Construct PyG Data object
    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=y,
        train_mask=train_mask
    ).to(device)

    return data

# If this script is run directly, load and print basic info about the data
if __name__ == "__main__":
    data = load_traffic_data()
    print("Data Information:")
    print(f"Number of nodes: {data.x.size(0)}")
    print(f"Number of edges: {data.edge_index.size(1)}")
    print(f"Node feature dimension: {data.x.size(1)}")
    print(f"Training mask sum: {data.train_mask.sum()}")
