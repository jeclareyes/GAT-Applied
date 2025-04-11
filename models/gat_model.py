# models/gat_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

class TrafficGAT(nn.Module):
    """
    Graph Attention Network (GAT) adapted for ZAT/Intersection Traffic Model.

    Predicts edge flows based on node embeddings learned via graph attention.
    Assumes node features represent [is_zat, is_int, net_demand].
    Uses a custom loss function (external) incorporating observed flows,
    conservation, and ZAT demand constraints.

    Args:
        in_channels (int): Input feature dimension for nodes (should be 3).
        hidden_dim (int, optional): Hidden layer dimension. Defaults to 16 or 64.
        heads (int, optional): Number of attention heads in the first GAT layer. Defaults to 2 or 4.
    """

    def __init__(self, in_channels=3, hidden_dim=16, heads=2): # Default hidden_dim and heads adjusted
        super().__init__()
        # First GAT convolution layer with multiple attention heads
        # Removed edge_dim, add_self_loops=False remains reasonable
        self.conv1 = GATConv(
            in_channels,
            hidden_dim,
            heads=heads,
            dropout=0.6, # Added dropout like in the example
            add_self_loops=False
        )

        # Second GAT convolution layer, outputting final node embeddings
        # Input dimension is hidden_dim * heads
        # Output dimension is hidden_dim (can be adjusted)
        # Heads set to 1 and concat=False for final embedding layer often
        self.conv2 = GATConv(
            hidden_dim * heads,
            hidden_dim, # Output dimension for node embeddings
            heads=1,
            concat=False, # Usually False for the last layer before prediction
            dropout=0.6, # Added dropout
            add_self_loops=False
        )

        # MLP for predicting edge flow from concatenated node embeddings
        # Input size is 2 * hidden_dim (from concatenated src and dst node embeddings)
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), # Hidden layer in MLP
            nn.ReLU(),
            nn.Linear(hidden_dim, 1) # Output layer: predicts 1 value (flow)
        )

        # Removed attention_weights storage and related methods for simplicity
        # Removed the old conservation_loss method

    def forward(self, data):
        """
        Forward pass of the Graph Attention Network.

        Args:
            data (Data): PyTorch Geometric Data object containing x and edge_index.

        Returns:
            torch.Tensor: Predicted non-negative traffic flows for edges.
        """
        # Extract node features and edge topology
        x, edge_index = data.x, data.edge_index

        # Apply first GAT layer
        # Note: edge_attr=None as we don't use edge features in GATConv here
        x = F.dropout(x, p=0.6, training=self.training) # Dropout before conv1
        x = self.conv1(x, edge_index, edge_attr=None)
        x = F.elu(x) # ELU activation after conv1

        # Apply second GAT layer
        x = F.dropout(x, p=0.6, training=self.training) # Dropout before conv2
        x = self.conv2(x, edge_index, edge_attr=None)
        # No activation needed after conv2 if followed by MLP? Or maybe ReLU/ELU? Let's keep it simple.
        # x now contains the final node embeddings (shape: [num_nodes, hidden_dim])

        # Predict edge flows using the MLP
        edge_src = x[edge_index[0]] # Get embeddings for source nodes of edges
        edge_dst = x[edge_index[1]] # Get embeddings for destination nodes of edges

        # Concatenate source and destination node embeddings
        edge_features_concat = torch.cat([edge_src, edge_dst], dim=-1) # Shape: [num_edges, hidden_dim * 2]

        # Pass concatenated features through the edge MLP
        edge_flows = self.edge_mlp(edge_features_concat).squeeze(-1) # Squeeze the last dim (size 1)

        # Ensure flows are non-negative
        edge_flows = F.relu(edge_flows)

        return edge_flows