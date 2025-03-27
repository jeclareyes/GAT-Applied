# models/gat.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv


class TrafficGAT(nn.Module):
    """
    Graph Attention Network (GAT) for Traffic Sensor Data.
    
    This model uses graph attention layers to learn node and edge representations
    for traffic flow prediction.
    
    Args:
        in_dim (int): Input feature dimension for nodes, 2 (generation, attraction)
        edge_dim (int): Dimension of edge attributes, 1
        hidden_dim (int, optional): Hidden layer dimension. Defaults to 64.
        heads (int, optional): Number of attention heads. Defaults to 4.
    """

    def __init__(self, in_dim, edge_dim, hidden_dim=64, heads=4):
        super().__init__()
        # First GAT convolution layer with multiple attention heads
        self.conv1 = GATConv(
            in_dim,
            hidden_dim,
            heads=heads,
            edge_dim=edge_dim,
            add_self_loops=False
        )

        # Second GAT convolution layer with reduced heads
        self.conv2 = GATConv(
            hidden_dim * heads,
            hidden_dim,
            heads=1,
            edge_dim=edge_dim,
            add_self_loops=False
        )

        # Final regression layer to predict traffic flows
        #  Regressor 1: Linear
        #self.regressor = nn.Linear(hidden_dim, 1)

        #  Regressor 2: MLP Regressor
        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        # Store attention weights for analysis
        self.attention_weights = []

    def forward(self, data):
        """
        Forward pass of the Graph Attention Network.
        
        Args:
            data (Data): PyTorch Geometric Data object
        
        Returns:
            torch.Tensor: Predicted traffic flows for edges
        """
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        # First layer with attention
        x, attn1 = self.conv1(
            x,
            edge_index,
            edge_attr=edge_attr,
            return_attention_weights=True
        )
        self.attention_weights.append(attn1[1].detach().cpu())

        # Second layer with attention
        x, attn2 = self.conv2(
            x,
            edge_index,
            edge_attr=edge_attr,
            return_attention_weights=True
        )
        self.attention_weights.append(attn2[1].detach().cpu())

        # Compute edge embeddings by summing source and destination node embeddings
        src, dst = edge_index
        edge_embeddings = x[src] + x[dst]

        # Predict traffic flows for edges
        return self.regressor(edge_embeddings).squeeze()

    def reset_attention_weights(self):
        """
        Reset the stored attention weights.
        Useful between different forward passes or model evaluations.
        """
        self.attention_weights = []

    def conservation_loss(self, pred_flows, data):
        """
        Calculate flow conservation loss to ensure network flow constraints.

        This function enforces that:
        1. The sum of outflows from each node equals its origin demand
        2. The sum of inflows to each node equals its destination demand

        Args:
            pred_flows (torch.Tensor): Predicted flow values for each edge
            data: PyTorch Geometric data object with node demands

        Returns:
            torch.Tensor: Conservation loss value (scalar)
        """
        # Get source and destination nodes for each edge
        src, dst = data.edge_index

        # Calculate total outflow from each node
        outflow = torch.zeros(data.num_nodes, device=data.x.device)
        # scatter_add_ accumulates flow values by source node
        outflow.scatter_add_(0, src, pred_flows)  # Sum outgoing flows for each node

        # Calculate total inflow to each node
        inflow = torch.zeros(data.num_nodes, device=data.x.device)
        # scatter_add_ accumulates flow values by destination node
        inflow.scatter_add_(0, dst, pred_flows)  # Sum incoming flows for each node

        # Get origin/destination demands from OD matrix
        # data.x[:,0] = origin demand, data.x[:,1] = destination demand

        # Calculate mean squared error between:
        # 1. Predicted outflows and origin demands
        # 2. Predicted inflows and destination demands
        loss = F.mse_loss(outflow, data.x[:, 0]) + F.mse_loss(inflow, data.x[:, 1])

        return loss
