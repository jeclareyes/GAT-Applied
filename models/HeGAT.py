#!/usr/bin/env python3
"""
HeGAT.py

Implementation of the heterogeneous HetGATPyG model for traffic networks, which uses:
  - PreprocessingLayer to transform input features into embeddings.
  - VEncoderLayerPyG to process virtual connections.
  - REncoderLayerPyG to process real connections.
  - EdgePredictionLayerPyG to predict flows from node representations.

This modular design captures both explicit (real links) and implicit (virtual links) relationships,
combining them for link flow prediction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.data import Data


class PreprocessingLayer(nn.Module):
    """
    Preprocessing layer that projects node input features
    into a fixed-dimensional embedding space.
    """

    def __init__(self, in_dim: int, embed_dim: int) -> None:
        super(PreprocessingLayer, self).__init__()
        self.linear = nn.Linear(in_dim, embed_dim)
        self.activation = nn.ReLU()
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.linear(x)
        h = self.activation(h)
        h = self.norm(h)
        return h


class VEncoderLayerPyG(nn.Module):
    """
    Encoder layer for virtual links.
    Applies a GAT layer followed by a feed-forward block with residual connections to update embeddings.
    """

    def __init__(self, embed_dim: int, num_heads: int, ff_hidden_dim: int, dropout: float = 0.1) -> None:
        super(VEncoderLayerPyG, self).__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be divisible by the number of heads"
        self.gat = GATConv(
            in_channels=embed_dim,
            out_channels=embed_dim // num_heads,
            heads=num_heads,
            dropout=dropout,
            concat=True
        )
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_hidden_dim, embed_dim)
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, virtual_edge_index: torch.Tensor) -> torch.Tensor:
        residual = x
        h = self.gat(x, virtual_edge_index)
        h = self.dropout(h)
        h = self.norm1(residual + h)
        residual = h
        h = self.ff(h)
        h = self.norm2(residual + h)
        return h


class REncoderLayerPyG(nn.Module):
    """
    Encoder layer for real links.
    Similar to VEncoderLayerPyG, it processes the observed topology to capture direct node relationships.
    """

    def __init__(self, embed_dim: int, num_heads: int, ff_hidden_dim: int, dropout: float = 0.1) -> None:
        super(REncoderLayerPyG, self).__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be divisible by the number of heads"
        self.gat = GATConv(
            in_channels=embed_dim,
            out_channels=embed_dim // num_heads,
            heads=num_heads,
            dropout=dropout,
            concat=True
        )
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_hidden_dim, embed_dim)
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, real_edge_index: torch.Tensor) -> torch.Tensor:
        residual = x
        h = self.gat(x, real_edge_index)
        h = self.dropout(h)
        h = self.norm1(residual + h)
        residual = h
        h = self.ff(h)
        h = self.norm2(residual + h)
        return h


class EdgePredictionLayerPyG(nn.Module):
    """
    Layer to predict the flow on each edge using the node representations.
    Combines the source and destination node embeddings and applies an MLP.
    """

    def __init__(self, embed_dim: int, hidden_dim: int) -> None:
        super(EdgePredictionLayerPyG, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        # Extract node features for the endpoints of each edge.
        src = x[edge_index[0]]
        dst = x[edge_index[1]]
        edge_features = torch.cat([src, dst], dim=-1)
        flow_pred = self.mlp(edge_features).squeeze(-1)
        return flow_pred


class HetGATPyG(nn.Module):
    """
    Heterogeneous Graph Attention Model (HetGATPyG) for traffic flow prediction.

    It combines:
      - A preprocessing layer to generate embeddings from input features.
      - Virtual link encoder blocks (VEncoder) to capture implicit relationships.
      - Real link encoder blocks (REncoder) to capture direct relationships.
      - A prediction layer that uses the combined representations to estimate edge flows.

    The input Data object is expected to include:
      - x: Node features.
      - edge_index: Real edge indices.
      - virtual_edge_index (optional): Virtual edge indices.
    """

    def __init__(self,
                 node_feat_dim: int,
                 embed_dim: int,
                 num_v_layers: int,
                 num_r_layers: int,
                 num_heads: int,
                 ff_hidden_dim: int,
                 pred_hidden_dim: int,
                 dropout: float = 0.6) -> None:
        super(HetGATPyG, self).__init__()
        self.preprocess = PreprocessingLayer(node_feat_dim, embed_dim)
        self.v_encoders = nn.ModuleList([
            VEncoderLayerPyG(embed_dim, num_heads, ff_hidden_dim, dropout)
            for _ in range(num_v_layers)
        ])
        self.r_encoders = nn.ModuleList([
            REncoderLayerPyG(embed_dim, num_heads, ff_hidden_dim, dropout)
            for _ in range(num_r_layers)
        ])
        self.edge_predictor = EdgePredictionLayerPyG(embed_dim, pred_hidden_dim)

    def forward(self, data: Data) -> torch.Tensor:
        """
        Forward pass of the HetGATPyG model.

        Args:
            data (torch_geometric.data.Data): Object containing:
                - x: Node feature tensor.
                - edge_index: Tensor [2, num_edges] with the real topology.
                - virtual_edge_index (optional): Tensor [2, num_virtual_edges] with the virtual topology.

        Returns:
            torch.Tensor: Flow predictions for each real edge.
        """
        x = self.preprocess(data.x)

        if hasattr(data, 'virtual_edge_index') and data.virtual_edge_index is not None and data.virtual_edge_index.numel() > 0:
            for layer in self.v_encoders:
                x = layer(x, data.virtual_edge_index)

        if hasattr(data, 'edge_index'):
            for layer in self.r_encoders:
                x = layer(x, data.edge_index)
        else:
            raise AttributeError("The Data object must contain 'edge_index' to process real links.")

        edge_flow = self.edge_predictor(x, data.edge_index)
        edge_flow = F.relu(edge_flow)
        return edge_flow
