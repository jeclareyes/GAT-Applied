#!/usr/bin/env python3
"""
gat_model.py

Definición del modelo TrafficGAT basado en Graph Attention Networks (GAT) adaptado para predecir flujos de tráfico.
Este modelo toma como entrada características de los nodos, aplica dos capas GAT y utiliza un MLP para predecir el flujo
de cada enlace del grafo.

La estructura es la siguiente:
    1. Una primera capa GAT con múltiples cabezas para aprender representaciones intermedias de los nodos.
    2. Una segunda capa GAT que reduce la dimensionalidad a una única representación por nodo.
    3. Un MLP que, a partir de la concatenación de las representaciones de los nodos fuente y destino de un enlace,
       predice el flujo asociado a dicho enlace.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
from torch_geometric.nn import GATConv


class TrafficGAT(nn.Module):
    """
    Modelo GAT para predicción de flujos de tráfico.

    Args:
        in_channels (int): Dimensión de las características de entrada para cada nodo.
        hidden_dim (int): Dimensión intermedia (y de salida final de la segunda capa GAT).
        heads (int): Número de cabezas de atención en la primera capa GAT.
    """

    def __init__(self, in_channels: int, hidden_dim: int = 16, heads: int = 2) -> None:
        super(TrafficGAT, self).__init__()

        # Primera capa GAT: con varias cabezas para capturar múltiples relaciones de atención.
        self.conv1 = GATConv(
            in_channels=in_channels,
            out_channels=hidden_dim,
            heads=heads,
            dropout=0.1,
            add_self_loops=False
        )

        # Segunda capa GAT: reduce la concatenación de las cabezas a una sola representación.
        self.conv2 = GATConv(
            in_channels=hidden_dim * heads,
            out_channels=hidden_dim,
            heads=1,
            concat=False,
            dropout=0.1,
            add_self_loops=False
        )

        # MLP para predecir flujos: utiliza la concatenación de las representaciones de nodo para el enlace.
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # Combina las representaciones de los nodos de origen y destino.
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)  # Salida: flujo predicho para el enlace.
        )

    def forward(self, data: torch_geometric.data.Data) -> torch.Tensor:
        """
        Realiza el forward pass del modelo.

        Args:
            data (torch_geometric.data.Data): Objeto que contiene al menos:
                - x: Tensor de características de nodo [num_nodes, in_channels].
                - edge_index: Tensor de topología del grafo [2, num_edges].

        Returns:
            torch.Tensor: Tensor unidimensional con el flujo predicho para cada enlace [num_edges].
        """
        x, edge_index = data.x, data.edge_index

        # Aplicar dropout antes de la primera capa para regularización
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.elu(x)  # Activación ELU para la salida de la primera capa

        # Segunda capa GAT con dropout para robustez
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)

        # Obtener las representaciones de los nodos para cada extremo del enlace
        edge_src = x[edge_index[0]]  # Nodo fuente
        edge_dst = x[edge_index[1]]  # Nodo destino

        # Concatenar las representaciones de los nodos para formar la entrada del MLP
        edge_features = torch.cat([edge_src, edge_dst], dim=-1)

        # Predecir flujo a partir de las características combinadas
        edge_flow = self.edge_mlp(edge_features).squeeze(-1)

        # Asegurar que los flujos sean no negativos (opcional, mediante ReLU)
        edge_flow = F.relu(edge_flow)

        return edge_flow
