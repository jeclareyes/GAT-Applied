#!/usr/bin/env python3
"""
HeGAT.py

Implementation of the heterogeneous HeGATPyG model for traffic networks.
Modificado para aceptar dropout como parámetro y para manejar G/A aprendibles para nodos AUX.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.data import Data
import logging

logger = logging.getLogger(__name__)

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
    def __init__(self, embed_dim: int, num_heads: int, ff_hidden_dim: int, dropout_rate: float = 0.1) -> None:
        super(VEncoderLayerPyG, self).__init__()
        # Asegurar que embed_dim sea divisible por num_heads o manejarlo
        if embed_dim % num_heads != 0:
            logger.warning(f"VEncoder: embed_dim ({embed_dim}) no es divisible por num_heads ({num_heads}). "
                           f"Ajustando out_channels de GATConv. Esto podría no ser lo ideal.")
            # Una posible forma de manejarlo es no concatenar si no es divisible,
            # o ajustar embed_dim/num_heads, pero GATConv espera que out_channels * heads sea la salida si concat=True.
            # Aquí se mantendrá la lógica original, pero se advierte.
            # Si concat=True, la salida es embed_dim. Si concat=False, la salida es embed_dim // num_heads.
            # GATConv out_channels es por cabeza.
            self.gat_out_channels = embed_dim // num_heads 
            gat_output_dim = self.gat_out_channels * num_heads # Esto debería ser embed_dim si es divisible
        else:
            self.gat_out_channels = embed_dim // num_heads
            gat_output_dim = embed_dim
            
        self.gat = GATConv(
            in_channels=embed_dim,
            out_channels=self.gat_out_channels, # out_channels por cabeza
            heads=num_heads,
            dropout=dropout_rate, # Dropout en la atención
            concat=True # Concatena las salidas de las cabezas
        )
        # El FF espera la dimensión de entrada igual a la salida de GAT (concatenada)
        self.ff = nn.Sequential(
            nn.Linear(gat_output_dim, ff_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate), # Dropout en el FF
            nn.Linear(ff_hidden_dim, embed_dim) # La salida del FF debe ser embed_dim para la conexión residual
        )
        self.norm1 = nn.LayerNorm(embed_dim) # Normaliza sobre embed_dim
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout_rate) # Dropout general después de GAT

    def forward(self, x: torch.Tensor, virtual_edge_index: torch.Tensor) -> torch.Tensor:
        residual = x
        h_gat = self.gat(x, virtual_edge_index)
        h_gat = self.dropout(h_gat) # Dropout después de la capa GAT
        
        # Asegurar que la dimensión de h_gat coincida con x para la conexión residual
        if h_gat.shape[-1] != residual.shape[-1]:
             # Esto puede ocurrir si embed_dim no era divisible por num_heads y concat=True
             # Se podría añadir una proyección lineal aquí si fuera necesario, o revisar la lógica de GATConv.
             # Por ahora, asumimos que las dimensiones coinciden o la lógica de GATConv (concat=True) lo maneja.
             logger.error(f"VEncoder: Mismatch de dimensiones para conexión residual. GAT output: {h_gat.shape}, Residual: {residual.shape}")
             # Podría ser necesario un ajuste aquí si embed_dim no es divisible por num_heads
             # Por ejemplo, si GATConv con concat=True siempre devuelve num_heads * out_channels (por cabeza)
             # y esto no es igual a embed_dim.

        h = self.norm1(residual + h_gat) # Conexión residual y normalización
        
        residual_ff = h
        h_ff = self.ff(h)
        # h_ff ya tiene dropout dentro del nn.Sequential
        h = self.norm2(residual_ff + h_ff) # Conexión residual y normalización
        return h


class REncoderLayerPyG(nn.Module):
    """
    Encoder layer for real links.
    """
    def __init__(self, embed_dim: int, num_heads: int, ff_hidden_dim: int, dropout_rate: float = 0.1) -> None:
        super(REncoderLayerPyG, self).__init__()
        if embed_dim % num_heads != 0:
            logger.warning(f"REncoder: embed_dim ({embed_dim}) no es divisible por num_heads ({num_heads}).")
            self.gat_out_channels = embed_dim // num_heads
            gat_output_dim = self.gat_out_channels * num_heads
        else:
            self.gat_out_channels = embed_dim // num_heads
            gat_output_dim = embed_dim

        self.gat = GATConv(
            in_channels=embed_dim,
            out_channels=self.gat_out_channels,
            heads=num_heads,
            dropout=dropout_rate,
            concat=True
        )
        self.ff = nn.Sequential(
            nn.Linear(gat_output_dim, ff_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(ff_hidden_dim, embed_dim)
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor, real_edge_index: torch.Tensor) -> torch.Tensor:
        residual = x
        h_gat = self.gat(x, real_edge_index)
        h_gat = self.dropout(h_gat)
        
        if h_gat.shape[-1] != residual.shape[-1]:
             logger.error(f"REncoder: Mismatch de dimensiones para conexión residual. GAT output: {h_gat.shape}, Residual: {residual.shape}")

        h = self.norm1(residual + h_gat)
        
        residual_ff = h
        h_ff = self.ff(h)
        h = self.norm2(residual_ff + h_ff)
        return h


class EdgePredictionLayerPyG(nn.Module):
    """
    Layer to predict the flow on each edge using the node representations.
    """
    def __init__(self, embed_dim: int, hidden_dim: int) -> None:
        super(EdgePredictionLayerPyG, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        src = x[edge_index[0]]
        dst = x[edge_index[1]]
        edge_features = torch.cat([src, dst], dim=-1)
        flow_pred = self.mlp(edge_features).squeeze(-1)
        return flow_pred


class HeGATPyG(nn.Module):
    """
    Heterogeneous Graph Attention Model (HeGATPyG) for traffic flow prediction.
    """
    def __init__(self,
                 node_feat_dim: int,    # Dimensión de data.x original
                 embed_dim: int,
                 num_v_layers: int,
                 num_r_layers: int,
                 num_heads: int,
                 ff_hidden_dim: int,
                 pred_hidden_dim: int,
                 dropout_rate: float = 0.6, # Tasa de dropout a aplicar
                 num_aux_nodes: int = 0, # Número de nodos AUX para G/A aprendibles
                 aux_learnable_ga_initial_scale: float = 1.0 # Escala para inicializar G/A de AUX
                 ) -> None:
        super(HeGATPyG, self).__init__()
        
        self.node_feat_dim = node_feat_dim
        self.embed_dim = embed_dim
        self.dropout_rate = dropout_rate # Guardar para usar en F.dropout si es necesario
        self.num_aux_nodes = num_aux_nodes

        # Capa de preprocesamiento para las características originales de los nodos
        # La dimensión de entrada es node_feat_dim (ej. 7 si [is_taz, is_aux, is_int, gen_taz, attr_taz, gen_aux, attr_aux])
        self.preprocess = PreprocessingLayer(node_feat_dim, embed_dim)

        # Parámetros aprendibles para generación y atracción de nodos AUX
        if self.num_aux_nodes > 0:
            # Inicializar con valores pequeños positivos, escalados
            self.aux_generation_params = nn.Parameter(torch.rand(self.num_aux_nodes, 1) * aux_learnable_ga_initial_scale)
            self.aux_attraction_params = nn.Parameter(torch.rand(self.num_aux_nodes, 1) * aux_learnable_ga_initial_scale)
            logger.info(f"HeGATPyG: Creados parámetros aprendibles para {self.num_aux_nodes} nodos AUX.")
        
        else:
            self.aux_generation_params = None
            self.aux_attraction_params = None
            logger.info("HeGATPyG: No se especificaron nodos AUX para G/A aprendibles.")

        self.v_encoders = nn.ModuleList([
            VEncoderLayerPyG(embed_dim, num_heads, ff_hidden_dim, dropout_rate)
            for _ in range(num_v_layers)
        ])
        self.r_encoders = nn.ModuleList([
            REncoderLayerPyG(embed_dim, num_heads, ff_hidden_dim, dropout_rate)
            for _ in range(num_r_layers)
        ])
        self.edge_predictor = EdgePredictionLayerPyG(embed_dim, pred_hidden_dim)

    def forward(self, data: Data) -> torch.Tensor:
        """
        Forward pass of the HeGATPyG model.
        """
        x_processed = data.x.clone() # Trabajar con una copia de x para modificarla

        # Actualizar las características de los nodos AUX con los parámetros aprendibles
        # Asumimos que data.aux_node_indices contiene los índices GLOBALES de los nodos AUX
        # y que data.x tiene placeholders en las columnas 5 y 6 para estos nodos.
        if self.num_aux_nodes > 0 and data.aux_node_indices.numel() > 0:
            if self.aux_generation_params is not None and self.aux_attraction_params is not None:
                # Asegurar que los parámetros aprendidos sean no negativos
                learned_gens = F.relu(self.aux_generation_params)
                learned_attrs = F.relu(self.aux_attraction_params)

                # Asignar a las columnas correspondientes de x_processed para los nodos AUX
                # Columna 5 para generación AUX, Columna 6 para atracción AUX
                # data.aux_node_indices son los índices globales de los nodos AUX
                # los parámetros aprendibles están ordenados según el orden de data.aux_node_indices
                x_processed[data.aux_node_indices, 5] = learned_gens.squeeze(-1)
                x_processed[data.aux_node_indices, 6] = learned_attrs.squeeze(-1)
            else:
                logger.warning("HeGATPyG: num_aux_nodes > 0 pero los parámetros aprendibles no están inicializados.")

        # Aplicar la capa de preprocesamiento a las características (posiblemente actualizadas)
        h_embed = self.preprocess(x_processed) # h_embed tiene dimensión embed_dim

        # Aplicar dropout general a las embeddings iniciales si se desea
        # h_embed = F.dropout(h_embed, p=self.dropout_rate, training=self.training)

        # Procesar con codificadores virtuales
        if hasattr(data, 'virtual_edge_index') and data.virtual_edge_index is not None and data.virtual_edge_index.numel() > 0:
            h_v = h_embed
            for layer in self.v_encoders:
                h_v = layer(h_v, data.virtual_edge_index)
        else:
            h_v = h_embed # Si no hay enlaces virtuales, pasar las embeddings directamente

        # Procesar con codificadores reales
        if hasattr(data, 'edge_index'):
            h_r = h_v # La entrada a REncoders es la salida de VEncoders (o h_embed si no hay VEncoders)
            for layer in self.r_encoders:
                h_r = layer(h_r, data.edge_index)
        else:
            raise AttributeError("El objeto Data debe contener 'edge_index' para procesar enlaces reales.")

        # La salida de REncoders (o VEncoders si no hay REncoders) es la representación final del nodo
        final_node_representation = h_r

        # Predecir flujos de arista
        edge_flow = self.edge_predictor(final_node_representation, data.edge_index)
        
        # Asegurar flujos no negativos
        edge_flow = F.relu(edge_flow)
        
        return edge_flow
