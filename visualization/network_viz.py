#!/usr/bin/env python3
"""
network_viz.py

Visualizador de redes basado en PyVis para visualizar la red de tráfico (nodos y enlaces)
a partir de un objeto Data de PyTorch Geometric y los flujos predichos por el modelo.
Se integra el método save_visualization para generar y guardar la visualización en un archivo HTML.
"""

import os
import logging
import torch
from pyvis.network import Network

class NetworkVisualizer_Pyvis:
    def __init__(self, data, predicted_flows, config=None):
        """
        Inicializa el visualizador usando el objeto Data de PyG y las predicciones del modelo.

        Args:
            data (torch_geometric.data.Data): Objeto Data con atributos:
                - node_types: lista con el tipo de cada nodo.
                - node_coordinates: lista/tensor con coordenadas para cada nodo.
                - node_id_map_rev: diccionario que mapea índices a IDs de nodos.
                - link_types: lista con el tipo ('road', 'logical', etc.) para cada enlace.
                - observed_flow_indices y observed_flow_values (opcional): información de flujos observados.
            predicted_flows (torch.Tensor): Tensor con los flujos predichos para cada enlace (forma [num_edges]).
            config (dict, opcional): Configuración para sobreescribir la configuración por defecto.
        """
        # Verificar atributos esenciales
        for attr in ['node_coordinates', 'node_id_map_rev', 'link_types']:
            if not hasattr(data, attr):
                raise ValueError(f"El objeto 'data' debe contener el atributo '{attr}'. Verifica 'loader.py'.")

        self.data = data
        self.predicted_flows = predicted_flows.detach().cpu().view(-1)  # Asegurar que sea 1D y esté en CPU

        # Crear diccionario de flujos observados para acceso rápido (si se dispone de ellos)
        if hasattr(data, 'observed_flow_indices') and hasattr(data, 'observed_flow_values'):
            self.observed_flows_dict = {
                idx.item(): val.item()
                for idx, val in zip(data.observed_flow_indices.cpu(), data.observed_flow_values.cpu())
            }
        else:
            self.observed_flows_dict = {}

        # Configuración por defecto del visualizador
        default_config = {
            "height": "750px",
            "width": "100%",
            "bgcolor": "white",
            "font_color": "black",
            "export_filepath": "visualization/exported_viz_data/network_visualization.html",
            "node_styles": {
                'zat': {'color': "#FFD700", 'shape': 'diamond', 'size': 20, 'border_color': "black", 'border_width': 2},
                'intersection': {'color': "#87CEEB", 'shape': 'dot', 'size': 12, 'border_color': "darkgrey", 'border_width': 1},
                'default': {'color': "#D3D3D3", 'shape': 'ellipse', 'size': 10}
            },
            "link_styles": {
                'road': {'color_observed_match': "#2E8B57", 'color_observed_mismatch': "#DC143C", 'color_unobserved': "#A9A9A9",
                         'width_factor': 0.8, 'dashes': False},
                'logical': {'color_observed_match': "#ADD8E6", 'color_observed_mismatch': "#FFA07A", 'color_unobserved': "#E0E0E0",
                            'width_factor': 0.5, 'dashes': [5, 5]},
                'default': {'color_observed_match': "#CCCCCC", 'color_observed_mismatch': "#CCCCCC", 'color_unobserved': "#CCCCCC",
                            'width_factor': 0.5, 'dashes': [2, 2]}
            },
            "link_label_format": ".1f",
            "link_observed_tolerance": 0.1,
            "link_width_base": 1.0,
            "link_width_scaling_method": "log",
            "link_max_width": 8.0,
            "physics": {
                "solver": "forceAtlas2Based",
                "options": {
                    "gravitationalConstant": -50,
                    "centralGravity": 0.01,
                    "springLength": 150,
                    "springConstant": 0.08,
                    "damping": 0.4,
                    "avoidOverlap": 0.5
                },
                "stabilization": {"enabled": True, "iterations": 1000}
            }
        }
        # Fusionar la configuración personalizada (si se proporciona) con la por defecto
        if config:
            default_config.update(config)
        self.config = default_config

    def _get_node_style(self, node_type):
        """Obtiene el estilo del nodo basado en su tipo."""
        return self.config["node_styles"].get(node_type, self.config["node_styles"]['default'])

    def _get_link_style(self, link_type):
        """Obtiene el estilo del enlace basado en su tipo."""
        return self.config["link_styles"].get(link_type, self.config["link_styles"]['default'])

    def _calculate_edge_width(self, flow_value, style):
        """
        Calcula el ancho del enlace basándose en el flujo, usando el método definido en la configuración.
        """
        base = self.config.get("link_width_base", 1.0)
        factor = style.get("width_factor", 0.5)
        max_width = self.config.get("link_max_width", 8.0)
        method = self.config.get("link_width_scaling_method", "log")
        if flow_value <= 0:
            return base
        if method == "log":
            import math
            width = base + factor * math.log1p(flow_value)
        else:  # Método lineal
            width = base + factor * flow_value
        return min(width, max_width)

    def _create_network(self):
        """
        Construye el objeto Network de PyVis basado en los datos y predicciones.
        Agrega los nodos y enlaces utilizando la configuración establecida.
        """
        net = Network(height=self.config["height"],
                      width=self.config["width"],
                      bgcolor=self.config["bgcolor"],
                      font_color=self.config["font_color"],
                      directed=True)
        # Establecer opciones de física
        net.set_options(f"""
            var options = {{
              "physics": {{
                "forceAtlas2Based": {self.config["physics"]["options"]},
                "stabilization": {self.config["physics"]["stabilization"]}
              }}
            }}
        """)

        # Agregar nodos: se recorren los nodos (usando el mapeo index -> ID)
        for idx, coord in enumerate(self.data.node_coordinates):
            node_id = self.data.node_id_map_rev.get(idx, str(idx))
            # Se asume que data tiene un atributo 'node_types'
            node_type = self.data.node_types[idx] if hasattr(self.data, 'node_types') else "default"
            style = self._get_node_style(node_type)
            title = f"{node_type.upper()}: {node_id}"
            # Agregar información adicional si se dispone (por ejemplo, demanda)
            net.add_node(n_id=node_id,
                         label=node_id,
                         title=title,
                         x=coord[0],
                         y=coord[1],
                         color=style.get("color"),
                         shape=style.get("shape"),
                         size=style.get("size"))
        # Agregar enlaces: se recorre edge_index y se usa link_types
        # Se asume que data.edge_index está en forma [2, num_edges]
        edge_index = self.data.edge_index.cpu().numpy()
        num_edges = edge_index.shape[1]
        for e in range(num_edges):
            source_idx = edge_index[0, e]
            target_idx = edge_index[1, e]
            source_id = self.data.node_id_map_rev.get(source_idx, str(source_idx))
            target_id = self.data.node_id_map_rev.get(target_idx, str(target_idx))
            # Obtener tipo de enlace
            link_type = self.data.link_types[e] if e < len(self.data.link_types) else "default"
            style = self._get_link_style(link_type)
            # Determinar color: si hay flujo observado y se compara con el predicho
            pred_flow = self.predicted_flows[e].item()
            obs_flow = self.observed_flows_dict.get(e, None)
            if obs_flow is not None:
                tol = self.config.get("link_observed_tolerance", 0.1)
                if abs(pred_flow - obs_flow) / (obs_flow + 1e-8) <= tol:
                    edge_color = style.get("color_observed_match")
                else:
                    edge_color = style.get("color_observed_mismatch")
                # Formatear etiqueta con ambos valores
                label = f"Pred: {pred_flow:{self.config['link_label_format']}}\nObs: {obs_flow:{self.config['link_label_format']}}"
            else:
                edge_color = style.get("color_unobserved")
                label = f"{pred_flow:{self.config['link_label_format']}}"
            # Calcular ancho del enlace
            width = self._calculate_edge_width(pred_flow, style)
            net.add_edge(source=source_id,
                         to=target_id,
                         value=pred_flow,
                         title=label,
                         color=edge_color,
                         width=width,
                         dashes=style.get("dashes", False))
        return net

    def save_visualization(self, filepath=None):
        """
        Genera la visualización de la red y la guarda en un archivo HTML.

        Args:
            filepath (str, opcional): Ruta donde se guardará el archivo HTML.
                Si no se proporciona, se usa la ruta predeterminada de la configuración.
        """
        if filepath is None:
            filepath = self.config.get("export_filepath", "network_visualization.html")
        logging.info(f"Generando visualización y guardando en '{filepath}' ...")
        net = self._create_network()
        try:
            net.show(filepath)
            logging.info("Visualización guardada exitosamente.")
        except Exception as e:
            logging.error(f"Error al guardar la visualización: {e}", exc_info=True)
