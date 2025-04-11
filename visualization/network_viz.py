# visualization/network_viz.py
import json
import logging
import math
import os
import torch
from pyvis.network import Network

# NetworkX es opcional ahora, solo si se usa para métricas de tamaño de nodo
# import networkx as nx

# Configuración del logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class NetworkVisualizer_Pyvis:
    def __init__(self, data, predicted_flows, config=None):
        """
        Inicializa el visualizador usando el objeto Data de PyG y las predicciones del modelo.

        Args:
            data (torch_geometric.data.Data): Objeto Data con toda la información de la red
                                              (incluyendo .node_types, .link_types, .node_coordinates,
                                              .observed_flow_indices, .observed_flow_values, .node_id_map_rev).
            predicted_flows (torch.Tensor): Tensor con los flujos predichos por el modelo GAT
                                            para cada arista en data.edge_index (forma [num_edges]).
            config (dict, opcional): Diccionario para sobreescribir la configuración por defecto.
        """
        if not hasattr(data, 'link_types'):
            raise ValueError(
                "El objeto 'data' debe contener el atributo 'link_types'. Asegúrate de que 'loader.py' lo añade.")
        if not hasattr(data, 'node_coordinates'):
            raise ValueError("El objeto 'data' debe contener el atributo 'node_coordinates'. Verifica 'loader.py'.")
        if not hasattr(data, 'node_id_map_rev'):
            raise ValueError("El objeto 'data' debe contener el atributo 'node_id_map_rev'. Verifica 'loader.py'.")

        self.data = data
        self.predicted_flows = predicted_flows.detach().cpu().view(-1)  # Asegurar que está en CPU y es 1D

        # Crear diccionario de flujos observados para búsqueda rápida: {edge_index: value}
        self.observed_flows_dict = {
            idx.item(): val.item()
            for idx, val in zip(data.observed_flow_indices.cpu(), data.observed_flow_values.cpu())
        }

        # --- Configuración por Defecto (Inspirada en graph_generator_g.py) ---
        default_config = {
            # --- General ---
            "height": "750px",
            "width": "100%",
            "notebook": False,
            "bgcolor": "white",
            "font_color": "black",  # Color de fuente general (puede ser sobreescrito)
            "export_filepath": "visualization/exported_viz_data/gat_network_visualization.html",
            "show_buttons": True,  # Mostrar botones de control de PyVis

            # --- Nodos ---
            "node_styles": {
                'zat': {'color': "#FFD700", 'shape': 'diamond', 'size': 20, 'font_weight': 'bold',
                        'border_color': "black", 'border_width': 2},
                'intersection': {'color': "#87CEEB", 'shape': 'dot', 'size': 12, 'border_color': "darkgrey",
                                 'border_width': 1},
                'default': {'color': "#D3D3D3", 'shape': 'ellipse', 'size': 10}  # Para tipos desconocidos
            },
            "node_label_font_size": 12,
            "node_show_label": True,  # Mostrar ID de nodo ('Z0', 'I1')
            "node_show_demand_title": True,  # Añadir Gen/Attr al tooltip de ZATs

            # --- Enlaces ---
            "link_styles": {
                # Colores: si hay obs. y coincide, si hay obs. y no coincide, si no hay obs.
                'road': {'color_observed_match': "#2E8B57",  # Verde mar
                         'color_observed_mismatch': "#DC143C",  # Carmesí
                         'color_unobserved': "#A9A9A9",  # Gris oscuro
                         'width_factor': 0.8, 'dashes': False},
                'logical': {'color_observed_match': "#ADD8E6",  # Azul claro
                            'color_observed_mismatch': "#FFA07A",  # Salmón claro
                            'color_unobserved': "#E0E0E0",  # Gris muy claro
                            'width_factor': 0.5, 'dashes': [5, 5]},
                'default': {'color_observed_match': "#CCCCCC", 'color_observed_mismatch': "#CCCCCC",
                            'color_unobserved': "#CCCCCC",
                            'width_factor': 0.5, 'dashes': [2, 2]}  # Para tipos desconocidos
            },
            "link_show_label": True,  # Mostrar etiqueta de flujo en el enlace
            "link_label_font_size": 9,
            "link_label_prefixes": {'predicted': "GAT:", 'observed': "Obs:"},
            "link_label_format": ".1f",  # Formato para mostrar flujos (e.g., 1 decimal)
            "link_observed_tolerance": 0.1,  # Tolerancia *relativa* para considerar "match" (abs(pred-obs)/obs <= tol)
            "link_width_base": 1.0,  # Ancho mínimo
            "link_width_scaling_method": "log",  # 'linear' o 'log' para escalar ancho con flujo
            "link_max_width": 8.0,  # Ancho máximo visual

            # --- Física (Ejemplo: forceAtlas2Based) ---
            "physics_solver": "forceAtlas2Based",
            "physics_options": {
                "forceAtlas2Based": {
                    "gravitationalConstant": -50, "centralGravity": 0.01,
                    "springLength": 150, "springConstant": 0.08, "damping": 0.4,
                    "avoidOverlap": 0.5
                },
                "stabilization": {"enabled": True, "iterations": 1000, "fit": True}
            }
        }
        self.config = default_config
        if config:
            # Fusiona la configuración proporcionada con la por defecto
            # Nota: para diccionarios anidados como 'node_styles', esto reemplaza, no fusiona profundamente.
            # Si necesitas fusión profunda, considera usar una librería o una función recursiva.
            self.config.update(config)

        self._validate_config()  # Validar configuración inicial

    def _validate_config(self):
        # Añadir validaciones básicas si es necesario
        if "node_styles" not in self.config or "link_styles" not in self.config:
            logging.warning(
                "Configuración incompleta: faltan 'node_styles' o 'link_styles'. Se usarán valores por defecto si existen.")
        # Podrías añadir más validaciones (tipos, rangos, etc.)

    def update_config(self, **kwargs):
        """Actualiza la configuración con las opciones proporcionadas."""
        self.config.update(kwargs)
        self._validate_config()  # Re-validar después de actualizar
        logging.info("Configuración del visualizador actualizada.")
        return self

    def _get_node_style(self, node_type):
        """Obtiene el diccionario de estilo para un tipo de nodo."""
        return self.config.get("node_styles", {}).get(node_type,
                                                      self.config.get("node_styles", {}).get('default', {}))

    def _get_link_style(self, link_type):
        """Obtiene el diccionario de estilo para un tipo de enlace."""
        return self.config.get("link_styles", {}).get(link_type,
                                                      self.config.get("link_styles", {}).get('default', {}))

    def _format_number(self, value, format_spec=".1f"):
        """Formatea un número según la especificación."""
        if value is None: return "N/A"
        try:
            return f"{value:{format_spec}}"
        except (ValueError, TypeError):
            return str(value)  # Fallback a string simple

    def _calculate_edge_width(self, flow_value, style):
        """Calcula el ancho del enlace basado en el flujo y el método de escalado."""
        base = self.config.get('link_width_base', 1.0)
        factor = style.get('width_factor', 0.5)  # Usa el factor del estilo específico
        max_w = self.config.get('link_max_width', 8.0)
        method = self.config.get('link_width_scaling_method', 'log')

        if flow_value <= 0: return base
        scaled_width = base

        if method == 'log':
            # Usar log1p para manejar flujos cercanos a 0
            scaled_width = base + math.log1p(flow_value) * factor
        elif method == 'linear':
            scaled_width = base + flow_value * factor
        else:  # Sin escalado
            scaled_width = base

        return min(scaled_width, max_w)  # Aplicar límite máximo

    def draw(self, html_filepath=None):
        """
        Genera la visualización interactiva y la guarda en un archivo HTML.

        Args:
            html_filepath (str, opcional): Ruta donde guardar el archivo HTML.
                                          Si es None, usa self.config["export_filepath"].

        Returns:
            str: Ruta absoluta al archivo HTML generado.
        """
        filepath = html_filepath if html_filepath else self.config.get("export_filepath", "network_visualization.html")
        output_dir = os.path.dirname(filepath)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            logging.info(f"Directorio de salida creado: {output_dir}")

        net = Network(
            height=self.config['height'],
            width=self.config['width'],
            directed=True,
            notebook=self.config['notebook'],
            bgcolor=self.config['bgcolor'],
            font_color=self.config['font_color']
        )

        # Configurar física
        physics_solver = self.config.get("physics_solver", "forceAtlas2Based")
        physics_opts_all = self.config.get("physics_options", {})
        physics_specific_opts = physics_opts_all.get(physics_solver, {})
        stabilization_opts = physics_opts_all.get("stabilization", {})

        # Aplicar opciones de física (simplificado, PyVis tiene métodos específicos)
        # net.set_options(...) es más flexible si la estructura JSON coincide
        """options_json = {
            "physics": {
                "enabled": True,
                "solver": physics_solver,
                physics_solver: physics_specific_opts,
                "stabilization": stabilization_opts
            }
        }"""

        options_json = {"configure": {},  # Inicializa el objeto configure vacío
                        "physics": {"enabled": True, "solver": physics_solver, physics_solver: physics_specific_opts,
                                    "stabilization": stabilization_opts}}

        # Mostrar botones si está configurado
        if self.config.get("show_buttons", False):
            net.show_buttons(filter_=["physics", "nodes", "edges", "layout", "interaction"])
        net.set_options(json.dumps(options_json))

        # --- Agregar Nodos ---
        node_map_rev = self.data.node_id_map_rev
        coords = self.data.node_coordinates.cpu().numpy()
        node_types = self.data.node_types

        for i in range(self.data.num_nodes):
            node_id_str = node_map_rev.get(i, f"Node_{i}")  # ID original 'Z0', 'I1'
            node_type = node_types[i] if i < len(node_types) else 'default'
            style = self._get_node_style(node_type)

            label = node_id_str if self.config.get("node_show_label", True) else ""
            title = f"<b>{node_type.capitalize()}: {node_id_str}</b>"
            title += f"<br>Index: {i}"
            title += f"<br>Coords: ({coords[i, 0]:.1f}, {coords[i, 1]:.1f})"

            # Añadir demanda al título si es ZAT y está configurado
            if node_type == 'zat' and self.config.get("node_show_demand_title", True):
                demand = self.data.zat_demands.get(node_id_str, [0.0, 0.0])
                title += f"<br>Gen: {demand[0]:.1f}, Attr: {demand[1]:.1f}"

            net.add_node(
                int(i),  # ID interno para PyVis (índice numérico)
                label=label,
                title=title,  # Tooltip HTML
                shape=style.get('shape', 'ellipse'),
                size=style.get('size', 10),
                color=style.get('color', '#D3D3D3'),
                borderWidth=style.get('border_width', 1),
                # Nota: El color del borde usa 'border', no 'border_color' en PyVis
                # color={'background': style.get('color', '#D3D3D3'), 'border': style.get('border_color', '#2B7CE9')},
                # Intentar configurar color de borde directamente si funciona:
                border_color=style.get('border_color', 'black'),
                # Esto puede no funcionar en todas las versiones/formas

                font={'size': self.config.get("node_label_font_size", 12),
                      'face': 'arial',
                      'color': style.get('font_color', self.config.get('font_color', 'black'))
                      # 'weight': style.get('font_weight', 'normal') # Puede no ser soportado directamente
                      },
                # Usar coordenadas físicas si están disponibles (ayuda a layout inicial)
                x=float(coords[i, 0]) * 5,  # Multiplicar para escalar en el canvas
                y=float(coords[i, 1]) * 5 * -1  # Invertir Y y escalar
            )
            # Añadir borde explícitamente si es necesario (puede requerir manipulación del diccionario de nodos)
            node_dict = net.nodes[-1]  # Obtener el último nodo añadido
            node_dict['color'] = {'background': style.get('color', '#D3D3D3'),
                                  'border': style.get('border_color', 'black')}

        # --- Agregar Enlaces ---
        edge_index = self.data.edge_index.cpu().numpy()
        link_types = self.data.link_types
        num_edges = edge_index.shape[1]
        label_prefixes = self.config.get("link_label_prefixes", {})
        pred_prefix = label_prefixes.get('predicted', "Pred:")
        obs_prefix = label_prefixes.get('observed', "Obs:")
        label_format = self.config.get("link_label_format", ".1f")
        tolerance = self.config.get("link_observed_tolerance", 0.1)

        for i in range(num_edges):
            u_idx, v_idx = edge_index[0, i], edge_index[1, i]
            u_id_str = node_map_rev.get(u_idx, f"Node_{u_idx}")
            v_id_str = node_map_rev.get(v_idx, f"Node_{v_idx}")

            link_type = link_types[i] if i < len(link_types) else 'default'
            style = self._get_link_style(link_type)

            pred_flow = self.predicted_flows[i].item()
            obs_flow = self.observed_flows_dict.get(i)  # Puede ser None

            # Determinar color basado en observación y coincidencia
            edge_color = style.get('color_unobserved', '#CCCCCC')
            match_status = "unobserved"
            if obs_flow is not None:
                if abs(pred_flow - obs_flow) <= tolerance * abs(obs_flow) or abs(
                        pred_flow - obs_flow) < 1e-3:  # Tolerancia relativa o absoluta pequeña
                    edge_color = style.get('color_observed_match', '#00FF00')
                    match_status = "match"
                else:
                    edge_color = style.get('color_observed_mismatch', '#FF0000')
                    match_status = "mismatch"

            # Crear etiqueta del enlace
            label_parts = []
            if self.config.get("link_show_label", True):
                label_parts.append(f"{pred_prefix} {self._format_number(pred_flow, label_format)}")
                if obs_flow is not None:
                    label_parts.append(f"{obs_prefix} {self._format_number(obs_flow, label_format)}")
            label_text = "\n".join(label_parts)

            # Crear tooltip del enlace
            title = f"<b>Link: {u_id_str} → {v_id_str}</b> (Idx: {i})"
            title += f"<br>Type: {link_type}"
            title += f"<br>Predicted Flow: {self._format_number(pred_flow, '.3f')}"
            if obs_flow is not None:
                title += f"<br>Observed Flow: {self._format_number(obs_flow, '.3f')}"
                title += f"<br>Match Status: {match_status}"

            # Calcular ancho
            width = self._calculate_edge_width(pred_flow, style)

            net.add_edge(
                int(u_idx), int(v_idx),  # Usar índices numéricos para origen/destino
                title=title,
                label=label_text,
                width=width,
                color=edge_color,
                dashes=style.get('dashes', False),
                arrows={'to': {'enabled': True, 'scaleFactor': 0.6}},
                font={'size': self.config.get("link_label_font_size", 10),
                      'align': 'middle',
                      'color': self.config.get('font_color', 'black')},
                smooth={'enabled': True, 'type': 'dynamic', 'roundness': 0.5}  # Opciones de suavizado
            )

        # Guardar el grafo
        try:
            net.save_graph(filepath)
            abs_path = os.path.abspath(filepath)
            logging.info(f"Visualización de red guardada exitosamente en: {abs_path}")
            return abs_path
        except Exception as e:
            logging.error(f"Error al guardar el archivo HTML de PyVis en {filepath}: {e}", exc_info=True)
            return None

    def load_config(self, filename="visualization/network_config.json"):
        """Carga la configuración desde un archivo JSON."""
        try:
            with open(filename, 'r') as f:
                loaded_config = json.load(f)
            self.config.update(loaded_config)  # Fusiona (puede reemplazar diccionarios anidados)
            self._validate_config()
            logging.info(f"Configuración cargada desde {filename}")
        except FileNotFoundError:
            logging.warning(f"Archivo de configuración '{filename}' no encontrado. Usando configuración actual.")
        except json.JSONDecodeError:
            logging.error(f"Error al decodificar JSON desde '{filename}'. Verifica el formato.")
        except Exception as e:
            logging.error(f"Error cargando configuración desde '{filename}': {e}", exc_info=True)
        return self

    def save_config(self, filename="visualization/network_config.json"):
        """Guarda la configuración actual en un archivo JSON."""
        try:
            output_dir = os.path.dirname(filename)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
            with open(filename, 'w') as f:
                json.dump(self.config, f, indent=4)
            logging.info(f"Configuración guardada en {filename}")
        except Exception as e:
            logging.error(f"Error guardando configuración en '{filename}': {e}", exc_info=True)
        return self
