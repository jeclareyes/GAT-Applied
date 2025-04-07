# visualization/network_viz.py
import json
import logging
import networkx as nx
from pyvis.network import Network
import os

#%% Ouputs
output_dir = os.path.join(os.getcwd())

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


class NetworkVisualizer_Pyvis:
    def __init__(self, nodes, links, estimated_flows, observed_flows, config=None):
        """
        Inicializa el visualizador de red usando tensores directamente.

        Args:
            nodes (tensor): Tensor de forma (N, 2) con las coordenadas de cada nodo.
            links (tensor): Tensor de forma (2, E) con índices (origen, destino).
            estimated_flows (tensor): Tensor de forma (E,) o (E,1) con flujos estimados.
            observed_flows (tensor): Tensor de forma (E,) o (E,1) con flujos observados.
        """
        self.nodes_tensor = nodes
        self.links_tensor = links
        self.estimated_flows = estimated_flows.view(-1)
        self.observed_flows = observed_flows.view(-1)

        # Configuración por defecto – se pueden actualizar posteriormente
        default_config = {
            "height": "600px",
            "width": "100%",
            "notebook": False,
            "export_filepath": output_dir,  # Ruta y nombre del HTML exportado
            "bidirectional": True,
            "show_node_labels": True,
            "node_label_font_size": 20,
            "node_size_scaling_method": "flow",  # Opciones: "flow", "degree", "betweenness"
            "node_size_multiplier": 1.0,  # float
            "default_node_size": 20,
            "node_default_color": "#97c2fc",

            "show_estimated_flows": True,
            "show_observed_flows": True,
            "edge_label_mode": "combined",  # Opciones: "combined", "estimated", "observed"
            "show_edge_labels": True,
            "base_edge_width": 1.0,  # float
            "edge_width_scaling": 0.01,  # float
            "flow_tolerance": 5.0,  # float
            "edge_label_font_size": 10,
            "edge_default_color": "#CCCCCC",

            "gravitational_constant": -100,
            "spring_length": 50,
            "show_buttons": False  # Mostrar botones nativos de vis.js
        }
        self.config = default_config
        if config:
            self.config.update(config)

        # Se usa el índice (0 a N-1) como ID
        self.num_nodes = nodes.shape[0]
        self.node_metrics = {}  # Para escalado de tamaño

    def update_config(self, **kwargs):
        """
        Actualiza la configuración con las opciones proporcionadas.
        """
        self.config.update(kwargs)
        logging.info("Configuración actualizada: %s", kwargs)
        return self

    def _compute_node_metrics(self):
        method = self.config.get("node_size_scaling_method", "flow")
        G = nx.DiGraph()

        # Agregamos nodos (ID: 0, 1, 2, ..., N-1)
        for i in range(self.num_nodes):
            G.add_node(i)

        # Número de enlaces
        num_edges = self.links_tensor.shape[1]
        for i in range(num_edges):
            u = int(self.links_tensor[0, i].item())
            v = int(self.links_tensor[1, i].item())
            if self.config["bidirectional"]:
                G.add_edge(u, v)
                G.add_edge(v, u)
            else:
                G.add_edge(u, v)

        metrics = {}
        if method == "flow":
            for node in G.nodes():
                inflow = sum(self.estimated_flows[i].item()
                             for i in range(num_edges)
                             if int(self.links_tensor[1, i].item()) == node)
                outflow = sum(self.estimated_flows[i].item()
                              for i in range(num_edges)
                              if int(self.links_tensor[0, i].item()) == node)
                metrics[node] = inflow + outflow
        elif method == "degree":
            for node in G.nodes():
                metrics[node] = G.degree(node)
        elif method == "betweenness":
            betw = nx.betweenness_centrality(G)
            metrics = betw
        else:
            for node in G.nodes():
                metrics[node] = 1

        # Normalización para evitar tamaños excesivos
        if metrics:
            max_val = max(metrics.values()) if max(metrics.values()) != 0 else 1
            for node in metrics:
                metrics[node] = (metrics[node] / max_val) * self.config["default_node_size"] * self.config[
                    "node_size_multiplier"]
        return metrics

    def _calculate_edge_width(self, flow_value):
        return self.config['base_edge_width'] + (flow_value * self.config['edge_width_scaling'])

    def _get_edge_label(self, est, obs):
        mode = self.config.get("edge_label_mode", "combined")
        if mode == "estimated":
            return f"{est:.1f}" if self.config["show_edge_labels"] else ""
        elif mode == "observed":
            return f"{obs:.1f}" if (obs is not None and self.config["show_edge_labels"]) else ""
        else:
            if obs is not None and self.config["show_edge_labels"]:
                return f"{est:.1f} (est) / {obs:.1f} (obs)"
            else:
                return f"{est:.1f} (est)" if self.config["show_edge_labels"] else ""

    def _get_edge_color(self, est, obs):
        # Se puede permitir que el usuario fuerce un color por defecto
        if not self.config.get("show_estimated_flows", True) and not self.config.get("show_observed_flows", True):
            return self.config.get("edge_default_color", "#CCCCCC")
        if obs is None:
            return "#FFA500"  # Naranja
        diff = est - obs
        tol = self.config.get("flow_tolerance", 5.0)
        if abs(diff) < tol:
            return "#4CAF50"  # Verde
        elif diff > 0:
            return "#FF0000"  # Rojo
        else:
            return "#2196F3"  # Azul

    def draw(self, html_filepath=None):
        """
        Genera la visualización y exporta el HTML.

        Args:
            html_filepath (str): Ruta y nombre del archivo HTML a exportar.
                                 Si no se especifica, se usa self.config["export_filepath"].

        Returns:
            str: Contenido HTML generado.
        """
        if html_filepath is None:
            html_filepath = self.config.get("export_filepath")
        net = Network(
            height=self.config['height'],
            width=self.config['width'],
            directed=True,
            notebook=self.config['notebook']
        )

        net.force_atlas_2based(
            gravity=self.config['gravitational_constant'],
            spring_length=self.config['spring_length'],
        )

        if self.config.get("show_buttons", False):
            net.show_buttons(filter_=["physics", "nodes", "edges", "layout"])

        self.node_metrics = self._compute_node_metrics()

        # Agregar nodos: se usa el índice como ID y se extraen las coordenadas
        for i in range(self.num_nodes):
            x = self.nodes_tensor[i, 0].item()
            y = self.nodes_tensor[i, 1].item()
            node_id = i
            label = f"({x:.0f}, {y:.0f})" if self.config["show_node_labels"] else ""
            size = self.node_metrics.get(i, self.config["default_node_size"])
            color = self.config.get("node_default_color", "#97c2fc")
            net.add_node(
                node_id,
                label=label,
                size=size,
                color=color,
                x=x,
                y=y,
                font={'size': self.config['node_label_font_size']}
            )

        # Agregar aristas
        num_edges = self.links_tensor.shape[1]
        processed = set()
        for i in range(num_edges):
            u = int(self.links_tensor[0, i].item())
            v = int(self.links_tensor[1, i].item())
            if self.config["bidirectional"]:
                pair = tuple(sorted((u, v)))
                if pair in processed:
                    continue
                processed.add(pair)
                # Sumar flujos en ambos sentidos, buscando índice inverso
                idx_rev = self._find_reverse_edge_index(u, v, num_edges)
                est = self.estimated_flows[i].item()
                if idx_rev is not None:
                    est += self.estimated_flows[idx_rev].item()
                idx_rev_obs = self._find_reverse_edge_index(u, v, num_edges)
                if idx_rev_obs is not None:
                    obs = self.observed_flows[i].item() + self.observed_flows[idx_rev_obs].item()
                else:
                    obs = self.observed_flows[i].item()
            else:
                est = self.estimated_flows[i].item()
                obs = self.observed_flows[i].item()

            if (self.config['show_estimated_flows'] and est > 0) or (
                    self.config['show_observed_flows'] and obs is not None):
                label = self._get_edge_label(est, obs)
                color = self._get_edge_color(est, obs)
                width = self._calculate_edge_width(est)
                net.add_edge(u, v,
                             title=label,
                             label=label,
                             color=color,
                             width=width,
                             font={'size': self.config['edge_label_font_size']})
        net.save_graph(html_filepath)
        logging.info("Visualización guardada en %s", html_filepath)
        with open(html_filepath, "r", encoding="utf-8") as f:
            html_content = f.read()
        return html_content

    def _find_reverse_edge_index(self, u, v, num_edges):
        """
        Busca el índice de la arista inversa (v, u) en el tensor de enlaces.
        Retorna el índice si se encuentra; de lo contrario, None.
        """
        for j in range(num_edges):
            u_j = int(self.links_tensor[0, j].item())
            v_j = int(self.links_tensor[1, j].item())
            if u_j == v and v_j == u:
                return j
        return None

    def load_config(self, filename="network_config.json"):
        with open(filename, 'r') as f:
            loaded_config = json.load(f)
            self.config.update(loaded_config)
        logging.info("Configuración cargada desde %s", os.path.join(output_dir,filename))
        return self

    def save_config(self, filename="network_config.json"):
        with open(filename, 'w') as f:
            json.dump(self.config, f, indent=4)
        logging.info("Configuración guardada en %s", os.path.join(output_dir, filename))
        return self
