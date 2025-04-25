# graph/exporter.py
import logging
import networkx as nx
import json
import os

logger = logging.getLogger(__name__)

class GraphExporter:
    """
    Exporta grafos a GraphML y JSON, asegurando compatibilidad de tipos.
    """
    def __init__(self, prefix_path):
        self.prefix = prefix_path
        self.logger = logger

    def export_graphml(self, G):
        # Asegurar atributos simples
        G_copy = G.copy()
        for n, attrs in G_copy.nodes(data=True):
            for k, v in list(attrs.items()):
                if isinstance(v, (list, dict, tuple, set)):
                    G_copy.nodes[n][k] = str(v)
        for u, v, attrs in G_copy.edges(data=True):
            for k, val in list(attrs.items()):
                if isinstance(val, (list, dict, tuple, set)):
                    G_copy.edges[u, v][k] = str(val)
        path = f"{self.prefix}.graphml"
        nx.write_graphml(G_copy, path)
        self.logger.info(f"Grafo exportado a GraphML en {path}")
        return path

    def export_json(self, G, route=None):
        data = nx.readwrite.json_graph.node_link_data(G)
        path = f"{self.prefix}.json"
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        self.logger.info(f"Grafo exportado a JSON en {path}")
        return path