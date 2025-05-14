# graph_tools.py: unificación de builder.py, analyzer.py y exporter.py

import logging
import networkx as nx
import json
import pandas as pd
from shapely.geometry import Point

logger = logging.getLogger(__name__)

class GraphBuilder:
    def __init__(self, node_type= 'node_type', edge_type='edge_type'):
        self.node_type = node_type
        self.edge_type = edge_type

    def build(self, gdf_segments, gdf_nodes):
        G = nx.DiGraph()

        # Iterando sobre los nodos
        for idx, row in gdf_nodes.iterrows():
            pos= row.geometry.x, row.geometry.y
            exclude = ['X', 'Y', 'geometry']
            attrs = {'pos': pos}
            attrs = {k: v for k, v in row.items() if k not in exclude}
            G.add_node(idx,
                       **attrs)
        logger.info(f"Agregados {len(G.nodes())} nodos al grafo")

        # Iterando sobre los links
        added = 0
        for idx, row in gdf_segments.iterrows():
            geom = row.geometry
            if geom is None or geom.is_empty or len(geom.coords) < 2:
                continue

            if row['edge_type'] == 'road':
                # Existen los dos links INODE y JNODE definen la direccion
                exclude = []
                attrs = {k: v for k, v in row.items() if k not in exclude}
                G.add_edge(row['INODE'], row['JNODE'], **attrs)
                added += 1
            if row['edge_type'] != 'road':
                # hay que duplicarlo y que hay que hacer tambien JNODE e INODE 
                # para hacer la otra direccion
                exclude = ['geometry']
                attrs = {k: v for k, v in row.items() if k not in exclude}
                G.add_edge(row['INODE'], row['JNODE'], **attrs)
                added += 1
                G.add_edge(row['JNODE'], row['INODE'], **attrs)
                added += 1
        logger.info(f"Agregadas {added} aristas al grafo")
        return G


class GraphAnalyzer:
    def analyze(self, G):
        report = {}
        wcc = list(nx.weakly_connected_components(G))
        report['n_componentes'] = len(wcc)
        report['tamaños_componentes'] = [len(c) for c in wcc]

        aislados = [n for n in G.nodes() if G.in_degree(n) == 0 and G.out_degree(n) == 0]
        report['n_aislados'] = len(aislados)

        in_deg = dict(G.in_degree())
        out_deg = dict(G.out_degree())
        report['grado_max_entrada'] = max(in_deg.values(), default=0)
        report['grado_max_salida'] = max(out_deg.values(), default=0)

        try:
            ciclos = list(nx.simple_cycles(G))
            report['n_ciclos'] = len(ciclos)
        except Exception as e:
            logger.error(f"Error detectando ciclos: {e}")
            report['n_ciclos'] = None

        logger.info(f"Análisis completado: {report}")
        return report


class GraphExporter:
    def __init__(self, prefix_path):
        self.prefix = prefix_path

    def export_graphml(self, G):
        G_copy = self._sanitize_attributes(G)
        path = f"{self.prefix}.graphml"
        nx.write_graphml(G_copy, path)
        logger.info(f"Grafo exportado a GraphML en {path}")
        return path

    def export_json(self, G):
        data = nx.readwrite.json_graph.node_link_data(G)
        path = f"{self.prefix}.json"
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(f"Grafo exportado a JSON en {path}")
        return path

    def _sanitize_attributes(self, G):
        G_copy = G.copy()

        def sanitize(value):
            if pd.isna(value) or value is None:
                return ""
            if isinstance(value, (list, dict, tuple, set)):
                return str(value)
            return value

        for n, attrs in G_copy.nodes(data=True):
            for k, v in list(attrs.items()):
                G_copy.nodes[n][k] = sanitize(v)

        for u, v, attrs in G_copy.edges(data=True):
            for k, val in list(attrs.items()):
                G_copy.edges[u, v][k] = sanitize(val)

        for k, v in list(G_copy.graph.items()):
            G_copy.graph[k] = sanitize(v)

        return G_copy