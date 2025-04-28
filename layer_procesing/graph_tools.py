# graph_tools.py: unificación de builder.py, analyzer.py y exporter.py

import logging
import networkx as nx
import json
from shapely.geometry import Point

logger = logging.getLogger(__name__)

class GraphBuilder:
    def __init__(self, direction_field='DIRECTION'):
        self.direction_field = direction_field

    def build(self, gdf_segments, gdf_nodes):
        G = nx.DiGraph()
        coord_to_id = {}

        for idx, row in gdf_nodes.iterrows():
            node_id = f"node_{idx}"
            x, y = row.geometry.x, row.geometry.y
            key = (round(x, 6), round(y, 6))
            coord_to_id[key] = node_id
            G.add_node(node_id,
                       tipo=row.get('tipo'),
                       coord_x=float(x),
                       coord_y=float(y))
        logger.info(f"Agregados {len(G.nodes())} nodos al grafo")

        added = 0
        for idx, row in gdf_segments.iterrows():
            geom = row.geometry
            if geom is None or geom.is_empty or len(geom.coords) < 2:
                continue
            start, end = geom.coords[0], geom.coords[-1]
            skey = (round(start[0], 6), round(start[1], 6))
            ekey = (round(end[0], 6), round(end[1], 6))
            if skey not in coord_to_id or ekey not in coord_to_id:
                logger.debug(f"Segmento {idx} sin nodos definidos")
                continue
            src, tgt = coord_to_id[skey], coord_to_id[ekey]
            direction = row.get(self.direction_field, 'forward')
            source, target = (tgt, src) if str(direction).lower() == 'reverse' else (src, tgt)
            attrs = {k: v for k, v in row.items() if k != 'geometry'}
            G.add_edge(source, target, **attrs)
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
        for n, attrs in G_copy.nodes(data=True):
            for k, v in list(attrs.items()):
                if isinstance(v, (list, dict, tuple, set)):
                    G_copy.nodes[n][k] = str(v)
        for u, v, attrs in G_copy.edges(data=True):
            for k, val in list(attrs.items()):
                if isinstance(val, (list, dict, tuple, set)):
                    G_copy.edges[u, v][k] = str(val)
        return G_copy