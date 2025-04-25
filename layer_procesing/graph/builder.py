# graph/builder.py
import logging
import networkx as nx
from shapely.geometry import Point

class GraphBuilder:
    """
    Construye un grafo dirigido a partir de GeoDataFrames de segmentos y nodos.
    """
    def __init__(self, direction_field='DIRECTION'):
        self.direction_field = direction_field
        self.logger = logger

    def build(self, gdf_segments, gdf_nodes):
        G = nx.DiGraph()

        # Mapeo coords a node_id
        coord_to_id = {}
        for idx, row in gdf_nodes.iterrows():
            node_id = f"node_{idx}"
            x, y = row.geometry.x, row.geometry.y
            coord_key = (round(x, 6), round(y, 6))
            coord_to_id[coord_key] = node_id
            G.add_node(node_id,
                       tipo=row.get('tipo'),
                       coord_x=float(x),
                       coord_y=float(y))
        self.logger.info(f"Agregados {len(G.nodes())} nodos al grafo")

        # Agregar aristas desde segmentos
        added = 0
        for idx, row in gdf_segments.iterrows():
            geom = row.geometry
            if geom is None or geom.is_empty or len(geom.coords) < 2:
                continue
            start, end = geom.coords[0], geom.coords[-1]
            start_key = (round(start[0], 6), round(start[1], 6))
            end_key = (round(end[0], 6), round(end[1], 6))
            if start_key not in coord_to_id or end_key not in coord_to_id:
                self.logger.debug(f"Segmento {idx} sin nodos definidos")
                continue
            src = coord_to_id[start_key]
            tgt = coord_to_id[end_key]
            # Determinar sentido
            direction = row.get(self.direction_field, 'forward')
            if isinstance(direction, str) and direction.lower() == 'reverse':
                source, target = tgt, src
            else:
                source, target = src, tgt
            # Filtrar atributos no serializables
            attrs = {k: v for k, v in row.items() if k != 'geometry'}
            G.add_edge(source, target, **attrs)
            added += 1
        self.logger.info(f"Agregadas {added} aristas al grafo")
        return G

logger = logging.getLogger(__name__)


