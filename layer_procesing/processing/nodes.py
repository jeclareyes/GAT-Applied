# processing/nodes.py
import logging
from shapely.geometry import Point
import geopandas as gpd

logger = logging.getLogger(__name__)


class NodeIdentifier:
    """
    Clase para identificar nodos de intersección y terminales en una red de segmentos.
    """

    def __init__(self, precision=6):
        self.precision = precision

    def identify(self, gdf):
        """
        Procesa un GeoDataFrame de líneas y retorna dos GeoDataFrames:
          - nodos: intersecciones y nodos finales con conectividad
          - orphan_segments: segmentos no conectados
        """
        coord_dict = {}  # {coordenada: [ELEMENT_IDs]}
        segment_dict = {}  # {segment_idx: (inicio_key, fin_key)}
        for idx, row in gdf.iterrows():
            coords = list(row.geometry.coords)
            eid = row.get("ELEMENT_ID", idx)
            inicio = coords[0]
            fin = coords[-1]
            inicio_key = (round(inicio[0], self.precision), round(inicio[1], self.precision))
            fin_key = (round(fin[0], self.precision), round(fin[1], self.precision))

            for key in [inicio_key, fin_key]:
                coord_dict.setdefault(key, {'count': 0, 'elementos': []})
                coord_dict[key]['count'] += 1
                coord_dict[key]['elementos'].append(eid)

            segment_dict[idx] = (inicio_key, fin_key)

        # Crear GeoDataFrame de nodos
        nodes = []
        for i, (coord, info) in enumerate(coord_dict.items()):
            tipo = 'Intersección' if info['count'] > 1 else 'Nodo_final'
            in_degree = info['count'] // 2
            out_degree = info['count'] - in_degree
            conectividad = in_degree + out_degree
            nodes.append({
                'fid': i + 1,
                'tipo': tipo,
                'conteo': info['count'],
                'elementos': info['elementos'],
                'node_id': f'node_{i}',
                'in_degree': in_degree,
                'out_degree': out_degree,
                'conectividad': conectividad,
                'geometry': Point(coord)
            })
        gdf_nodes = gpd.GeoDataFrame(nodes, geometry='geometry', crs=gdf.crs)

        # Detectar segmentos huérfanos
        orphan_idxs = []
        for idx, (start_key, end_key) in segment_dict.items():
            if coord_dict[start_key]['count'] == 1 and coord_dict[end_key]['count'] == 1:
                orphan_idxs.append(idx)

        if orphan_idxs:
            orphan_segs = gdf.loc[orphan_idxs].copy()
            logger.info(f"Encontrados {len(orphan_idxs)} segmentos huérfanos")
        else:
            orphan_segs = None
            logger.info("No se encontraron segmentos huérfanos")

        return gdf_nodes, orphan_segs
