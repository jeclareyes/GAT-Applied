# visualization/folium_viz.py
import logging
import folium
from configs.viz_styles import FOLIUM_MAP_ZOOM_START, folium_style
from visualization.utils import get_map_center

logger = logging.getLogger(__name__)

class FoliumMapBuilder:
    """
    Construye un mapa interactivo con Folium para segmentos y nodos.
    """
    def __init__(self, zoom_start=FOLIUM_MAP_ZOOM_START):
        self.zoom_start = zoom_start
        self.logger = logger

    def build(self, gdf_layers, center_gdf=None):
        """
        gdf_layers: list de GeoDataFrames para agregar (segmentos, nodos, marcadores)
        center_gdf: GeoDataFrame para centrar el mapa (por defecto usa centroid de primeros datos)
        """

        if center_gdf is None and gdf_layers:
            center_gdf = gdf_layers[0]
        if center_gdf is None or center_gdf.empty:
            m = folium.Map(zoom_start=self.zoom_start)
        else:
            center = get_map_center(center_gdf)
            m = folium.Map(location=center, zoom_start=self.zoom_start)

        for gdf in gdf_layers:
            if gdf is None or gdf.empty:
                continue
            folium.GeoJson(
                gdf.to_json(),
                style_function=folium_style
            ).add_to(m)
            self.logger.debug(f'Agregada capa GeoJSON al mapa, registros: {len(gdf)}')

        self.logger.info('Mapa interactivo Folium construido.')
        return m

