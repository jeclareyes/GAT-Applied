# map_visualization.py: fusión de folium_viz.py, networkx_viz.py y utils.py

import logging
import folium
import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
from configs.viz_styles import (
    FOLIUM_MAP_ZOOM_START, folium_style,
    NODE_SIZE, ARROW_STYLE, ARROW_SIZE,
    EDGE_COLOR, NODE_COLOR_DEFAULT
)

logger = logging.getLogger(__name__)

# --- Backend para matplotlib ---
SAVE_FIGURE_INSTEAD_OF_SHOW = True
SAVE_FILENAME = "grafo_vial.png"
if SAVE_FIGURE_INSTEAD_OF_SHOW:
    matplotlib.use('Agg')
else:
    try:
        matplotlib.use('Qt5Agg')
    except:
        try:
            matplotlib.use('TkAgg')
        except:
            pass


def get_map_center(gdf):
    try:
        centroid = gdf.geometry.centroid
        lat = centroid.y.mean()
        lon = centroid.x.mean()
        return [lat, lon]
    except Exception as e:
        logger.error(f'Error calculando centro del mapa: {e}')
        return [0, 0]


class FoliumMapBuilder:
    def __init__(self, zoom_start=FOLIUM_MAP_ZOOM_START):
        self.zoom_start = zoom_start

    def build(self, gdf_layers, center_gdf=None):
        if center_gdf is None and gdf_layers:
            center_gdf = gdf_layers[0]
        center = get_map_center(center_gdf) if center_gdf is not None else [0, 0]
        m = folium.Map(location=center, zoom_start=self.zoom_start)

        for gdf in gdf_layers:
            if gdf is None or gdf.empty:
                continue
            folium.GeoJson(
                gdf.to_json(),
                style_function=folium_style
            ).add_to(m)
        logger.info('Mapa interactivo Folium construido.')
        return m


class NetworkXPlotter:
    def __init__(self, node_size=NODE_SIZE, arrowstyle=ARROW_STYLE,
                 arrowsize=ARROW_SIZE, edge_color=EDGE_COLOR, node_color=NODE_COLOR_DEFAULT):
        self.node_size = node_size
        self.arrowstyle = arrowstyle
        self.arrowsize = arrowsize
        self.edge_color = edge_color
        self.node_color = node_color

    def plot(self, G, with_labels=False, title='Grafo Vial'):
        if G.number_of_nodes() == 0:
            logger.warning('El grafo no tiene nodos para visualizar.')
            return

        pos = {n: (d.get('coord_x', 0), d.get('coord_y', 0)) for n, d in G.nodes(data=True)}
        plt.figure(figsize=(12, 8))
        nx.draw(G, pos,
                with_labels=with_labels,
                node_size=self.node_size,
                arrowstyle=self.arrowstyle,
                arrowsize=self.arrowsize,
                node_color=self.node_color,
                edge_color=self.edge_color)
        plt.title(title)
        plt.tight_layout()

        if SAVE_FIGURE_INSTEAD_OF_SHOW:
            plt.savefig(SAVE_FILENAME)
            logger.info(f'Visualización NetworkX guardada en {SAVE_FILENAME}.')
        else:
            plt.show()
            logger.info('Visualización NetworkX mostrada.')
