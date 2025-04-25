# visualization/utils.py
import logging

logger = logging.getLogger(__name__)


def get_map_center(gdf):
    """
    Calcula el centro geográfico de un GeoDataFrame.
    Retorna [lat, lon].
    """
    try:
        centroid = gdf.geometry.centroid
        lat = centroid.y.mean()
        lon = centroid.x.mean()
        return [lat, lon]
    except Exception as e:
        logger.error(f'Error calculando centro del mapa: {e}')
        return [0, 0]
