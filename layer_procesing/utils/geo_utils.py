# utils/geo_utils.py
import logging
import geopandas as gpd
from shapely.geometry import mapping, shape

logger = logging.getLogger(__name__)

def ensure_crs(gdf, target_crs):
    """
    Asegura que el GeoDataFrame esté en el CRS objetivo.
    Si no, reproyecta.
    """
    if gdf.crs != target_crs:
        try:
            return gdf.to_crs(target_crs)
        except Exception as e:
            logger.error(f"Error al reproyectar a {target_crs}: {e}")
    return gdf


def round_geometry_coords(gdf, precision=6):
    """
    Redondea las coordenadas de las geometrías a la precisión dada.
    """
    def _round_geom(geom):
        if geom is None:
            return None
        geo_json = mapping(geom)
        def round_coords(coords):
            return [tuple(round(c, precision) for c in pt) for pt in coords]
        # Manejar LineString y Polygon
        if 'coordinates' in geo_json:
            coords = geo_json['coordinates']
            if isinstance(coords[0][0], (float, int)):
                geo_json['coordinates'] = round_coords(coords)
            else:
                geo_json['coordinates'] = [round_coords(ring) for ring in coords]
        return shape(geo_json)

    gdf['geometry'] = gdf['geometry'].apply(_round_geom)
    return gdf

