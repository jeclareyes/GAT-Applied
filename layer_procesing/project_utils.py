# project_utils.py: unificación de geo_utils.py, text_utils.py y timer.py

import re
import time
import unicodedata
import logging
from difflib import get_close_matches
import geopandas as gpd
from shapely.geometry import mapping, shape

logger = logging.getLogger(__name__)

# --- Decorador de tiempo ---
def timeit(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        logger.info(f"'{func.__name__}' ejecutado en {elapsed:.2f}s")
        return result
    return wrapper

# --- Utilidades de texto ---
def normalize_text(text):
    if not isinstance(text, str):
        return text
    text = unicodedata.normalize('NFD', text)
    text = ''.join(ch for ch in text if unicodedata.category(ch) != 'Mn')
    replacements = {'ä': 'a', 'Ä': 'A', 'ö': 'o', 'Ö': 'O', 'å': 'a', 'Å': 'A'}
    for orig, repl in replacements.items():
        text = text.replace(orig, repl)
    text = text.lower().strip()
    return re.sub(r'\s+', ' ', text)

def find_best_match(target, candidates, cutoff=0.6):
    if not isinstance(target, str) or not candidates.empty:
        return None
    norm_target = normalize_text(target)
    norm_map = {normalize_text(c): c for c in candidates if isinstance(c, str)}
    matches = get_close_matches(norm_target, norm_map.keys(), n=1, cutoff=cutoff)
    return norm_map[matches[0]] if matches else None

# --- Utilidades geográficas ---
def ensure_crs(gdf, target_crs):
    if gdf.crs != target_crs:
        try:
            return gdf.to_crs(target_crs)
        except Exception as e:
            logger.error(f"Error al reproyectar a {target_crs}: {e}")
    return gdf

def round_geometry_coords(gdf, precision=6):
    def _round_geom(geom):
        if geom is None:
            return None
        geo_json = mapping(geom)
        def round_coords(coords):
            return [tuple(round(c, precision) for c in pt) for pt in coords]
        if 'coordinates' in geo_json:
            coords = geo_json['coordinates']
            if isinstance(coords[0][0], (float, int)):
                geo_json['coordinates'] = round_coords(coords)
            else:
                geo_json['coordinates'] = [round_coords(ring) for ring in coords]
        return shape(geo_json)
    gdf['geometry'] = gdf['geometry'].apply(_round_geom)
    return gdf
