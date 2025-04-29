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

### --- Funciones de comparación de geometrías ---
from shapely.geometry.base import BaseGeometry
from shapely.geometry import LineString, MultiLineString
from shapely.ops import unary_union, linemerge

def compare_exact(geom1: BaseGeometry, geom2: BaseGeometry) -> tuple[bool, str | None]:
    if geom1.equals(geom2):
        return True, 'exact'
    return False, None


def compare_distance(geom1: BaseGeometry, geom2: BaseGeometry, threshold: float) -> tuple[bool, str | None]:
    if geom1.distance(geom2) <= threshold:
        return True, 'distance'
    return False, None


def compare_diff(geom1: BaseGeometry, geom2: BaseGeometry, threshold: float) -> tuple[bool, str | None]:
    if geom1.geom_type in ['LineString', 'MultiLineString'] and geom2.geom_type in ['LineString', 'MultiLineString']:
        if geom1.symmetric_difference(geom2).length <= threshold:
            return True, 'diff'
    return False, None


def compare_overlap_ratio(geom1: BaseGeometry, geom2: BaseGeometry, threshold: float) -> tuple[bool, str | None]:
    if geom1.geom_type in ['LineString', 'MultiLineString'] and geom2.geom_type in ['LineString', 'MultiLineString']:
        inter = geom1.intersection(geom2)
        min_len = min(geom1.length, geom2.length)
        if min_len > 0 and inter.length / min_len >= threshold:
            return True, 'overlap_ratio'
    return False, None


def compare_buffer_overlap(geom1: BaseGeometry, geom2: BaseGeometry, buffer_amount: float) -> tuple[bool, str | None]:
    buf = geom1.buffer(buffer_amount)
    if buf.intersects(geom2):
        inter_len = buf.intersection(geom2).length
        if geom2.length > 0 and inter_len / geom2.length >= 0.8:
            return True, 'buffer_overlap'
    return False, None


def compare_bibuffer_overlap(geom1: BaseGeometry, geom2: BaseGeometry, buffer_amount: float, threshold: float = 0.8) -> tuple[bool, str | None]:
    buf1 = geom1.buffer(buffer_amount)
    buf2 = geom2.buffer(buffer_amount)
    overlap1 = buf1.intersection(geom2).length / geom2.length if geom2.length > 0 else 0
    overlap2 = buf2.intersection(geom1).length / geom1.length if geom1.length > 0 else 0
    if max(overlap1, overlap2) >= threshold:
        return True, 'bibuffer_overlap'
    return False, None


def compare_similarity_index(geom1: BaseGeometry, geom2: BaseGeometry, buffer_amount: float = 50.0, threshold: float = 0.7) -> tuple[bool, str | None]:
    buf1 = geom1.buffer(buffer_amount)
    buf2 = geom2.buffer(buffer_amount)
    inter_len = buf1.intersection(buf2).length
    total = geom1.length + geom2.length
    similarity = (2*inter_len)/total if total>0 else 0
    similarity = min(1.0, similarity)
    if similarity >= threshold:
        return True, 'similarity_index'
    return False, None





