# project_utils.py: unificación de geo_utils.py, text_utils.py y timer.py

import re
import time
import unicodedata
import logging
import pickle
import pandas as pd
from difflib import get_close_matches
import geopandas as gpd
from shapely.geometry import mapping, shape
from configs.settings import Pipeline

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

def find_best_match(target: str, candidates: list[str], cutoff: float = 0.6) -> str | None:
    """
    Devuelve el candidato más parecido a 'target' según difflib, o None.
    """
    norm_target = normalize_text(target)
    norm_map = { normalize_text(c): c for c in candidates if isinstance(c, str) }
    matches = get_close_matches(norm_target, norm_map.keys(), n=1, cutoff=cutoff)
    if matches:
        return norm_map[matches[0]]
    return None

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

# Wrapper de estrategias compatible con booleano y score

class StrategyWrapper_old:
    def __init__(self, func, buffer_key, threshold_key):
        self.func = func
        self.buffer_key = buffer_key
        self.threshold_key = threshold_key

    def match(self, g1, g2, thresholds):
        return self.func(
            g1, g2,
            thresholds[self.buffer_key],
            thresholds[self.threshold_key]
        )

    def score(self, g1, g2, thresholds):
        match, _ = self.match(g1, g2, thresholds)
        if not match:
            return 0.0
        if self.func.__name__ == 'compare_bibuffer_overlap':
            buf1 = g1.buffer(thresholds[self.buffer_key])
            buf2 = g2.buffer(thresholds[self.buffer_key])
            o1 = buf1.intersection(g2).length / g2.length if g2.length > 0 else 0
            o2 = buf2.intersection(g1).length / g1.length if g1.length > 0 else 0
            return max(o1, o2)
        elif self.func.__name__ == 'compare_similarity_index':
            buf1 = g1.buffer(thresholds[self.buffer_key])
            buf2 = g2.buffer(thresholds[self.buffer_key])
            inter_len = buf1.intersection(buf2).length
            total = g1.length + g2.length
            sim = (2 * inter_len) / total if total > 0 else 0
            return min(1.0, sim)
        else:
            return 0.0

### --- Funcion joblib para ayudar a encontrar geometrias homologas ---

def _process_group_joblib(group, strategy_map, strategies, id_col, year_col):
        
        """
        Función auxiliar para procesar un grupo de geometrías homólogas.
        Ejecutada en paralelo por joblib.
        """
        import uuid
        from shapely.ops import unary_union

        rows = group.reset_index(drop=True)
        used = [False] * len(rows)
        records = []

        for i, row in rows.iterrows():
            if used[i]:
                continue
            used[i] = True
            current = row.geometry
            group_idxs = [i]
            for j, other in rows.iterrows():
                if used[j]:
                    continue
                for strat in strategies:
                    match, strat_name = strategy_map[strat](current, other.geometry)
                    if match:
                        strat_used = strat_name
                        used[j] = True
                        group_idxs.append(j)
                        current = unary_union([current, other.geometry])
                        break
            # asignar blend_id y temporal_sources
            blend_id = str(uuid.uuid4())
            years = sorted(rows.loc[group_idxs, year_col].unique().tolist())
            for idx in group_idxs:
                rec = rows.loc[idx].drop('geometry').to_dict()
                rec.update({
                    'Strategy': strat_used,
                    'blend_id': blend_id,
                    'temporal_sources': years,
                    'geometry': rows.loc[idx].geometry
                })
                records.append(rec)
        return records


## Funcion para encontrar todas las columnas a pesar de que solo se haya dado la lista de los prefijos

def find_columns_given_preffixes(df, preffixes):
    """
    Encuentra todas las columnas que comienzan con los prefijos dados.
    """
    return [col for col in df.columns if any(col.startswith(p) for p in preffixes)]


def attribute_processing(df):
    return

def process_aadt_columns(df):
    # Extract range of years from Pipeline
    year_suffixes = [str(year) for year in Pipeline.YEARS_TO_ASSESS]

    exclude_aadt_measurements = []

    # Process each year's columns
    for year in year_suffixes:
        # Find all columns with current year suffix
        year_columns = [col for col in df.columns if col.endswith(f"_{year}")]

        # Key columns expected
        matars_col = f"Matarsperiod_{year}"
        matmetod_col = f"Matmetod_{year}"

        exclude_aadt_measurements.append(matars_col)
        exclude_aadt_measurements.append(matmetod_col)

        if matars_col in df.columns and matmetod_col in df.columns:
            # Extract first 4 digits from Matarsperiod (as year string)
            year_in_column = df[matars_col].astype(str).str[:4]

            # Check if year from Matarsperiod matches the column suffix year
            match_year = year_in_column == year

            # Check if Matmetod is "Stickprovsmätning"
            valid_method = df[matmetod_col] == "Stickprovsmätning"

            # Rows that satisfy both conditions
            valid_rows = match_year & valid_method

            # Set all columns with this year suffix to 0 where the row is invalid
            df.loc[~valid_rows, year_columns] = pd.NA

            # Keep only the year in Matarsperiod for valid rows
            df.loc[valid_rows, matars_col] = year_in_column[valid_rows].astype(int)
        else:
            # If required columns are missing, reset all columns for that year
            df[year_columns] = 0

    if not Pipeline.DELETE_DOUBLE_LINKS:
        aadt_columns = []
        for year in year_suffixes:
            aadt_columns.append(df.filter(regex=f'_{year}$', axis=1).columns.tolist())

        aadt_columns = [elemento for sublista in aadt_columns for elemento in sublista]
        aadt_measurement_columns = list(set(aadt_columns) - set(exclude_aadt_measurements))

        # Division del AADT por sentido
        df[aadt_measurement_columns] = df[aadt_measurement_columns]/2
    return df

def reindex_dataframes(nodes_all: pd.DataFrame,
                               links_all: pd.DataFrame,
                               start: int = 0):
    """
    Reindexa los DataFrames de nodos y enlaces:
      - nodes_all: renombra 'ID' → 'ID_old', crea 'ID' consecutivo.
      - links_all: renombra 'INODE' → 'INODE_old', 'JNODE' → 'JNODE_old', y crea las nuevas columnas.
    Devuelve:
      nodes_new, links_new, m_old2new, m_new2old
    donde m_old2new es dict {id_old: id_new} y m_new2old su inverso.
    
    Parámetros:
      start: valor inicial para el nuevo ID (por defecto 0; poner 1 si prefieres IDs desde 1).
    """
    # --- 1) Copiar para no mutar originales
    nodes = nodes_all.copy()
    links = links_all.copy()

    # --- 2) Crear mapeo old → new
    unique_ids = sorted(nodes['ID'].unique())
    m_old2new = {old: new for new, old in enumerate(unique_ids, start=start)}
    m_new2old = {new: old for old, new in m_old2new.items()}

    # --- 3) Reindexar nodes
    nodes = nodes.rename(columns={'ID': 'ID_old'})
    nodes['ID'] = nodes['ID_old'].map(m_old2new)
    nodes = nodes.set_index('ID', drop=True)

    # --- 4) Reindexar links
    links = links.rename(columns={'INODE': 'INODE_old', 'JNODE': 'JNODE_old'})
    links['INODE'] = links['INODE_old'].map(m_old2new)
    links['JNODE'] = links['JNODE_old'].map(m_old2new)
    links = links.reset_index(drop=True)

    # --- 5) Devolver resultados
    return nodes, links, m_old2new, m_new2old

import pickle
import os

class HandlePickle:

    def __init__(self):
        pass

    def save_pickle(self, route: str, filename: str, **variables):
        """
        Guarda múltiples variables nombradas en un archivo pickle.
        Se pasan como argumentos nombrados: save_pickle(x=var1, y=var2, route=..., filename=...)

        :param route: Ruta del archivo.
        :param filename: Nombre del archivo.
        :param variables: Variables a guardar (como nombre=valor).
        """
        full_path = os.path.join(route, f"{filename}.pkl")
        with open(full_path, 'wb') as f:
            pickle.dump(variables, f)

    def open_pickle(self, route: str, filename: str):
        """
        Carga variables desde un archivo pickle.

        :param route: Ruta del archivo.
        :param filename: Nombre del archivo.
        :return: Diccionario con variables recuperadas.
        """
        full_path = os.path.join(route, f"{filename}.pkl")
        with open(full_path, 'rb') as f:
            variables = pickle.load(f)
        return variables

