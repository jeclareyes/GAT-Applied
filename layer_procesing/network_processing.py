# geometry_processing.py: contiene las clases GeometryCleaner y LayerMerger

import geopandas as gpd
import numpy as np
import pandas as pd
import logging
import uuid
import concurrent.futures
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from functools import partial
from joblib import Parallel, delayed
from shapely.geometry import base, LineString, Point, MultiLineString
from shapely.geometry.base import BaseMultipartGeometry, BaseGeometry
from scipy.spatial import cKDTree
from shapely.ops import unary_union, linemerge
from shapely.errors import TopologicalError
from project_utils import (
    compare_exact, compare_distance, compare_diff,
    compare_overlap_ratio, compare_buffer_overlap,
    compare_bibuffer_overlap, compare_similarity_index, StrategyWrapper_old, _process_group_joblib,
    find_columns_given_preffixes
)
from configs.settings import Fields, Layer

logger = logging.getLogger(__name__)


class GeometryCleaner:
    """
    Limpieza y fusión intra-layer utilizando estrategias de comparación.
    """
    def __init__(self, thresholds=None):
        # Umbrales por defecto
        self.thresholds = thresholds or {
            'distance': 5.0,
            'diff': 5.0,
            'overlap_ratio': 0.5,
            'buffer_overlap': 50.0,
            'bibuffer_overlap': 50.0,
            'bibuffer_overlap_threshold': 0.8,
            'similarity_index_buffer': 50.0,
            'similarity_index_threshold': 0.7
        }

    def clean(self, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Limpia un GeoDataFrame aplicando validaciones y correcciones básicas.

        Elimina geometrías nulas o vacías y corrige geometrías inválidas
        utilizando el método buffer(0).

        Parámetros:
        - gdf: (GeoDataFrame) El GeoDataFrame a limpiar.

        Retorna:
        - Un nuevo GeoDataFrame con geometrías válidas y no nulas/vacías.
          Las filas con geometrías no recuperables son eliminadas.
        """
        if not isinstance(gdf, gpd.GeoDataFrame):
            logger.error("La entrada a clean no es un GeoDataFrame.")
            return gdf  # Retornar el original si el tipo es incorrecto

        initial_rows = len(gdf)
        to_drop = []  # Lista para registrar los índices con geometrías no recuperables
        cleaned_geometries = []  # Lista para almacenar las geometrías limpias

        # Recorremos cada fila del GeoDataFrame para validar y corregir su geometría
        for idx, row in gdf.iterrows():
            geom = row.geometry

            # Si la geometría es nula o vacía, la marcamos para eliminación
            if geom is None or geom.is_empty:
                logger.error(f"Geometría nula/vacía en índice {idx}")
                to_drop.append(idx)
                cleaned_geometries.append(None)  # Usamos None como placeholder
                continue

            # Si la geometría es inválida, intentamos corregirla con buffer(0)
            if not geom.is_valid:
                fixed = geom.buffer(0)
                if fixed.is_valid:
                    cleaned_geometries.append(fixed)
                    logger.warning(
                        f"Índice {idx}: geometría corregida con buffer(0)"
                    )
                else:
                    logger.error(
                        f"Índice {idx}: geometría inválida no pudo corregirse"
                    )
                    to_drop.append(idx)
                    cleaned_geometries.append(None)  # Usamos None como placeholder
            else:
                cleaned_geometries.append(geom)  # La geometría ya es válida

        # Actualizar las geometrías corregidas en el GeoDataFrame
        # Usamos .loc para evitar SettingWithCopyWarning
        gdf_cleaned = gdf.copy()  # Trabajar sobre una copia para seguridad
        gdf_cleaned['geometry'] = cleaned_geometries

        # Eliminamos las filas con geometrías problemáticas que no se pudieron recuperar
        if to_drop:
            # Filtrar las filas a mantener
            valid_indices = [i for i in range(initial_rows) if i not in to_drop]
            gdf_cleaned = gdf_cleaned.iloc[valid_indices].reset_index(drop=True)
            logger.info(f"Eliminados {len(to_drop)} registros inválidos o no recuperables")

        logger.info(
            f"Proceso de limpieza completado. Filas iniciales: {initial_rows}, Filas finales: {len(gdf_cleaned)}")
        return gdf_cleaned  # Retornamos el GeoDataFrame limpio

    def find_homologs(self, gdf: gpd.GeoDataFrame, id_col='Avsnittsidentitet',
                      year_col='year', strategies=None) -> gpd.GeoDataFrame:
        """
        Identifica grupos de geometrías homólogas según las estrategias sin fusionar geometría.
        Devuelve un GeoDataFrame con campos extra:
          - 'blend_id': UUID por grupo
          - 'temporal_sources': lista de años involucrados
        """
        strategies = strategies or ['exact', 'distance', 'diff', 'overlap_ratio',
                                     'buffer_overlap', 'bibuffer_overlap', 'similarity_index']
        strategy_map = self._get_strategy_map()

        if year_col not in gdf.columns:
            raise ValueError(f"GeoDataFrame debe contener '{year_col}' para trazabilidad.")

        records = []
        for eid, group in gdf.groupby(id_col):
            rows = group.reset_index(drop=True)
            used = [False] * len(rows)
            for i, row in rows.iterrows():
                if used[i]: continue
                used[i] = True
                current = row.geometry
                group_idxs = [i]
                for j, other in rows.iterrows():
                    if used[j]: continue
                    for strat in strategies:
                        match, strat_name = strategy_map[strat](current, other.geometry)
                        if match:
                            strat_used = strat_name
                            used[j] = True
                            group_idxs.append(j)
                            # extiende current para la detección consecutiva
                            current = unary_union([current, other.geometry])
                            break
                # asignar blend_id y temporal_sources
                blend_id = str(uuid.uuid4())
                years = sorted(rows.loc[group_idxs, year_col].unique().tolist())
                for idx in group_idxs:
                    rec = rows.loc[idx].drop('geometry').to_dict()
                    rec.update({'Strategy': strat_used,
                                'blend_id': blend_id,
                                'temporal_sources': years,
                                'geometry': rows.loc[idx].geometry})
                    records.append(rec)
        return gpd.GeoDataFrame(records, crs=gdf.crs)

    def blend_geometries(self, homologs_gdf: gpd.GeoDataFrame, columns_to_aggregate_values: list, output_final=True) -> gpd.GeoDataFrame:
        """
        Fusiona las geometrías de cada grupo identificado (
        'blend_id') en una geometría única por grupo.
        Si se especifica output_final, exporta la capa consolidada.
        """
        columns_to_aggregate_values = find_columns_given_preffixes(
            homologs_gdf, columns_to_aggregate_values)

        merged = []
        for bid, group in homologs_gdf.groupby('blend_id'):
            unioned = unary_union(group.geometry.tolist())
            parts = self._decompose_and_linemerge(unioned)
            
            base_attr = {}
            for col in columns_to_aggregate_values:
                col_values = group[col]
                first_valid = col_values.dropna().iloc[0] if not col_values.dropna().empty else None
                base_attr[col] = first_valid 

            other_cols = [c for c in group.columns if c not in columns_to_aggregate_values + ['geometry']]

            for col in other_cols:
                base_attr[col] = group.iloc[0][col]            
            
            for geom in parts:
                rec = base_attr.copy()
                rec['geometry'] = geom
                merged.append(rec)
        return gpd.GeoDataFrame(merged, crs=homologs_gdf.crs)
    
    def _get_strategy_map(self) -> dict:
        """Construye el mapeo de estrategias a funciones."""
        return {
            'exact': compare_exact,
            'distance': lambda g1, g2: compare_distance(g1, g2, self.thresholds['distance']),
            'diff': lambda g1, g2: compare_diff(g1, g2, self.thresholds['diff']),
            'overlap_ratio': lambda g1, g2: compare_overlap_ratio(g1, g2, self.thresholds['overlap_ratio']),
            'buffer_overlap': lambda g1, g2: compare_buffer_overlap(g1, g2, self.thresholds['buffer_overlap']),
            'bibuffer_overlap': lambda g1, g2: compare_bibuffer_overlap(
                g1, g2, self.thresholds['bibuffer_overlap'], self.thresholds['bibuffer_overlap_threshold']
            ),
            'similarity_index': lambda g1, g2: compare_similarity_index(
                g1, g2, self.thresholds['similarity_index_buffer'], self.thresholds['similarity_index_threshold']
            )
        }
    
    def _decompose_and_linemerge(self, geom: BaseGeometry) -> list[BaseGeometry]:
        """
        Descompone y fusiona geometrías multi-LineString en partes simples.
        """
        geometries = []
        if isinstance(geom, MultiLineString):
            try:
                merged = linemerge(geom)
                if isinstance(merged, MultiLineString):
                    geometries.extend(list(merged.geoms))
                else:
                    geometries.append(merged)
            except Exception:
                geometries.extend(list(geom.geoms))
        else:
            geometries.append(geom)
        return [g for g in geometries if g and not g.is_empty]

    #  Haciendo pruebas con joblib

    def find_homologs_joblib(self, gdf: gpd.GeoDataFrame, id_col='Avsnittsidentitet',
                         year_col='year', strategies=None, n_jobs=-1) -> gpd.GeoDataFrame:
        """
        Versión paralela usando joblib para identificar geometrías homólogas.
        """
        strategies = strategies or ['exact', 'distance', 'diff', 'overlap_ratio',
                                    'buffer_overlap', 'bibuffer_overlap', 'similarity_index']
        strategy_map = self._get_strategy_map_joblib()

        if year_col not in gdf.columns:
            raise ValueError(f"GeoDataFrame debe contener '{year_col}' para trazabilidad.")

        # Agrupar por id_col
        groups = [group for _, group in gdf.groupby(id_col)]

        # Ejecutar en paralelo
        results = Parallel(n_jobs=n_jobs)(
            delayed(_process_group_joblib)(group, strategy_map, strategies, id_col, year_col)
            for group in groups
        )

        # Aplanar resultados
        records = [rec for group_recs in results for rec in group_recs]

        return gpd.GeoDataFrame(records, crs=gdf.crs)

    def _get_strategy_map_joblib(self) -> dict:
        """Construye el mapeo de estrategias con funciones picklables para joblib."""
        return {
            'exact': compare_exact,
            'distance': partial(compare_distance, threshold=self.thresholds['distance']),
            'diff': partial(compare_diff, threshold=self.thresholds['diff']),
            'overlap_ratio': partial(compare_overlap_ratio, threshold=self.thresholds['overlap_ratio']),
            'buffer_overlap': partial(compare_buffer_overlap, buffer_amount=self.thresholds['buffer_overlap']),
            'bibuffer_overlap': partial(compare_bibuffer_overlap,
                                        buffer_amount=self.thresholds['bibuffer_overlap'],
                                        threshold=self.thresholds['bibuffer_overlap_threshold']),
            'similarity_index': partial(compare_similarity_index,
                                        buffer_amount=self.thresholds['similarity_index_buffer'],
                                        threshold=self.thresholds['similarity_index_threshold']),
        }



class LayerMerger:
    """
    Empareja dos layers A y B y transfiere atributos de B a A.
    """
    def __init__(self, thresholds: dict = None, strategies: list[str] = None):
        # Umbrales por defecto
        self.thresholds = thresholds or {
            'bibuffer_overlap': 50.0,
            'bibuffer_overlap_threshold': 0.7,
            'similarity_index_buffer': 50.0,
            'similarity_index_threshold': 0.7,
            'vicinity_search': 500.0
        }
        self.strategies = strategies or ['bibuffer_overlap', 'similarity_index']

        # Mapear estrategias con wrappers
        self.strategy_wrappers = {
            'bibuffer_overlap': StrategyWrapper_old(compare_bibuffer_overlap, 'bibuffer_overlap', 'bibuffer_overlap_threshold'),
            'similarity_index': StrategyWrapper_old(compare_similarity_index, 'similarity_index_buffer', 'similarity_index_threshold')
        }

    def _match_with_strategies(self, line_a, line_b) -> tuple[bool,str|None]:
        for strat in self.strategies:
            match, name = self.strategy_wrappers[strat].match(line_a, line_b, self.thresholds)
            if match:
                return True, name
        return False, None

    def _compute_match_score(self, geom1, geom2) -> float:
        scores = [
            self.strategy_wrappers[strat].score(geom1, geom2, self.thresholds)
            for strat in self.strategies
        ]
        return sum(scores) / len(scores) if scores else 0.0

    def _find_best_match_idx(self, row_a: pd.Series, gdf_b: gpd.GeoDataFrame) -> tuple[int|None, float]:
        buf = row_a.geometry.buffer(self.thresholds['vicinity_search'])
        buf_gdf = gpd.GeoDataFrame(geometry=[buf], crs=gdf_b.crs)
        candidates = (
            gpd.sjoin(gdf_b, buf_gdf, how='inner', predicate='intersects')
            .drop(columns=['index_right'])
        )

        best_idx = None
        best_score = -1

        for idx_b, row_b in candidates.iterrows():
            score = self._compute_match_score(row_a.geometry, row_b.geometry)
            if score > best_score:
                best_idx = idx_b
                best_score = score

        return best_idx, best_score if best_score >= 0 else None

    def merge_layers(self, gdf_a: gpd.GeoDataFrame, gdf_b: gpd.GeoDataFrame,
                 attrs_to_transfer: list[str] = None, prefix: str = 'emme_') -> gpd.GeoDataFrame:
        # Renombrar columnas de gdf_a con el prefijo, excepto 'geometry'
        renamed_columns = {
            col: prefix + col for col in gdf_a.columns if col not in ['geometry', 'INODE', 'JNODE', 'LENGTH', 'LOG_DIRECTION']
        }
        result = gdf_a.rename(columns=renamed_columns).copy()

        attrs = attrs_to_transfer or [c for c in gdf_b.columns if c != 'geometry']

        match_idxs = []
        match_scores = []

        for _, row_a in gdf_a.iterrows():  # ¡OJO! Usamos el original para buscar coincidencias
            idx_b, score = self._find_best_match_idx(row_a, gdf_b)
            match_idxs.append(idx_b)
            match_scores.append(score if score is not None else 0)

        for attr in attrs:
            result[attr] = [
                gdf_b.at[idx, attr] if idx is not None else None
                for idx in match_idxs
            ]

        result['match_score'] = match_scores

        result = self.merge_multilines(result)
        return result
    
    def merge_layers_parallel(self, gdf_a: gpd.GeoDataFrame, gdf_b: gpd.GeoDataFrame,
                 attrs_to_transfer: list[str] = None, prefix: str = 'emme_') -> gpd.GeoDataFrame:

        # Renombrar columnas de gdf_a con el prefijo, excepto 'geometry'
        renamed_columns = {
            col: prefix + col for col in gdf_a.columns if col != 'geometry'
        }
        result = gdf_a.rename(columns=renamed_columns).copy()

        attrs = attrs_to_transfer or [c for c in gdf_b.columns if c != 'geometry']

        def find_match(row):
            return self._find_best_match_idx(row, gdf_b)

        with ThreadPoolExecutor() as executor:
            futures = list(executor.map(find_match, [row for _, row in gdf_a.iterrows()]))

        match_idxs, match_scores = zip(*[
            (idx, score if score is not None else 0)
            for idx, score in futures
        ])

        for attr in attrs:
            result[attr] = [
                gdf_b.at[idx, attr] if idx is not None else None
                for idx in match_idxs
            ]

        result['match_score'] = match_scores

        result = self.merge_multilines(result)
        return result
    
    def merge_multilines(self, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Procesa un GeoDataFrame, intentando fusionar cada MultiLineString en un único LineString.
        Lanza un warning si alguna geometría no se puede fusionar completamente.

        Args:
            gdf (gpd.GeoDataFrame): GeoDataFrame con geometrías LineString o MultiLineString.

        Returns:
            gpd.GeoDataFrame: Nuevo GeoDataFrame con geometrías procesadas.
        """
        new_geometries = []

        for idx, geom in enumerate(gdf.geometry):
            if isinstance(geom, MultiLineString):
                merged = linemerge(geom)
                if isinstance(merged, MultiLineString):
                    logger.warning(
                        f"Advertencia: la geometría en el índice {idx} no se pudo fusionar completamente en un LineString."
                    )
                new_geometries.append(merged)
            else:
                new_geometries.append(geom)

        gdf_processed = gdf.copy()
        gdf_processed.geometry = new_geometries
        return gdf_processed


class AttributeManager:
    """
    Manages attributes of a GeoDataFrame, particularly handling temporal fields
    with year suffixes.
    """
    def __init__(self, temporal_fields=None, drop_fields=None):
        """
        Initializes the AttributeManager.

        Args:
            temporal_fields (list, optional): List of base names for temporal fields.
                                              Defaults to Fields.TEMPORAL_FIELDS.
            drop_fields (list, optional): List of fields to potentially drop later.
                                          Defaults to Fields.DROP_FIELDS_LASTKAJEN.
        """
        # Use provided lists or defaults from Fields class
        self.temporal_fields = temporal_fields if temporal_fields is not None else Fields.TEMPORAL_FIELDS
        self.drop_fields = drop_fields if drop_fields is not None else Fields.DROP_FIELDS_LASTKAJEN

    def consolidate(self, gdf, year):
        """
        Renames temporal fields by adding a year suffix. Handles potential mismatches.

        Args:
            gdf (gpd.GeoDataFrame): The input GeoDataFrame.
            year (int or str): The year to append as a suffix.

        Returns:
            gpd.GeoDataFrame: GeoDataFrame with temporal fields renamed.
        """
        gdf = gdf.copy()
        suffix = f"_{year}"
        # Determine fields to process: temporal fields excluding those marked for dropping
        fields_to_process = list(set(self.temporal_fields) - set(self.drop_fields))

        # Keep track of columns that have been renamed to avoid renaming multiple matches
        renamed_columns = set()

        for field in fields_to_process:
            # Check if the exact field name exists and hasn't been renamed yet
            if field in gdf.columns and field not in renamed_columns:
                new_name = field + suffix
                gdf.rename(columns={field: new_name}, inplace=True)
                renamed_columns.add(new_name)
                #  logger.info(f"Renamed '{field}' to '{new_name}' for year {year}")
            else:
                # If exact field not found or already processed, try finding a match
                match = find_best_match(field, [col for col in gdf.columns if col not in renamed_columns])
                if match:
                    new_name = match + suffix
                    gdf.rename(columns={match: new_name}, inplace=True)
                    renamed_columns.add(new_name)
                    logger.info(f"Found match '{match}' for '{field}' and renamed to '{new_name}' for year {year}")
                # Only log warning if the base field itself wasn't found AND no match was found
                elif field not in gdf.columns:
                     logger.warning(f"Campo base '{field}' ni ninguna coincidencia encontrada para el año {year}")

        gdf = self.reorder_attributes_by_year(gdf)
        
        return gdf

    def delete_attributes(self, gdf, attributes_to_delete):
        """
        Deletes specified attributes (columns) from the GeoDataFrame.
        It safely ignores attributes that are not found in the DataFrame.

        Args:
            gdf (gpd.GeoDataFrame): The GeoDataFrame to modify.
            attributes_to_delete (list): A list of attribute names (strings) to delete.

        Returns:
            gpd.GeoDataFrame: The GeoDataFrame with specified attributes removed.
        """
        # Make a copy to avoid modifying the original DataFrame unexpectedly
        gdf_copy = gdf.copy()
        # Find which of the attributes to delete actually exist in the DataFrame
        existing_attributes_to_delete = [attr for attr in attributes_to_delete if attr in gdf_copy.columns]

        if not existing_attributes_to_delete:
            logger.warning("Ninguno de los atributos especificados para eliminar existe en el GeoDataFrame.")
            return gdf_copy # Return the copy without changes

        # Drop the existing attributes
        try:
            gdf_copy.drop(columns=existing_attributes_to_delete, inplace=True)
            logger.info(f"Atributos eliminados: {existing_attributes_to_delete}")
        except Exception as e:
            logger.error(f"Error al eliminar atributos: {e}")
            # Optionally re-raise the exception or handle it as needed
            # raise e

        return gdf_copy

    def reorder_attributes_by_year(self, gdf):
        """
        Reorders the columns of the GeoDataFrame. It groups columns first by their
        base name (preserving the original relative order of base names) and then
        sorts them by year within each group. Non-temporal columns are placed first.

        Example: a_2020, b_2020, a_2021, b_2021 -> a_2020, a_2021, b_2020, b_2021

        Args:
            gdf (gpd.GeoDataFrame): The GeoDataFrame to reorder.

        Returns:
            gpd.GeoDataFrame: The GeoDataFrame with reordered columns.
        """
        gdf_copy = gdf.copy()
        original_columns = list(gdf_copy.columns)

        temporal_cols_by_base = OrderedDict() # Group by base name {base: [{'year': YYYY, 'full': name}, ...]}
        non_temporal_cols = [] # List for columns without year suffix
        base_name_order = [] # To keep track of the original order of base names

        # Regex to find columns ending with _YYYY (e.g., _2020)
        year_suffix_pattern = re.compile(r"_(\d{4})$")

        # --- Step 1: Classify columns and group temporal ones by base name ---
        for col in original_columns:
            match = year_suffix_pattern.search(col)
            if match:
                year = int(match.group(1)) # Extract year
                base_name = col[:match.start()] # Extract base name

                # If this is the first time seeing this base name, add it to the order list
                if base_name not in temporal_cols_by_base:
                    temporal_cols_by_base[base_name] = []
                    base_name_order.append(base_name) # Maintain original appearance order

                temporal_cols_by_base[base_name].append({'year': year, 'full': col})
            else:
                # Keep non-temporal columns (like 'geometry', 'id', etc.) separate
                non_temporal_cols.append(col)

        # --- Step 2: Sort non-temporal columns (optional, keeps them consistent) ---
        # Keep 'geometry' at the end if it exists
        if 'geometry' in non_temporal_cols:
             non_temporal_cols.remove('geometry')
             non_temporal_cols.sort() # Sort the rest alphabetically
             non_temporal_cols.append('geometry') # Append geometry at the end
        else:
             non_temporal_cols.sort() # Sort all alphabetically if no geometry

        # --- Step 3: Build the final ordered list ---
        ordered_columns = list(non_temporal_cols) # Start with non-temporal

        # Iterate through base names in their original appearance order
        for base_name in base_name_order:
            # Get all columns for this base name
            cols_for_base = temporal_cols_by_base[base_name]
            # Sort these columns by year
            sorted_cols_for_base = sorted(cols_for_base, key=lambda x: x['year'])
            # Append the full names of the sorted columns to the final list
            ordered_columns.extend([item['full'] for item in sorted_cols_for_base])

        # --- Step 4: Reindex the DataFrame ---
        # Check if the ordered list contains all original columns (sanity check)
        if set(ordered_columns) != set(original_columns):
             logger.warning("Discrepancia en las columnas durante la reordenación. Verifique la lógica.")
             # Fallback to original order or handle error appropriately
             return gdf_copy # Return original copy in case of error

        return gdf_copy[ordered_columns]

class NodeIdentifier:
    """
    Identifica nodos de inicio y fin en geometrías LineString de un GeoDataFrame.
    Clasifica los nodos como 'Intersección' o 'Nodo_final' y detecta segmentos huérfanos.
    """
    
    def __init__(self, precision=1):
        """
        Inicializa el objeto con la precisión decimal para redondeo de coordenadas.
        
        Args:
            precision (int): Cantidad de decimales a considerar en las coordenadas.
        """
        self.precision = precision

    def _explode_multilines(self, gdf):
        """
        Convierte MultiLineStrings en LineStrings individuales.
        """
        exploded = gdf.explode(index_parts=False)
        exploded = exploded[exploded.geometry.type == 'LineString']
        return exploded.reset_index(drop=True)

    def identify(self, gdf, auto_explode_multilines=False):
        if auto_explode_multilines:
            gdf = self._explode_multilines(gdf)

        coord_info = {}
        seg_map = {}

        for idx, row in gdf.iterrows():
            coords = list(row.geometry.coords)
            eid = row.get('ELEMENT_ID', idx)
            start = tuple(round(c, self.precision) for c in coords[0])
            end = tuple(round(c, self.precision) for c in coords[-1])

            for key in (start, end):
                coord_info.setdefault(key, {'count': 0, 'elements': []})
                coord_info[key]['count'] += 1
                coord_info[key]['elements'].append(eid)

            seg_map[idx] = (start, end)

        nodes = []
        for i, (coord, info) in enumerate(coord_info.items()):
            nodes.append({
                'ID': int(i),
                'X': coord[0],
                'Y': coord[1],
                'node_type': 'Intersection',
                'links_Top': info['elements'],
                'geometry': Point(coord)
            })

        gdf_nodes = gpd.GeoDataFrame(nodes, crs=gdf.crs)

        orphans = [
            idx for idx, (s, e) in seg_map.items()
            if coord_info[s]['count'] == 1 and coord_info[e]['count'] == 1
        ]
        orphan_segs = gdf.loc[orphans].copy() if orphans else None

        return gdf_nodes, orphan_segs

    def assign_border_nodes(self, points_gdf, polygon_gdf, tolerance=1.0):
        """
        Asigna el tipo 'Nodo_final' a puntos que tocan el borde del polígono dentro de una tolerancia.

        Args:
            points_gdf (GeoDataFrame): GeoDataFrame con geometrías tipo Point y columna 'node_type'.
            polygon_gdf (GeoDataFrame): GeoDataFrame con un único polígono.
            tolerance (float): Distancia de tolerancia para considerar que un punto está sobre el borde.

        Returns:
            GeoDataFrame: points_gdf con 'node_type' actualizado.
        """
        polygon = polygon_gdf.geometry.iloc[0]
        boundary_buffer = polygon.boundary.buffer(tolerance)

        points_gdf['node_type'] = points_gdf.apply(
            lambda row: 'Border_Node' if boundary_buffer.contains(row.geometry) else row['node_type'],
            axis=1
        )
        return points_gdf


    
    def remove_orphans(self, gdf, orphan_segs=None):
        """
        Elimina segmentos huérfanos de un GeoDataFrame.
        
        Puede usar huérfanos preidentificados (más eficiente) o calcularlos si no se proveen.

        Args:
            gdf (GeoDataFrame): Contiene geometrías tipo LineString.
            orphan_segs (GeoDataFrame or None): Segmentos huérfanos ya identificados (opcional).

        Returns:
            GeoDataFrame: GeoDataFrame sin los segmentos huérfanos.
        """
        if orphan_segs is None:
            _, orphan_segs = self.identify(gdf, auto_explode_multilines=False)
        
        if orphan_segs is not None:
            return gdf.drop(index=orphan_segs.index)
        else:
            return gdf.copy()
        
    def heredar_IDs(self, gdf_nodes, gdf_ref, max_dist=5):
        """
        Asigna ID único a nodos basándose en cercanía a puntos de referencia.
        Los nodos sin coincidencias reciben un ID entero nuevo, comenzando en 75000,
        evitando repeticiones con los ya asignados.

        Args:
            gdf_nodes (GeoDataFrame): Nodos generados por identify().
            gdf_ref (GeoDataFrame): Capa de puntos con campo 'ID'.
            max_dist (float): Distancia máxima en unidades del CRS para considerar una coincidencia.

        Returns:
            GeoDataFrame: gdf_nodes con columna 'ID' heredada o generada.
        """
        import pandas as pd
        import geopandas as gpd

        # Asegurar CRS consistente
        gdf_ref = gdf_ref.to_crs(gdf_nodes.crs)

        # Renombrar 'ID' si ya existe
        if 'ID' in gdf_nodes.columns:
            gdf_nodes = gdf_nodes.rename(columns={'ID': 'ID_original'})

        # Unión espacial por cercanía
        joined = gpd.sjoin_nearest(
            gdf_nodes,
            gdf_ref[['geometry', 'ID']],
            how='left',
            max_distance=max_dist,
            distance_col='dist'
        )

        # Ordenar y eliminar duplicados por ID (tomar el más cercano)
        joined = joined.sort_values(by='dist')
        joined = joined.drop_duplicates(subset='ID', keep='first')

        # Combinar con nodos originales
        gdf_result = gdf_nodes.copy()
        gdf_result = gdf_result.merge(joined[['geometry', 'ID']], on='geometry', how='left')

        # IDs ya usados (convertidos a enteros si posible)
        ids_existentes = set(pd.to_numeric(gdf_result['ID'], errors='coerce').dropna().astype(int))

        # Generar nuevos IDs enteros para NaN, comenzando en 75000
        na_mask = gdf_result['ID'].isna()
        nuevos_ids = []
        id_actual = 75000
        while len(nuevos_ids) < na_mask.sum():
            if id_actual not in ids_existentes:
                nuevos_ids.append(id_actual)
            id_actual += 1

        # Asignar nuevos IDs
        gdf_result.loc[na_mask, 'ID'] = nuevos_ids
        gdf_result['ID'] = gdf_result['ID'].astype(int)
        gdf_result = gdf_result.drop(columns=['ID_original'], errors='ignore')

        return gdf_result

def control_topology(links_gdf, nodes_gdf, search_radius=5.0):
    if links_gdf.crs != nodes_gdf.crs:
        print(f"Advirtiendo: CRS de enlaces ({links_gdf.crs}) y nodos ({nodes_gdf.crs}) son diferentes. Convirtiendo nodos al CRS de enlaces.")
        nodes = nodes_gdf.to_crs(links_gdf.crs)
    else:
        nodes = nodes_gdf.copy()

    if nodes.empty:
        print("Advertencia: El GeoDataFrame de nodos está vacío.")
        result = links_gdf.copy()
        result['INODE'] = None
        result['JNODE'] = None
        return result

    inodes = []
    jnodes = []

    for line in links_gdf.geometry:
        start_pt = Point(line.coords[0])
        end_pt = Point(line.coords[-1])

        # INODE
        nearby_start = nodes[nodes.geometry.distance(start_pt) <= search_radius]
        if not nearby_start.empty:
            inode_id = nearby_start.loc[nearby_start.geometry.distance(start_pt).idxmin(), 'ID']
            inodes.append(inode_id)
        else:
            print(f"No INODE para línea: {line.wkt}")
            inodes.append(None)

        # JNODE
        nearby_end = nodes[nodes.geometry.distance(end_pt) <= search_radius]
        if not nearby_end.empty:
            jnode_id = nearby_end.loc[nearby_end.geometry.distance(end_pt).idxmin(), 'ID']
            jnodes.append(jnode_id)
        else:
            print(f"No JNODE para línea: {line.wkt}")
            jnodes.append(None)

    result = links_gdf.copy()
    result['INODE'] = inodes
    result['JNODE'] = jnodes
    return result


### -------- INTENTO DE PARELIZACION


def compare_bibuffer_overlap_score(geom1, geom2, buffer_dist, overlap_ratio_threshold) -> float:
    """
    Calcula una puntuación basada en la superposición de buffers.
    Devuelve una puntuación entre 0.0 y 1.0+ (si la intersección es mayor que el buffer más pequeño),
    o 0.0 si no cumple el umbral o hay error.
    """
    if geom1 is None or geom2 is None or geom1.is_empty or geom2.is_empty:
        return 0.0
    if not (isinstance(buffer_dist, (int, float)) and buffer_dist >= 0 and
            isinstance(overlap_ratio_threshold, (int, float)) and 0 <= overlap_ratio_threshold <= 1):
        logger.debug(f"Valores de umbral inválidos para bibuffer: dist={buffer_dist}, thresh={overlap_ratio_threshold}")
        return 0.0
    try:
        # Considera validar/limpiar geometrías aquí si es necesario, ej: geom1.buffer(0)
        # pero puede ser costoso hacerlo repetidamente. Mejor limpiar datos de antemano.

        buf1 = geom1.buffer(buffer_dist)
        buf2 = geom2.buffer(buffer_dist)

        if not buf1.is_valid or not buf2.is_valid or buf1.is_empty or buf2.is_empty:
            return 0.0
        
        intersection = buf1.intersection(buf2)
        if intersection.is_empty or intersection.area < 1e-9: # Ignorar intersecciones minúsculas
            return 0.0

        min_buffer_area = min(buf1.area, buf2.area)
        if min_buffer_area < 1e-9: # Evitar división por cero si los buffers son puntos/líneas
             # Si ambos son líneas y se superponen perfectamente, el área del buffer podría ser pequeña.
             # Considera una métrica diferente para líneas si el área no es representativa.
            return 0.0 

        score = intersection.area / min_buffer_area
        return score if score >= overlap_ratio_threshold else 0.0
    except TopologicalError:
        logger.debug(f"Error topológico en compare_bibuffer_overlap_score.", exc_info=False)
        return 0.0
    except Exception as e:
        logger.debug(f"Excepción en compare_bibuffer_overlap_score: {e}", exc_info=False)
        return 0.0

def compare_similarity_index_score(geom1, geom2, buffer_dist, similarity_index_threshold) -> float:
    """
    Calcula una puntuación basada en un índice de similitud (ej. Jaccard sobre buffers).
    Devuelve una puntuación entre 0.0 y 1.0.
    """
    if geom1 is None or geom2 is None or geom1.is_empty or geom2.is_empty:
        return 0.0
    if not (isinstance(buffer_dist, (int, float)) and buffer_dist >= 0 and
            isinstance(similarity_index_threshold, (int, float)) and 0 <= similarity_index_threshold <= 1):
        logger.debug(f"Valores de umbral inválidos para similarity_index: dist={buffer_dist}, thresh={similarity_index_threshold}")
        return 0.0
    try:
        buf1 = geom1.buffer(buffer_dist)
        buf2 = geom2.buffer(buffer_dist)

        if not buf1.is_valid or not buf2.is_valid or buf1.is_empty or buf2.is_empty:
            return 0.0

        intersection_area = buf1.intersection(buf2).area
        if intersection_area < 1e-9 and not (buf1.geom_type == 'Point' and buf2.geom_type == 'Point' and buf1.equals(buf2)): # Permitir puntos idénticos
             pass # Continuar para calcular union_area, el Jaccard será 0

        union_area = buf1.area + buf2.area - intersection_area

        if union_area < 1e-9: # Si ambos buffers son puntos/líneas sin área y no se superponen
            return 1.0 if buf1.equals(buf2) and similarity_index_threshold <=1.0 else 0.0 # Geometrías idénticas

        jaccard_index = intersection_area / union_area
        return jaccard_index if jaccard_index >= similarity_index_threshold else 0.0
    except TopologicalError:
        logger.debug(f"Error topológico en compare_similarity_index_score.", exc_info=False)
        return 0.0
    except Exception as e:
        logger.debug(f"Excepción en compare_similarity_index_score: {e}", exc_info=False)
        return 0.0

class StrategyWrapper:
    def __init__(self, score_func, param_key_1: str, param_key_2: str, strategy_name: str = "unknown_strategy"):
        self.score_func = score_func
        self.param_key_1 = param_key_1
        self.param_key_2 = param_key_2
        self.strategy_name = strategy_name

    def score(self, geom1, geom2, thresholds: dict) -> float:
        if geom1 is None or geom2 is None or geom1.is_empty or geom2.is_empty:
            return 0.0
        
        val1 = thresholds.get(self.param_key_1)
        val2 = thresholds.get(self.param_key_2)

        if val1 is None or val2 is None:
            logger.warning(f"Umbrales no encontrados para la estrategia {self.strategy_name} en el diccionario de umbrales: '{self.param_key_1}' o '{self.param_key_2}'.")
            return 0.0
        
        try:
            return self.score_func(geom1, geom2, val1, val2)
        except Exception as e:
            logger.error(f"Error calculando la puntuación con la estrategia {self.strategy_name}: {e}", exc_info=True)
            return 0.0

# --- Clase LayerMergerSequential ---
class LayerMergerSequential:
    """
    Empareja dos capas A y B secuencialmente y transfiere atributos de B a A.
    """
    def __init__(self, thresholds: dict = None, strategies: list[str] = None):
        default_thresholds = {
            'bibuffer_overlap': 50.0, # Distancia del buffer para la estrategia bibuffer_overlap
            'bibuffer_overlap_threshold': 0.7, # Umbral de ratio de superposición para bibuffer_overlap
            'similarity_index_buffer': 50.0, # Distancia del buffer para similarity_index
            'similarity_index_threshold': 0.7, # Umbral para similarity_index
            'vicinity_search': 500.0, # Distancia del buffer para la búsqueda de candidatos inicial
            'min_match_score': 0.1 # Umbral de puntuación mínima para considerar una coincidencia válida
        }
        self.thresholds = default_thresholds
        if thresholds:
            self.thresholds.update(thresholds)

        self.strategies = strategies or ['bibuffer_overlap', 'similarity_index']

        # Mapear estrategias con wrappers
        # Asegúrate de que los nombres de estrategia y las claves en thresholds coincidan.
        self.strategy_wrappers = {
            'bibuffer_overlap': StrategyWrapper(compare_bibuffer_overlap_score, 
                                                'bibuffer_overlap', 'bibuffer_overlap_threshold',
                                                strategy_name='bibuffer_overlap'),
            'similarity_index': StrategyWrapper(compare_similarity_index_score, 
                                                'similarity_index_buffer', 'similarity_index_threshold',
                                                strategy_name='similarity_index')
        }
        # Filtrar wrappers para solo usar las estrategias especificadas
        self.active_strategy_wrappers = {
            s: self.strategy_wrappers[s] for s in self.strategies if s in self.strategy_wrappers
        }


    def _compute_match_score(self, geom1, geom2) -> float:
        """Calcula una puntuación de coincidencia agregada entre dos geometrías."""
        if not self.active_strategy_wrappers or geom1 is None or geom2 is None or geom1.is_empty or geom2.is_empty:
            return 0.0
        
        scores = [
            wrapper.score(geom1, geom2, self.thresholds)
            for _, wrapper in self.active_strategy_wrappers.items()
        ]
        return sum(scores) / len(scores) if scores else 0.0

    def _find_best_match_idx(self, row_a: pd.Series, gdf_a_crs, gdf_b: gpd.GeoDataFrame) -> tuple[object | None, float | None]:
        """
        Encuentra el índice de la mejor coincidencia en gdf_b para row_a.
        Args:
            row_a (pd.Series): La fila de gdf_a para la que se busca coincidencia.
            gdf_a_crs: El CRS de gdf_a.
            gdf_b (gpd.GeoDataFrame): El GeoDataFrame donde buscar coincidencias.
        Returns:
            tuple[object | None, float | None]: (índice en gdf_b, puntuación). pd.NA para no encontrados.
        """
        geom_a = row_a.geometry
        if geom_a is None or geom_a.is_empty:
            return pd.NA, pd.NA

        # Validar CRS
        if gdf_a_crs is None:
            logger.warning("CRS de gdf_a no está definido. Las operaciones espaciales pueden ser incorrectas.")
            # No se puede continuar de forma fiable sin CRS para el buffer.
            return pd.NA, pd.NA
        if gdf_b.crs is None:
            logger.warning("CRS de gdf_b no está definido. Las operaciones espaciales pueden ser incorrectas.")
            # Se podría intentar continuar, pero sjoin será problemático.
        elif gdf_a_crs != gdf_b.crs:
            logger.warning(f"CRS de gdf_a ({gdf_a_crs}) y gdf_b ({gdf_b.crs}) son diferentes. "
                           "Se recomienda reproyectar gdf_b al CRS de gdf_a antes de la fusión.")
            # Para este ejemplo, no se reproyectará aquí, lo que puede llevar a errores/resultados incorrectos.
            # En una implementación robusta, se debería manejar esto (ej. error o reproyección).

        search_buffer_dist = self.thresholds.get('vicinity_search', 0.0)
        try:
            # if not geom_a.is_valid: geom_a = geom_a.buffer(0) # Opcional: limpiar geometría
            # if geom_a.is_empty: return pd.NA, pd.NA
            
            # El buffer se crea en el CRS de geom_a (gdf_a_crs)
            search_area = geom_a.buffer(search_buffer_dist)
            if search_area.is_empty:
                return pd.NA, pd.NA
        except Exception as e:
            logger.debug(f"Error creando buffer de búsqueda para la geometría: {e}", exc_info=False)
            return pd.NA, pd.NA

        # Crear GeoDataFrame para el área de búsqueda con el CRS correcto
        search_gdf = gpd.GeoDataFrame(geometry=[search_area], crs=gdf_a_crs)

        candidates = gpd.GeoDataFrame()
        try:
            # sjoin requiere que ambos GDFs estén en el mismo CRS.
            # Si gdf_b.crs != gdf_a_crs, esto fallará o dará resultados incorrectos.
            # Asumimos que el usuario ha asegurado la consistencia de CRS antes.
            if not gdf_b.empty and gdf_b.crs == search_gdf.crs : # Solo si hay candidatos y CRS coinciden
                 candidates = gpd.sjoin(gdf_b, search_gdf, how='inner', predicate='intersects')
                 if 'index_right' in candidates.columns: # Columna del índice de search_gdf
                    candidates = candidates.drop(columns=['index_right'])
            elif not gdf_b.empty and gdf_b.crs != search_gdf.crs:
                logger.warning(f"Saltando sjoin debido a CRS incompatibles: gdf_b ({gdf_b.crs}), search_gdf ({search_gdf.crs})")


        except Exception as e:
            logger.debug(f"Error durante sjoin: {e}", exc_info=False)
            return pd.NA, pd.NA # No se pueden encontrar candidatos

        if candidates.empty:
            return pd.NA, pd.NA

        best_idx_b = pd.NA
        best_score = -1.0

        for original_b_idx, row_b in candidates.iterrows():
            score = self._compute_match_score(geom_a, row_b.geometry)
            if score > best_score:
                best_idx_b = original_b_idx # Índice original de gdf_b
                best_score = score
        
        min_acceptable_score = self.thresholds.get('min_match_score', 0.0)
        if best_score < min_acceptable_score or pd.isna(best_idx_b):
            return pd.NA, pd.NA # O best_score si se quiere registrar aunque no sea aceptable

        return best_idx_b, best_score

    def merge_layers(self, gdf_a: gpd.GeoDataFrame, gdf_b: gpd.GeoDataFrame,
                     attrs_to_transfer: list[str] = None, prefix: str = 'merged_') -> gpd.GeoDataFrame:
        if not isinstance(gdf_a, gpd.GeoDataFrame) or not isinstance(gdf_b, gpd.GeoDataFrame):
            raise ValueError("gdf_a y gdf_b deben ser GeoDataFrames.")

        # Columnas a no renombrar
        exclude_rename_cols = ['geometry', 'INODE', 'JNODE', 'LENGTH', 'LOG_DIRECTION']
        renamed_cols_a = {
            col: prefix + col for col in gdf_a.columns if col not in exclude_rename_cols
        }
        result_gdf = gdf_a.rename(columns=renamed_cols_a).copy()

        if gdf_a.empty:
            logger.info("gdf_a está vacío. Devolviendo copia renombrada sin fusiones.")
            # Asegurar que las columnas de atributos y score existan si gdf_a está vacío pero se esperan
            attrs_to_transfer_actual = attrs_to_transfer or [c for c in gdf_b.columns if c != gdf_b.geometry.name]
            for attr in attrs_to_transfer_actual:
                if attr not in result_gdf.columns: result_gdf[attr] = pd.NA
            if 'match_score' not in result_gdf.columns: result_gdf['match_score'] = pd.NA
            return result_gdf
        
        # Validar CRS antes del bucle
        if gdf_a.crs is None:
            logger.warning("gdf_a no tiene CRS. Resultados de fusión pueden ser incorrectos.")
        if gdf_b.crs is None:
            logger.warning("gdf_b no tiene CRS. Resultados de fusión pueden ser incorrectos.")
        elif gdf_a.crs != gdf_b.crs:
            logger.warning(f"CRS de gdf_a ({gdf_a.crs}) y gdf_b ({gdf_b.crs}) son diferentes. "
                           "Se recomienda encarecidamente reproyectar gdf_b al CRS de gdf_a antes de llamar a merge_layers.")


        attrs_to_transfer_actual = attrs_to_transfer or [c for c in gdf_b.columns if c != gdf_b.geometry.name]
        
        match_indices_b = []
        match_scores_list = []

        for _, row_a in gdf_a.iterrows():
            idx_b, score = self._find_best_match_idx(row_a, gdf_a.crs, gdf_b)
            match_indices_b.append(idx_b)
            match_scores_list.append(score if pd.notna(score) else pd.NA)

        # Transferir atributos
        for attr in attrs_to_transfer_actual:
            attr_values = []
            for b_idx in match_indices_b:
                if pd.notna(b_idx) and b_idx in gdf_b.index:
                    attr_values.append(gdf_b.loc[b_idx, attr])
                else:
                    attr_values.append(pd.NA)
            result_gdf[attr] = pd.Series(attr_values, index=result_gdf.index)
            
        result_gdf['match_score'] = pd.Series(match_scores_list, index=result_gdf.index)
        
        result_gdf = self.merge_multilines(result_gdf)
        return result_gdf

    def merge_multilines(self, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """Procesa geometrías, intentando fusionar MultiLineStrings."""
        if gdf.empty or 'geometry' not in gdf.columns:
            return gdf

        new_geometries = []
        for idx, geom in enumerate(gdf.geometry):
            if isinstance(geom, MultiLineString):
                try:
                    merged = linemerge(geom)
                    if isinstance(merged, MultiLineString):
                        logger.warning(f"Geometría MultiLineString en el índice {gdf.index[idx]} no se pudo fusionar completamente.")
                    new_geometries.append(merged)
                except Exception as e:
                    logger.error(f"Error fusionando MultiLineString en el índice {gdf.index[idx]}: {e}", exc_info=True)
                    new_geometries.append(geom) # Mantener la geometría original en caso de error
            else:
                new_geometries.append(geom)
        
        # Es más seguro crear una nueva serie de geometría y asignarla
        gdf_processed = gdf.copy()
        gdf_processed.geometry = gpd.GeoSeries(new_geometries, crs=gdf.crs, index=gdf.index)
        return gdf_processed

# --- Clase LayerMergerParallel ---
class LayerMergerParallel(LayerMergerSequential): # Hereda __init__, _compute_match_score, merge_multilines
    """
    Empareja dos capas A y B usando paralelización con joblib y transfiere atributos de B a A.
    """
    def _task_find_best_match_for_geom(self, geom_a, gdf_a_crs, gdf_b_ref: gpd.GeoDataFrame) -> tuple[object | None, float | None]:
        """
        Tarea para joblib: Encuentra la mejor coincidencia para una única geometría de gdf_a.
        Esta función es similar a _find_best_match_idx pero toma geom_a directamente.
        """
        # Esta función es casi idéntica a _find_best_match_idx, pero adaptada para tomar geom_a
        # en lugar de row_a. Esto puede ser ligeramente más eficiente en términos de serialización
        # si solo se necesita la geometría y su CRS.
        if geom_a is None or geom_a.is_empty:
            return pd.NA, pd.NA

        # CRS checks (informativo, la corrección debe ser previa)
        if gdf_a_crs is None: # Este CRS es para geom_a
            # logger.warning("_task: geom_a_crs no definido.") # Puede ser ruidoso en paralelo
            return pd.NA, pd.NA
        if gdf_b_ref.crs is None:
            # logger.warning("_task: gdf_b_ref.crs no definido.")
            pass # Continuar con precaución
        elif gdf_a_crs != gdf_b_ref.crs:
            # logger.warning(f"_task: CRS de geom_a ({gdf_a_crs}) y gdf_b ({gdf_b_ref.crs}) difieren.")
            pass # Asumir que el usuario lo maneja o que sjoin fallará/será incorrecto

        search_buffer_dist = self.thresholds.get('vicinity_search', 0.0)
        try:
            search_area = geom_a.buffer(search_buffer_dist)
            if search_area.is_empty:
                return pd.NA, pd.NA
        except Exception:
            return pd.NA, pd.NA

        search_gdf = gpd.GeoDataFrame(geometry=[search_area], crs=gdf_a_crs)
        
        candidates = gpd.GeoDataFrame()
        try:
            if not gdf_b_ref.empty and gdf_b_ref.crs == search_gdf.crs:
                 candidates = gpd.sjoin(gdf_b_ref, search_gdf, how='inner', predicate='intersects')
                 if 'index_right' in candidates.columns:
                    candidates = candidates.drop(columns=['index_right'])
            # No loguear advertencias de CRS aquí para evitar spam desde workers paralelos
        except Exception:
            return pd.NA, pd.NA

        if candidates.empty:
            return pd.NA, pd.NA

        best_idx_b = pd.NA
        best_score = -1.0

        for original_b_idx, row_b in candidates.iterrows():
            score = self._compute_match_score(geom_a, row_b.geometry) # Llama al método de la instancia
            if score > best_score:
                best_idx_b = original_b_idx
                best_score = score
        
        min_acceptable_score = self.thresholds.get('min_match_score', 0.0)
        if best_score < min_acceptable_score or pd.isna(best_idx_b):
            return pd.NA, pd.NA

        return best_idx_b, best_score

    def merge_layers(self, gdf_a: gpd.GeoDataFrame, gdf_b: gpd.GeoDataFrame,
                     attrs_to_transfer: list[str] = None, prefix: str = 'merged_',
                     n_jobs: int = 6, prefer: str = None, require: str = None) -> gpd.GeoDataFrame:
        """
        Fusiona atributos de gdf_b a gdf_a usando paralelización con joblib.
        Args:
            n_jobs (int): Número de trabajos para joblib.
            prefer (str): Argumento 'prefer' para joblib.Parallel (e.g., "threads", "processes").
            require (str): Argumento 'require' para joblib.Parallel.
        """
        if not isinstance(gdf_a, gpd.GeoDataFrame) or not isinstance(gdf_b, gpd.GeoDataFrame):
            raise ValueError("gdf_a y gdf_b deben ser GeoDataFrames.")

        exclude_rename_cols = ['geometry', 'INODE', 'JNODE', 'LENGTH', 'LOG_DIRECTION']
        renamed_cols_a = {
            col: prefix + col for col in gdf_a.columns if col not in exclude_rename_cols
        }
        result_gdf = gdf_a.rename(columns=renamed_cols_a).copy()

        if gdf_a.empty:
            logger.info("gdf_a está vacío. Devolviendo copia renombrada sin fusiones.")
            attrs_to_transfer_actual = attrs_to_transfer or [c for c in gdf_b.columns if c != gdf_b.geometry.name]
            for attr in attrs_to_transfer_actual:
                if attr not in result_gdf.columns: result_gdf[attr] = pd.NA
            if 'match_score' not in result_gdf.columns: result_gdf['match_score'] = pd.NA
            return result_gdf

        # --- Validación de CRS CRUCIAL ---
        if gdf_a.crs is None:
            logger.error("¡CRÍTICO! gdf_a no tiene CRS. La fusión paralela no puede continuar de forma segura.")
            # Podrías optar por devolver result_gdf o lanzar un error.
            return result_gdf # Opcional: añadir columnas vacías de atributos y score
        if gdf_b.crs is None:
            logger.warning("gdf_b no tiene CRS. Los resultados de la fusión pueden ser incorrectos o fallar.")
            # Considera reproyectar gdf_b o lanzar error si es crítico.
        elif gdf_a.crs != gdf_b.crs:
            logger.warning(f"CRS de gdf_a ({gdf_a.crs}) y gdf_b ({gdf_b.crs}) son diferentes. "
                           "Para un rendimiento y corrección óptimos en paralelo, "
                           "REPROYECTA gdf_b AL CRS DE gdf_a ANTES de llamar a este método.")
            # Aquí podrías forzar una reproyección de gdf_b, pero es mejor hacerlo fuera.
            # gdf_b = gdf_b.to_crs(gdf_a.crs) # ¡CUIDADO! Modifica gdf_b para todas las tareas.
                                           # Esto es aceptable si gdf_b es una copia o si es el comportamiento deseado.

        attrs_to_transfer_actual = attrs_to_transfer or [c for c in gdf_b.columns if c != gdf_b.geometry.name]

        # Preparar tareas para joblib
        # Pasamos la geometría y su CRS, y una referencia a gdf_b.
        # self (la instancia de LayerMergerParallel) también se serializa y se envía,
        # lo que permite que _task_find_best_match_for_geom llame a self._compute_match_score.
        tasks = [
            delayed(self._task_find_best_match_for_geom)(row.geometry, gdf_a.crs, gdf_b)
            for _, row in gdf_a.iterrows() # Iterar sobre gdf_a original para obtener geometrías
        ]

        if not tasks: # No debería ocurrir si gdf_a.empty ya se manejó, pero por si acaso.
            for attr in attrs_to_transfer_actual:
                if attr not in result_gdf.columns: result_gdf[attr] = pd.NA
            if 'match_score' not in result_gdf.columns: result_gdf['match_score'] = pd.NA
            return result_gdf

        logger.info(f"Iniciando procesamiento paralelo de {len(tasks)} tareas con n_jobs={n_jobs}...")
        
        # `backend='loky'` es el predeterminado y generalmente robusto.
        # `prefer` y `require` pueden dar más control sobre el backend.
        parallel_results = Parallel(n_jobs=n_jobs, prefer=prefer, require=require, verbose=5)(tasks)
        # verbose=5 para un poco de feedback de joblib durante la ejecución. Ajusta según necesidad.

        logger.info("Procesamiento paralelo completado. Ensamblando resultados...")

        match_indices_b = [res[0] for res in parallel_results]
        match_scores_list = [res[1] for res in parallel_results]

        # Transferir atributos
        for attr in attrs_to_transfer_actual:
            attr_values = []
            for b_idx in match_indices_b:
                if pd.notna(b_idx) and b_idx in gdf_b.index:
                    attr_values.append(gdf_b.loc[b_idx, attr])
                else:
                    attr_values.append(pd.NA)
            result_gdf[attr] = pd.Series(attr_values, index=result_gdf.index)
            
        result_gdf['match_score'] = pd.Series(match_scores_list, index=result_gdf.index)
        
        result_gdf = self.merge_multilines(result_gdf) # Método heredado
        logger.info("Fusión de capas completada.")
        return result_gdf

