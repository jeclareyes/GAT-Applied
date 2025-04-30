# geometry_processing.py: contiene las clases GeometryCleaner y LayerMerger


import geopandas as gpd
import pandas as pd
import logging
from shapely.geometry import base, LineString, Point, MultiLineString
from shapely.geometry.base import BaseMultipartGeometry
from shapely.ops import unary_union
from project_utils import (
    compare_exact, compare_distance, compare_diff,
    compare_overlap_ratio, compare_buffer_overlap,
    compare_bibuffer_overlap, compare_similarity_index
)


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

    def _decompose_and_linemerge(self, merged_geom: base.BaseGeometry) -> list[base.BaseGeometry]:
        """
        Descompone una geometría (posiblemente multipart) en geometrías simples.
        Si la geometría es un MultiLineString, intenta fusionar las partes
        en LineStrings más largos usando linemerge antes de descomponer.
        """
        geometries = []
        if isinstance(merged_geom, BaseMultipartGeometry):
            # Si es un MultiLineString, intenta fusionar partes conectadas
            if isinstance(merged_geom, MultiLineString):
                try:
                    # linemerge puede retornar un LineString o un MultiLineString
                    merged_lines = linemerge(merged_geom)
                    if isinstance(merged_lines, BaseMultipartGeometry):
                        # Si linemerge todavía resulta en multipart, añadir las partes
                        geometries.extend(list(merged_lines.geoms))
                    else:
                        # Si linemerge resulta en un LineString simple, añadirlo
                        geometries.append(merged_lines)
                    logger.debug("MultiLineString procesado con linemerge.")
                except Exception as e:
                    # Si linemerge falla por alguna razón, descomponer directamente
                    logger.warning(f"linemerge falló: {e}. Descomponiendo MultiLineString directamente.")
                    geometries.extend(list(merged_geom.geoms))
            else:
                # Para otros tipos multipart (MultiPolygon, MultiPoint), descomponer directamente
                geometries.extend(list(merged_geom.geoms))
        else:
            # Si ya es una geometría simple, añadirla directamente
            geometries.append(merged_geom)

        # Filtrar geometrías nulas o vacías que puedan haber resultado de las operaciones
        return [geom for geom in geometries if geom is not None and not geom.is_empty]

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

    def blend_duplicates(self, gdf: gpd.GeoDataFrame, id_col='Avsnittsidentitet',
                         strategies=None) -> gpd.GeoDataFrame:
        # Configura estrategias
        strategies = strategies or [
            'exact', 'distance', 'diff', 'overlap_ratio',
            'buffer_overlap', 'bibuffer_overlap', 'similarity_index'
        ]
        # Mapa de funciones
        strategy_map = {
            'exact': compare_exact,
            'distance': lambda g1,g2: compare_distance(g1,g2,self.thresholds['distance']),
            'diff': lambda g1,g2: compare_diff(g1,g2,self.thresholds['diff']),
            'overlap_ratio': lambda g1,g2: compare_overlap_ratio(g1,g2,self.thresholds['overlap_ratio']),
            'buffer_overlap': lambda g1,g2: compare_buffer_overlap(g1,g2,self.thresholds['buffer_overlap']),
            'bibuffer_overlap': lambda g1,g2: compare_bibuffer_overlap(
                g1,g2,self.thresholds['bibuffer_overlap'], self.thresholds['bibuffer_overlap_threshold']
            ),
            'similarity_index': lambda g1,g2: compare_similarity_index(
                g1,g2,self.thresholds['similarity_index_buffer'], self.thresholds['similarity_index_threshold']
            )
        }

        merged = []
        for eid, group in gdf.groupby(id_col):
            rows = group.reset_index(drop=True)
            used = [False]*len(rows)
            for i,row in rows.iterrows():
                if used[i]: continue
                used[i]=True
                current = row.geometry
                merged_idx=[i]
                strat_used='None'
                for j,other in rows.iterrows():
                    if used[j]: continue
                    for strat in strategies:
                        match,name = strategy_map[strat](current, other.geometry)
                        if match:
                            strat_used = name
                            used[j]=True
                            merged_idx.append(j)
                            break
                subset=rows.loc[merged_idx]
                unioned=unary_union(subset.geometry.tolist())
                parts = self._decompose_and_linemerge(unioned)
                for geom in parts:
                    attr=subset.iloc[0].drop('geometry').to_dict()
                    attr['geometry']=geom
                    attr['Similiarity_Strategy']=strat_used
                    merged.append(attr)
        return gpd.GeoDataFrame(merged, crs=gdf.crs)

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
        self.strategies = strategies or (
            [ 'bibuffer_overlap', 'similarity_index' ]) 

    def _match_with_strategies(self, line_a, line_b) -> tuple[bool,str|None]:
        # Mapa de funciones
        strategy_map = {
            'bibuffer_overlap': lambda g1,g2: 
            compare_bibuffer_overlap(g1,g2,self.thresholds['bibuffer_overlap'], self.thresholds['bibuffer_overlap_threshold']),
            'similarity_index': lambda g1,g2: 
            compare_similarity_index(g1,g2,self.thresholds['similarity_index_buffer'], self.thresholds['similarity_index_threshold'])
            }

        for strat in self.strategies:
            match, name = strategy_map[strat](line_a, line_b)
            if match:
                return True, name
        return False, None
    

    def _find_best_match_idx(self, 
                         row_a: pd.Series, 
                         gdf_b: gpd.GeoDataFrame) -> tuple[int|None, str|None]: 
        """ 
        Usa un buffer igual al umbral de bibuffer para hacer sjoin 
        y luego aplica las estrategias en ese subconjunto. 
        Devuelve (best_idx, best_strategy). 
        """
        # 1. Crear GeoDataFrame temporal con el buffer de tolerancia 
        buf = row_a.geometry.buffer(self.thresholds['vicinity_search']) 
        buf_gdf = gpd.GeoDataFrame(geometry=[buf], crs=gdf_b.crs) 
    
        # 2. Spatial join para candidatos cercanos 
        candidates = ( 
            gpd.sjoin(gdf_b, buf_gdf, how='inner', predicate='intersects') 
            .drop(columns=['index_right']) 
        ) 
    
        # 3. Iterar solo esos candidatos 
        for idx_b, row_b in candidates.iterrows(): 
            match, strat = self._match_with_strategies(row_a.geometry, row_b.geometry) 
            if match: 
                return idx_b, strat 
    
        return None, None 


    def merge_layers(self, 
                 gdf_a: gpd.GeoDataFrame, 
                 gdf_b: gpd.GeoDataFrame, 
                 attrs_to_transfer: list[str] = None, 
                 prefix: str = 'Emme_' 
                ) -> gpd.GeoDataFrame: 
        """ 
        Para cada feature de A: 
        - Busca candidatos en B dentro del buffer. 
        - Aplica estrategias en orden. 
        - Copia atributos de B (vectorizado) cuando hay match. 
        """ 
        result = gdf_a.copy() 
        attrs = attrs_to_transfer or [c for c in gdf_b.columns if c != 'geometry'] 
    
        # Prepara arrays para vectorizar la asignación 
        match_idxs = [] 
        match_strats = [] 
    
        # 1. Para cada fila de A, determinar correspondencia 
        for _, row_a in result.iterrows(): 
            idx_b, strat = self._find_best_match_idx(row_a, gdf_b) 
            match_idxs.append(idx_b) 
            match_strats.append(strat or 'None') 
    
        # 2. Transferencia vectorizada de atributos 
        for attr in attrs: 
            result[prefix + attr] = [ 
                (gdf_b.at[idx, attr] if idx is not None else None) 
                for idx in match_idxs 
            ] 
    
        # 3. Guardar la estrategia usada 
        result['match_strategy'] = match_strats 
    
        return result 

class AttributeConsolidator:
    def __init__(self, temporal_fields=None):
        from configs.settings import Fields
        self.temporal_fields = temporal_fields or Fields.TEMPORAL_FIELDS
        self.drop_fields = Fields.DROP_FIELDS_LASTKAJEN or []

    def consolidate(self, gdf, year):
        gdf = gdf.copy()
        suffix = f"_{year}"
        for field in list(set(self.temporal_fields) - set(self.drop_fields)):
            if field in gdf.columns:
                gdf.rename(columns={field: field + suffix}, inplace=True)
            else:
                match = find_best_match(field, gdf.columns)
                if match:
                    gdf.rename(columns={match: match + suffix}, inplace=True)
                else:
                    logger.warning(f"Campo '{field}' no encontrado para año {year}")
        return gdf

class NodeIdentifier:
    def __init__(self, precision=6):
        self.precision = precision

    def identify(self, gdf):
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
            tipo = 'Intersección' if info['count'] > 1 else 'Nodo_final'
            nodes.append({'node_id': f'node_{i}', 'tipo': tipo, 'elements': info['elements'], 'geometry': Point(coord)})
        gdf_nodes = gpd.GeoDataFrame(nodes, crs=gdf.crs)
        orphans = [idx for idx, (s, e) in seg_map.items() if
                   coord_info[s]['count'] == 1 and coord_info[e]['count'] == 1]
        orphan_segs = gdf.loc[orphans].copy() if orphans else None
        return gdf_nodes, orphan_segs



