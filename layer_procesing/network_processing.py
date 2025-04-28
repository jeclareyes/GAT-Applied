# network_processing.py: fusión de geometry.py, attributes.py, matching.py, nodes.py y topology_updater.py

import geopandas as gpd
import pandas as pd
import logging
from shapely.geometry import base, LineString, MultiLineString, Point
from shapely.geometry.base import BaseMultipartGeometry
from shapely.ops import unary_union, linemerge
from project_utils import find_best_match

logger = logging.getLogger(__name__)


class GeometryCleaner:
    """
    Clase para validar, corregir, opcionalmente simplificar y fusionar geometrías
    contenidas en un GeoDataFrame.

    Funciones principales:
      - Remover geometrías nulas o vacías.
      - Corregir geometrías inválidas usando buffer(0).
      - Simplificar geometrías con un nivel de tolerancia dado (funcionalidad no implementada
        en este fragmento, pero la estructura lo permite).
      - Fusionar geometrías duplicadas o similares basándose en múltiples estrategias.
      - Descomponer geometrías multipart resultantes de la fusión.
      - Registrar cambios y errores mediante logging.
    """

    def __init__(self):
        """
        Inicializa el limpiador de geometrías.
        """
        # Puedes añadir parámetros de inicialización aquí si son necesarios
        # para toda la clase, como una tolerancia general o un logger específico.
        pass

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

    def _compare_ids(self, geom1: base.BaseGeometry, geom2: base.BaseGeometry,
                     row1: pd.Series, row2: pd.Series) -> tuple[bool, str | None]:
        """
        Compara no sólo geometrías sino también valores de columnas 'Avsnittsidentitet', 'seq' y 'longitud'.
        """
        if (row1.get('SEQ_NO') == row2.get('SEQ_NO') and
            row1.get('DIRECTION') == row2.get('DIRECTION') and
            row1.get('ROLE') == row2.get('ROLE') and
            abs(row1.get('EXTENT_LENGTH') - row2.get('EXTENT_LENGTH')) < 5.0):
            return True, 'id_match'
        return False, None

    def _compare_exact(self, geom1: base.BaseGeometry, geom2: base.BaseGeometry) -> tuple[bool, str | None]:
        """Compara si dos geometrías son exactamente iguales."""
        if geom1.equals(geom2):
            return True, "exact"
        return False, None

    def _compare_distance(self, geom1: base.BaseGeometry, geom2: base.BaseGeometry, threshold: float) -> tuple[
        bool, str | None]:
        """Compara si la distancia entre dos geometrías es menor o igual al umbral."""
        if geom1.distance(geom2) <= threshold:
            return True, "distance"
        return False, None

    def _compare_diff(self, geom1: base.BaseGeometry, geom2: base.BaseGeometry, threshold: float) -> tuple[
        bool, str | None]:
        """Compara si la diferencia simétrica entre dos geometrías tiene una longitud menor o igual al umbral."""
        # Asegurarse de que las geometrías son lineales para calcular la longitud de la diferencia
        if geom1.geom_type in ['LineString', 'MultiLineString'] and geom2.geom_type in ['LineString',
                                                                                        'MultiLineString']:
            if geom1.symmetric_difference(geom2).length <= threshold:
                return True, "diff"
        # Considerar otras tipos de geometría si es necesario, adaptando la métrica de comparación
        # Por ahora, solo aplica a líneas.
        return False, None

    def _compare_overlap_ratio(self, geom1: base.BaseGeometry, geom2: base.BaseGeometry, threshold: float) -> tuple[
        bool, str | None]:
        """Compara si la relación de solapamiento (intersección / longitud mínima) es mayor o igual al umbral."""
        # Esta estrategia es más adecuada para líneas o polígonos
        if geom1.geom_type in ['LineString', 'MultiLineString'] and geom2.geom_type in ['LineString',
                                                                                        'MultiLineString']:
            inter = geom1.intersection(geom2)
            min_len = min(geom1.length, geom2.length)
            # Evitar división por cero
            if min_len > 0 and inter.length / min_len >= threshold:
                return True, "overlap_ratio"
        # Puedes añadir lógica para polígonos usando área si es necesario
        return False, None

    def _compare_buffer_overlap(self, geom1: base.BaseGeometry, geom2: base.BaseGeometry, buffer_amount: float) -> \
    tuple[bool, str | None]:
        """Compara si la geometría 2 intersecta significativamente el buffer de la geometría 1."""
        # Esta estrategia es útil para encontrar geometrías cercanas o parcialmente solapadas.
        # Se asume que un solapamiento del 80% de la longitud de geom2 con el buffer de geom1
        # indica similitud. Este umbral (0.8) puede ser otro parámetro si se desea.
        buf = geom1.buffer(buffer_amount)
        if buf.intersects(geom2):
            # Calcular la longitud de la intersección del buffer con geom2
            inter_len = buf.intersection(geom2).length
            # Evitar división por cero
            if geom2.length > 0 and inter_len / geom2.length >= 0.8:
                return True, "buffer_overlap"
        return False, None

    def _compare_bibuffer_overlap(self, geom1: base.BaseGeometry, geom2: base.BaseGeometry,
                                  buffer_amount: float = 50.0, threshold: float = 0.7) -> tuple[bool, str | None]:
        """Compara solapamiento mutuo de buffers en ambas direcciones."""
        buffer_amount = 50.0
        buf1 = geom1.buffer(buffer_amount)
        buf2 = geom2.buffer(buffer_amount)
        overlap1 = buf1.intersection(geom2).length / geom2.length if geom2.length > 0 else 0
        overlap2 = buf2.intersection(geom1).length / geom1.length if geom1.length > 0 else 0
        return (True, "bibuffer_overlap") if max(overlap1, overlap2) >= threshold else (False, None)


    def _similarity_index(self, geom1: base.BaseGeometry, geom2: base.BaseGeometry,
                          buffer_amount: float = 50.0, threshold: float = 0.7) -> tuple[bool, str | None]:
        """Índice de similitud basado en intersección de buffers."""
        buffer_amount = 50.0
        buf1 = geom1.buffer(buffer_amount)
        buf2 = geom2.buffer(buffer_amount)
        inter_len = buf1.intersection(buf2).length
        total_len = geom1.length + geom2.length
        similarity = (2 * inter_len) / total_len if total_len > 0 else 0
        similarity = min(1.0, similarity)
        return (True, "similarity_index") if similarity >= threshold else (False, None)

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

    def blend_duplicates(self, gdf, id_col="Avsnittsidentitet",
                         strategies=None, thresholds=None) -> gpd.GeoDataFrame:
        if strategies is None:
            strategies = ["exact", "distance", "diff", "overlap_ratio", "buffer_overlap", "bibuffer_overlap",
                          "similarity_index"]
        if thresholds is None:
            thresholds = {
                "distance": 5.0,
                "diff": 5.0,
                "overlap_ratio": 0.5,
                "buffer_overlap": 50.0,
                "bibuffer_overlap": 50.0,  # meters
                "bibuffer_overlap_threshold": 0.7,  # percentage
                "similarity_index": 50.0,
                "similarity_index_threshold": 0.7,
            }
            logger.info(f"Usando umbrales por defecto: {thresholds}")
        else:
            logger.info(f"Usando umbrales proporcionados: {thresholds}")

        # Mapa de estrategia a método, sin lambdas extrañas
        strategy_map = {
             "id_match": lambda g1, g2, r1, r2: self._compare_ids(g1, g2, r1, r2),
             "exact": self._compare_exact,
             "distance": lambda g1, g2: self._compare_distance(g1, g2, thresholds["distance"]),
             "diff": lambda g1, g2: self._compare_diff(g1, g2, thresholds["diff"]),
             "overlap_ratio": lambda g1, g2: self._compare_overlap_ratio(g1, g2, thresholds["overlap_ratio"]),
             "buffer_overlap": lambda g1, g2: self._compare_buffer_overlap(g1, g2, thresholds["buffer_overlap"]),
             "bibuffer_overlap": lambda g1, g2: self._compare_bibuffer_overlap(g1, g2, thresholds["bibuffer_overlap"], thresholds.get("bibuffer_overlap_threshold")),
             "similarity_index": lambda g1, g2: self._similarity_index(g1, g2, thresholds.get("similarity_index_buffer"), thresholds.get("similarity_index_threshold"))
             }

        # Lista para guardar los resultados finales
        merged = []

        # 1. Agrupamos el GeoDataFrame por la columna que define el grupo (por ejemplo, 'ELEMENT_ID')
        for eid, group in gdf.groupby(id_col):

            # 2. Inicializamos un marcador para saber qué filas ya hemos usado en fusiones
            used = [False] * len(group)

            # 3. Reseteamos el índice del grupo para trabajar con índices ordenados (0, 1, 2, ...)
            rows = group.reset_index(drop=True)

            # 4. Recorremos cada fila en el grupo
            for i, row in rows.iterrows():
                if used[i]:
                    continue  # Si esta fila ya se usó en otra fusión, la saltamos

                # 5. Inicializamos un nuevo grupo de fusión
                current = row.geometry  # Geometría de referencia
                used[i] = True  # Marcamos que ya estamos usando esta fila
                merged_indices = [i]  # Lista de índices que serán fusionados
                sim_strategy = "None"  # Estrategia usada para fusionar (por ahora ninguna)

                # 6. Buscamos otras filas dentro del mismo grupo que puedan fusionarse
                for j, other in rows.iterrows():
                    if used[j]:
                        continue  # Si esta fila ya se usó, la saltamos

                    # 7. Probamos todas las estrategias de similitud en orden
                    for strat in strategies:
                        match, name = (
                            strategy_map[strat](row.geometry, other.geometry, row, other)
                            if strat == 'id_match'
                            else strategy_map[strat](row.geometry, other.geometry)
                        )

                        if match:
                            # Si hay match según esta estrategia:
                            sim_strategy = name or strat  # Guardamos el nombre de la estrategia usada
                            merged_indices.append(j)  # Agregamos el índice de esta fila para fusionar
                            used[j] = True  # Marcamos que esta fila ya está usada
                            break  # No seguimos probando más estrategias para este par

                # 8. Una vez identificadas las filas similares, las fusionamos
                subset = rows.loc[merged_indices]  # Seleccionamos las filas a fusionar
                unioned = unary_union(subset.geometry.tolist())  # Fusionamos todas las geometrías en una sola

                # 9. A veces la fusión genera geometrías multipartes (MultiLineString)
                # Usamos una función para dividirlas en geometrías más simples
                decomposed_geometries = self._decompose_and_linemerge(unioned)

                # 10. Guardamos cada geometría simple como una nueva fila
                for simple_geom in decomposed_geometries:
                    # Para cada columna del subset (excepto 'geometry'), Si al eliminar los NaN de esta columna
                    # queda algún valor tomamos el primer valor no-NaN encontrado Si no queda ningún valor
                    # (todos eran NaN), asignamos None
                    merged_attr = {
                        col: (subset[col].dropna().iloc[0] if not subset[col].dropna().empty else None)
                        for col in subset.columns if col != "geometry"
                    }
                    # Copiamos los atributos de la primera fila
                    merged_attr["geometry"] = simple_geom  # Asignamos la nueva geometría
                    merged_attr[
                        "Similarity Strategy"] = sim_strategy  # Guardamos la estrategia que se usó para la fusión
                    merged.append(merged_attr)  # Añadimos esta fila al resultado final

        return gpd.GeoDataFrame(merged, crs=gdf.crs)


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


def match_segments(gdf_links, gdf_nodes, node_tolerance=5.0):
    links = gdf_links.copy()
    nodes = gdf_nodes.copy()
    links['start_node_id'] = None;
    links['end_node_id'] = None;
    links['direction_log'] = 0
    nodes['incoming_links'] = [[] for _ in range(len(nodes))];
    nodes['outgoing_links'] = [[] for _ in range(len(nodes))]
    starts = [Point(g.coords[0]) for g in links.geometry if isinstance(g, LineString)]
    ends = [Point(g.coords[-1]) for g in links.geometry if isinstance(g, LineString)]
    gdf_s = gpd.GeoDataFrame(geometry=starts, crs=links.crs)
    gdf_e = gpd.GeoDataFrame(geometry=ends, crs=links.crs)
    sm = gpd.sjoin_nearest(gdf_s, nodes[['node_id', 'geometry']], how='left', max_distance=node_tolerance)
    em = gpd.sjoin_nearest(gdf_e, nodes[['node_id', 'geometry']], how='left', max_distance=node_tolerance)
    links['matched_start_node_id'] = sm['node_id'].values
    links['matched_end_node_id'] = em['node_id'].values
    for i, row in links.iterrows():
        s, e = row['matched_start_node_id'], row['matched_end_node_id']
        if pd.isna(s) or pd.isna(e): continue
        role, dirv = row.get('ROLE'), row.get('DIRECTION')
        d = 0
        if role == 'Normal':
            d = 0
        elif role in ['Syskon fram', 'Syskon bak']:
            d = 1;
            s, e = (e, s) if dirv == 'Mot' else (s, e)
        links.at[i, 'start_node_id'], links.at[i, 'end_node_id'], links.at[i, 'direction_log'] = s, e, d
        si = nodes[nodes.node_id == s].index[0];
        ei = nodes[nodes.node_id == e].index[0]
        if d == 1:
            nodes.at[si, 'outgoing_links'].append(row['ELEMENT_ID']);
            nodes.at[ei, 'incoming_links'].append(row['ELEMENT_ID'])
        else:
            for idxn in (si, ei):
                nodes.at[idxn, 'incoming_links'].append(row['ELEMENT_ID']);
                nodes.at[idxn, 'outgoing_links'].append(row['ELEMENT_ID'])
    links.drop(columns=['matched_start_node_id', 'matched_end_node_id'], inplace=True)
    return links, nodes
