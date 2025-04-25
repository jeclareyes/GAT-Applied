# processing/geometry.py
import logging
from shapely.geometry import base, LineString, MultiLineString
from shapely.ops import unary_union, linemerge
from shapely.geometry.base import BaseMultipartGeometry
import geopandas as gpd
import pandas as pd # Importar pandas para manejar datos

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
            return gdf # Retornar el original si el tipo es incorrecto

        initial_rows = len(gdf)
        to_drop = []  # Lista para registrar los índices con geometrías no recuperables
        cleaned_geometries = [] # Lista para almacenar las geometrías limpias

        # Recorremos cada fila del GeoDataFrame para validar y corregir su geometría
        for idx, row in gdf.iterrows():
            geom = row.geometry

            # Si la geometría es nula o vacía, la marcamos para eliminación
            if geom is None or geom.is_empty:
                logger.error(f"Geometría nula/vacía en índice {idx}")
                to_drop.append(idx)
                cleaned_geometries.append(None) # Usamos None como placeholder
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
                    cleaned_geometries.append(None) # Usamos None como placeholder
            else:
                cleaned_geometries.append(geom) # La geometría ya es válida

        # Actualizar las geometrías corregidas en el GeoDataFrame
        # Usamos .loc para evitar SettingWithCopyWarning
        gdf_cleaned = gdf.copy() # Trabajar sobre una copia para seguridad
        gdf_cleaned['geometry'] = cleaned_geometries

        # Eliminamos las filas con geometrías problemáticas que no se pudieron recuperar
        if to_drop:
            # Filtrar las filas a mantener
            valid_indices = [i for i in range(initial_rows) if i not in to_drop]
            gdf_cleaned = gdf_cleaned.iloc[valid_indices].reset_index(drop=True)
            logger.info(f"Eliminados {len(to_drop)} registros inválidos o no recuperables")

        logger.info(f"Proceso de limpieza completado. Filas iniciales: {initial_rows}, Filas finales: {len(gdf_cleaned)}")
        return gdf_cleaned  # Retornamos el GeoDataFrame limpio

    def _compare_exact(self, geom1: base.BaseGeometry, geom2: base.BaseGeometry) -> tuple[bool, str | None]:
        """Compara si dos geometrías son exactamente iguales."""
        if geom1.equals(geom2):
            return True, "exact"
        return False, None

    def _compare_distance(self, geom1: base.BaseGeometry, geom2: base.BaseGeometry, threshold: float) -> tuple[bool, str | None]:
        """Compara si la distancia entre dos geometrías es menor o igual al umbral."""
        if geom1.distance(geom2) <= threshold:
            return True, "distance"
        return False, None

    def _compare_diff(self, geom1: base.BaseGeometry, geom2: base.BaseGeometry, threshold: float) -> tuple[bool, str | None]:
        """Compara si la diferencia simétrica entre dos geometrías tiene una longitud menor o igual al umbral."""
        # Asegurarse de que las geometrías son lineales para calcular la longitud de la diferencia
        if geom1.geom_type in ['LineString', 'MultiLineString'] and geom2.geom_type in ['LineString', 'MultiLineString']:
             if geom1.symmetric_difference(geom2).length <= threshold:
                return True, "diff"
        # Considerar otras tipos de geometría si es necesario, adaptando la métrica de comparación
        # Por ahora, solo aplica a líneas.
        return False, None

    def _compare_overlap_ratio(self, geom1: base.BaseGeometry, geom2: base.BaseGeometry, threshold: float) -> tuple[bool, str | None]:
        """Compara si la relación de solapamiento (intersección / longitud mínima) es mayor o igual al umbral."""
        # Esta estrategia es más adecuada para líneas o polígonos
        if geom1.geom_type in ['LineString', 'MultiLineString'] and geom2.geom_type in ['LineString', 'MultiLineString']:
            inter = geom1.intersection(geom2)
            min_len = min(geom1.length, geom2.length)
            # Evitar división por cero
            if min_len > 0 and inter.length / min_len >= threshold:
                return True, "overlap_ratio"
        # Puedes añadir lógica para polígonos usando área si es necesario
        return False, None

    def _compare_buffer_overlap(self, geom1: base.BaseGeometry, geom2: base.BaseGeometry, buffer_amount: float) -> tuple[bool, str | None]:
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


    def blend_duplicates(self,
                         gdf: gpd.GeoDataFrame,
                         id_col: str = "ELEMENT_ID",
                         strategies: list = ["exact", "distance", "diff", "overlap_ratio", "buffer_overlap"],
                         thresholds: dict = None) -> gpd.GeoDataFrame:
        """
        Agrupa y combina filas duplicadas o similares en un GeoDataFrame usando múltiples estrategias,
        y descompone geometrías multipart resultantes.

        Las estrategias de comparación se aplican secuencialmente en el orden especificado.
        Una vez que un par de geometrías se considera similar por *cualquier* estrategia,
        se agrupan para la fusión.

        Parámetros:
        - gdf: GeoDataFrame con geometrías.
        - id_col: Nombre de la columna de ID para agrupar las geometrías.
        - strategies: Lista de nombres de estrategias a aplicar
                      ('exact', 'distance', 'diff', 'overlap_ratio', 'buffer_overlap').
                      El orden importa, ya que se aplican secuencialmente.
        - thresholds: Diccionario de umbrales por estrategia. Las claves deben coincidir
                      con los nombres de las estrategias que requieren un umbral.
                      Ejemplo: {"distance": 1.0, "diff": 5.0, "overlap_ratio": 0.8, "buffer_overlap": 1.0}.

        Retorna:
        - Un nuevo GeoDataFrame con los grupos de geometrías fusionados y descompuestos
          en geometrías simples. Incluye una columna 'SIMILARITY_STRATEGY' indicando
          qué estrategias se usaron para fusionar cada grupo original.
        """
        if not isinstance(gdf, gpd.GeoDataFrame):
            logger.error("La entrada a blend_duplicates no es un GeoDataFrame.")
            return gdf

        if id_col not in gdf.columns:
            logger.error(f"La columna de ID '{id_col}' no se encuentra en el GeoDataFrame.")
            return gdf # Retornar el original si la columna no existe

        if thresholds is None:
            # Umbrales por defecto si no se proporcionan
            thresholds = {
                "distance": 1.0,          # Distancia máxima para considerar similar (en unidades del CRS)
                "diff": 5.0,              # Longitud máxima de la diferencia simétrica para líneas
                "overlap_ratio": 0.8,     # Ratio mínimo de solapamiento para líneas
                "buffer_overlap": 1.0     # Tamaño del buffer para la estrategia buffer_overlap (en unidades del CRS)
            }
            logger.info(f"Usando umbrales por defecto: {thresholds}")
        else:
             logger.info(f"Usando umbrales proporcionados: {thresholds}")


        # Mapeo de nombres de estrategia a métodos de comparación
        strategy_map = {
            "exact": self._compare_exact,
            "distance": self._compare_distance,
            "diff": self._compare_diff,
            "overlap_ratio": self._compare_overlap_ratio,
            "buffer_overlap": self._compare_buffer_overlap,
        }

        # Validar que las estrategias solicitadas existan en el mapa
        valid_strategies = [s for s in strategies if s in strategy_map]
        if len(valid_strategies) != len(strategies):
            invalid = set(strategies) - set(valid_strategies)
            logger.warning(f"Estrategias no reconocidas serán ignoradas: {invalid}")
        strategies_to_use = valid_strategies

        merged_rows_list = [] # Lista para acumular los diccionarios de filas fusionadas

        # Agrupar por la columna de ID
        for eid, group in gdf.groupby(id_col):
            geoms_with_original_index = [(row.geometry, idx) for idx, row in group.iterrows()]
            num_geoms = len(geoms_with_original_index)
            used_indices_in_group = [False] * num_geoms # Para rastrear qué geometrías ya fueron procesadas en este grupo de ID

            for i in range(num_geoms):
                if used_indices_in_group[i]:
                    continue # Saltar si esta geometría ya fue incluida en un grupo de fusión

                # Iniciar un nuevo grupo de fusión con la geometría actual
                current_geom, original_idx_i = geoms_with_original_index[i]
                group_to_merge_indices_in_group = [i] # Índices dentro del grupo actual (no índices del gdf original)
                strategies_applied = {"self"} # Conjunto para registrar las estrategias usadas en este grupo de fusión
                used_indices_in_group[i] = True # Marcar como usada

                # Comparar la geometría actual con las siguientes en el grupo
                for j in range(i + 1, num_geoms):
                    if used_indices_in_group[j]:
                        continue # Saltar si ya fue usada

                    compare_geom, original_idx_j = geoms_with_original_index[j]
                    is_similar = False
                    strategy_used_for_pair = None

                    # Aplicar estrategias de comparación secuencialmente
                    for strategy_name in strategies_to_use:
                        compare_func = strategy_map[strategy_name]
                        threshold = thresholds.get(strategy_name) # Obtener umbral si existe

                        # Pasar el umbral solo si la función de comparación lo requiere
                        if strategy_name in ["distance", "diff", "overlap_ratio", "buffer_overlap"]:
                             sim, strat = compare_func(current_geom, compare_geom, threshold)
                        else: # Estrategias que no requieren umbral (ej: exact)
                             sim, strat = compare_func(current_geom, compare_geom)

                        if sim:
                            is_similar = True
                            strategy_used_for_pair = strat
                            break # Encontramos una similitud, no necesitamos probar más estrategias para este par

                    if is_similar:
                        # Si son similares, añadir al grupo de fusión y marcar como usada
                        used_indices_in_group[j] = True
                        group_to_merge_indices_in_group.append(j)
                        strategies_applied.add(strategy_used_for_pair) # Añadir la estrategia usada

                # Ahora fusionar las geometrías encontradas en este grupo de fusión
                indices_in_original_gdf = [geoms_with_original_index[k][1] for k in group_to_merge_indices_in_group]
                subset_to_merge = gdf.loc[indices_in_original_gdf]
                merged_geom_result = unary_union(subset_to_merge.geometry.tolist())

                # Descomponer la geometría fusionada (y aplicar linemerge si es MultiLineString de líneas)
                decomposed_geometries = self._decompose_and_linemerge(merged_geom_result)

                # Crear nuevas filas para cada geometría descompuesta
                for simple_geom in decomposed_geometries:
                    merged_attr = {}
                    # Copiar atributos de la primera fila del subconjunto (o definir lógica de agregación)
                    # Aquí simplemente tomamos los atributos de la primera fila del subconjunto
                    # Puedes refinar esta lógica si necesitas agregar o combinar atributos de otra manera
                    first_row_data = subset_to_merge.iloc[0].drop('geometry').to_dict()
                    merged_attr.update(first_row_data)

                    merged_attr["geometry"] = simple_geom
                    # Registrar todas las estrategias que llevaron a fusionar *cualquier* par en este grupo
                    merged_attr["SIMILARITY_STRATEGY"] = ",".join(sorted(list(strategies_applied))) # Usar sorted para consistencia
                    merged_rows_list.append(merged_attr)

        # Crear el GeoDataFrame final a partir de la lista de diccionarios
        # Asegurarse de incluir todas las columnas originales más la nueva
        output_columns = gdf.columns.tolist() + ["SIMILARITY_STRATEGY"]
        result_gdf = gpd.GeoDataFrame(merged_rows_list, columns=output_columns, crs=gdf.crs)

        logger.info(f"Proceso de fusión completado. Filas iniciales (antes de fusionar por ID): {len(gdf)}, Filas finales (después de fusión y descomposición): {len(result_gdf)}")

        return result_gdf

    # Puedes añadir aquí un método para simplificar geometrías si es necesario
    # def simplify_geometries(self, gdf: gpd.GeoDataFrame, tolerance: float) -> gpd.GeoDataFrame:
    #     """
    #     Simplifica las geometrías en un GeoDataFrame.
    #     """
    #     if not isinstance(gdf, gpd.GeoDataFrame):
    #         logger.error("La entrada a simplify_geometries no es un GeoDataFrame.")
    #         return gdf
    #
    #     logger.info(f"Simplificando geometrías con tolerancia: {tolerance}")
    #     # Aplicar simplify a cada geometría
    #     gdf_simplified = gdf.copy()
    #     gdf_simplified['geometry'] = gdf_simplified.geometry.simplify(tolerance, preserve_topology=True)
    #     logger.info("Simplificación completada.")
    #     return gdf_simplified

