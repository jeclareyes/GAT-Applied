import os
import geopandas as gpd
import pandas as pd
import re
import networkx as nx
import folium
import fiona
import matplotlib.pyplot as plt
from shapely.geometry import Point, LineString, MultiLineString
from shapely.ops import unary_union, linemerge
import math
from tqdm import tqdm
import multiprocessing as mp
import traceback

# Módulo de Reporte de Errores: Clase para almacenar y exportar logs.
class ErrorLogger:
    def __init__(self, log_file='reporte_errores.txt'):
        self.log_file = log_file
        self.logs = []

    def log(self, nivel, categoria, año, element_id, coords, descripcion, tolerancia=None):
        log_entry = {
            'nivel': nivel,
            'categoria': categoria,
            'año': año,
            'element_id': element_id,
            'coordenadas': coords,
            'descripcion': descripcion,
            'tolerancia': tolerancia
        }
        self.logs.append(log_entry)

    def exportar(self):
        with open(self.log_file, 'w', encoding='utf-8') as f:
            for entry in self.logs:
                linea = f"{entry['nivel']} | {entry['categoria']} | Año: {entry['año']} | ELEMENT_ID: {entry['element_id']} | Coords: {entry['coordenadas']} | Tolerancia: {entry['tolerancia']} | {entry['descripcion']}\n"
                f.write(linea)
        print(f"Reporte de errores exportado a {self.log_file}")


# Módulo de Lectura y Carga
def leer_geopackage(ruta, logger, año):
    """
    Lee el GeoPackage y verifica que la capa TRAFIK_DK_O_105_Trafik exista.
    Realiza el registro inicial de incidencias.
    """
    try:
        # Se listan las capas disponibles en el archivo usando fiona directamente
        capas = fiona.listlayers(ruta)

        if "TRAFIK_DK_O_105_Trafik" not in capas:
            logger.log("ERROR", "capa_inexistente", año, None, None,
                       f"La capa 'TRAFIK_DK_O_105_Trafik' no existe en {ruta}")
            return None

        # También podemos usar force_2d=True como en tu código que funciona
        gdf = gpd.read_file(ruta, layer="TRAFIK_DK_O_105_Trafik", force_2d=True)
        print(f"Archivo {ruta} cargado correctamente para el año {año}")
        return gdf
    except Exception as e:
        logger.log("ERROR", "lectura", año, None, None, f"Error al leer el archivo {ruta}: {e}")
        return None

# Módulo de Limpieza Geométrica
def limpiar_geometrias(gdf, logger, año, tolerancia=1.0, longitud_minima=20, distancia_maxima=20):
    """
    Limpia y corrige geometrías:
      - Valida y corrige con buffer(0) si son inválidas.
      - Intenta unir geometrías pequeñas con otras del mismo ELEMENT_ID si están cerca.
      - Funde atributos evitando valores nulos.
      - Registra el proceso con logger.
      - Asegura que las geometrías resultantes sean LineString y no MultiLineString.
    """

    total_original = len(gdf)
    corregidas = 0
    no_corregidas = 0

    indices_a_eliminar = []
    gdf = gdf.copy()
    eliminar = []
    for idx, row in gdf.iterrows():
        geom = row.geometry
        elem_id = row.get("ELEMENT_ID", None)

        if geom is None or geom.is_empty:
            logger.log("ERROR", "geometría", año, elem_id, None, "Geometría nula o vacía")
            indices_a_eliminar.append(idx)
            continue

        if not geom.is_valid:
            geom_corregida = geom.buffer(0)
            if geom_corregida.is_valid:
                gdf.at[idx, 'geometry'] = geom_corregida
                logger.log("WARNING", "geometría", año, elem_id, geom.centroid.coords[0],
                           "Geometría corregida con buffer(0)", tolerancia)
                corregidas += 1
            else:
                logger.log("ERROR", "geometría", año, elem_id, None,
                           "La geometría es inválida y no pudo corregirse")
                indices_a_eliminar.append(idx)
                no_corregidas += 1
                continue

    gdf.drop(index=indices_a_eliminar, inplace=True)
    gdf.reset_index(drop=True, inplace=True)

    # ------------------------------
    # TRATAMIENTO DE GEOMETRÍAS PEQUEÑAS
    # ------------------------------
    pequeños = gdf[gdf.geometry.length < longitud_minima]
    ya_usados = set()

    # Crear un DataFrame para almacenar las nuevas filas
    nuevas_filas = []

    for idx, row in pequeños.iterrows():
        if idx in ya_usados:
            continue

        geom = row.geometry
        elem_id = row["ELEMENT_ID"]

        candidatas = gdf[
            (gdf.index != idx) &
            (gdf["ELEMENT_ID"] == elem_id) &
            (gdf.geometry.distance(geom) <= distancia_maxima)
            ]

        if candidatas.empty:
            logger.log("WARNING", "geometría", año, elem_id, geom.centroid.coords[0],
                       f"No se encontró geometría cercana para unir (idx {idx})")
            no_corregidas += 1
            continue

        # Construir nueva geometría
        geometrías_a_unir = [geom] + list(candidatas.geometry)

        # Primero unimos las geometrías
        unida = unary_union(geometrías_a_unir)

        # Luego aplicamos linemerge para intentar convertir MultiLineString en LineString
        nueva_geom = linemerge(unida)

        # Si después de todo sigue siendo una MultiLineString, intentamos tomar la más larga
        if isinstance(nueva_geom, MultiLineString):
            # Encontrar la línea más larga
            max_length = 0
            longest_line = None
            for line in nueva_geom.geoms:
                if line.length > max_length:
                    max_length = line.length
                    longest_line = line

            if longest_line is not None:
                nueva_geom = longest_line
                logger.log("WARNING", "geometría", año, elem_id, geom.centroid.coords[0],
                           f"MultiLineString convertida a LineString (se tomó la línea más larga)")

        # Fusión de atributos evitando nulls
        columnas = gdf.columns.difference(['geometry'])
        nuevo_atributo = {}
        for col in columnas:
            valores = [row[col]] + list(candidatas[col])
            valores_validos = [v for v in valores if pd.notnull(v)]
            nuevo_atributo[col] = valores_validos[0] if valores_validos else None

        # Añadir la geometría al nuevo atributo
        nuevo_atributo['geometry'] = nueva_geom

        # Añadir a la lista de nuevas filas
        nuevas_filas.append(nuevo_atributo)

        # Eliminar filas anteriores
        indices = [idx] + list(candidatas.index)
        ya_usados.update(indices)
        gdf.drop(index=indices, inplace=True)

        logger.log("INFO", "geometría", año, elem_id, geom.centroid.coords[0],
                   f"Unidas {len(indices)} geometrías con longitud < {longitud_minima} m. Tipo resultante: {type(nueva_geom).__name__}")
        corregidas += len(indices)

    # Concatenar todas las nuevas filas de una sola vez
    if nuevas_filas:
        # Crear un DataFrame con todas las nuevas filas
        nuevas_gdf = pd.DataFrame(nuevas_filas)
        # Concatenar con el DataFrame original
        gdf = pd.concat([gdf, nuevas_gdf], ignore_index=True)

    gdf.reset_index(drop=True, inplace=True)

    # ------------------------------
    # VERIFICACIÓN FINAL DE TIPOS DE GEOMETRÍA
    # ------------------------------
    for idx, row in gdf.iterrows():
        geom = row.geometry
        elem_id = row.get("ELEMENT_ID", None)

        if isinstance(geom, MultiLineString):
            try:
                # Intentar convertir MultiLineString a LineString
                merged = linemerge(geom)

                # Si sigue siendo MultiLineString, tomar la parte más larga
                if isinstance(merged, MultiLineString):
                    max_length = 0
                    longest_line = None
                    for line in merged.geoms:
                        if line.length > max_length:
                            max_length = line.length
                            longest_line = line

                    if longest_line is not None:
                        gdf.at[idx, 'geometry'] = longest_line
                        logger.log("WARNING", "geometría", año, elem_id, geom.centroid.coords[0],
                                   "MultiLineString convertida a LineString (se tomó la línea más larga)")
                else:
                    gdf.at[idx, 'geometry'] = merged
                    logger.log("INFO", "geometría", año, elem_id, geom.centroid.coords[0],
                               "MultiLineString fusionada correctamente")
            except Exception as e:
                logger.log("ERROR", "geometría", año, elem_id, None,
                           f"Error al intentar fusionar MultiLineString: {str(e)}")

    # ------------------------------
    # ESTADÍSTICAS FINALES
    # ------------------------------
    print(f"Limpieza geométrica completada para el año {año}")
    print(f"Total original: {total_original}")
    print(f"Corregidas (buffer o unión): {corregidas}")
    print(f"No corregidas: {no_corregidas}")

    return gdf

#---------------------------
# Modulo de unión de capas y procesado
def unir_capas(lista_rutas, campos_temporales, tolerancia_union=1.0):
    """
    Lee múltiples GeoPackages, limpia geometrías, consolida atributos y unifica segmentos por ELEMENT_ID.
    Luego identifica nodos y asigna atributos start_node y end_node a segmentos.
    Retorna gdf_segmentos y gdf_nodos.
    """
    gdf_list = []
    logger = ErrorLogger()

    for ruta in lista_rutas:
        year_match = re.search(r"\d{4}", ruta)
        año = int(year_match.group()) if year_match else None
        if año is None:
            logger.log("ERROR", "formato_ruta", None, None, None,
                       f"No se pudo extraer el año de {ruta}")
            continue

        gdf = leer_geopackage(ruta, logger, año)
        if gdf is None:
            continue

        gdf = limpiar_geometrias(gdf, logger, año)
        gdf = consolidar_atributos(gdf, año, campos_temporales, logger)
        gdf_list.append(gdf)

    # Concatenar todo
    unified = gpd.GeoDataFrame(pd.concat(gdf_list, ignore_index=True), crs=gdf_list[0].crs)

    # Unir segmentos por ELEMENT_ID
    merged = merge_segments_by_element(unified, tolerancia_union)

    # Identificar nodos y asignar IDs
    gdf_segmentos, gdf_nodos = identify_network(merged)

    logger.exportar()
    return gdf_segmentos, gdf_nodos


def merge_segments_by_element(gdf, tolerance=1.0):
    """
    Agrupa por ELEMENT_ID, une geometrías que se superpongan o estén dentro de tolerance,
    preservando atributos no nulos.
    """
    resultados = []
    for elem_id, group in gdf.groupby('ELEMENT_ID'):
        # Buffer positivo para solapar proximidades
        buffered = group.geometry.buffer(tolerance)
        unioned = unary_union(buffered)
        # Deshacer buffer
        geom_union = unioned.buffer(-tolerance)
        # Asegurar LineString
        if isinstance(geom_union, MultiLineString):
            # tomar la línea más larga
            geom_union = max(geom_union.geoms, key=lambda l: l.length)
        # Consolidar atributos: primer valor no nulo
        attrs = {}
        for col in gdf.columns.difference(['geometry']):
            vals = group[col].dropna().tolist()
            attrs[col] = vals[0] if vals else None
        resultados.append({**attrs, 'geometry': geom_union})
    return gpd.GeoDataFrame(resultados, crs=gdf.crs)


def identify_network(gdf, precision=6):
    """
    Construye nodos con atributos de conectividad y asigna start/end node a segmentos.
    Retorna gdf_segmentos (con start_node, end_node) y gdf_nodos.
    """
    nodos_dict = {}
    # Primer pase para contar in/out grados
    for idx, row in gdf.iterrows():
        line = row.geometry
        start, end = line.coords[0], line.coords[-1]
        for pos, key_type in [(start, 'out'), (end, 'in')]:
            key = (round(pos[0], precision), round(pos[1], precision))
            rec = nodos_dict.setdefault(key, {'coords': pos, 'in': 0, 'out': 0, 'elements': []})
            rec[key_type] += 1
            rec['elements'].append(row.ELEMENT_ID)
    # Crear gdf_nodos con ID
    nodes = []
    for i, (key, info) in enumerate(nodos_dict.items()):
        conn = info['in'] + info['out']
        tipo = 'Intersección' if conn > 1 else 'Nodo_final'
        nodes.append({
            'node_id': i,
            'geometry': Point(info['coords']),
            'in_degree': info['in'],
            'out_degree': info['out'],
            'connectivity': conn,
            'segment_ids': list(set(info['elements'])),
            'count': len(info['elements']),
            'tipo': tipo
        })
    gdf_nodos = gpd.GeoDataFrame(nodes, crs=gdf.crs)
    # Mapear cada punto a node_id
    key_to_id = {key: node['node_id'] for key, node in zip(nodos_dict.keys(), nodes)}
    # Construir segmentos con start/end node
    segs = []
    for idx, row in gdf.iterrows():
        line = row.geometry
        start_key = (round(line.coords[0][0], precision), round(line.coords[0][1], precision))
        end_key = (round(line.coords[-1][0], precision), round(line.coords[-1][1], precision))
        attrs = row.to_dict()
        attrs['start_node'] = key_to_id[start_key]
        attrs['end_node'] = key_to_id[end_key]
        segs.append(attrs)
    gdf_segs = gpd.GeoDataFrame(segs, geometry='geometry', crs=gdf.crs)
    return gdf_segs, gdf_nodos


# Módulo de Consolidación de Atributos

def normalizar_texto(texto):
    """
    Normaliza texto convirtiendo caracteres especiales suecos a equivalentes básicos.
    """
    reemplazos = {
        'ä': 'a', 'Ä': 'A',
        'ö': 'o', 'Ö': 'O',
        'å': 'a', 'Å': 'A'
    }
    for especial, normal in reemplazos.items():
        texto = texto.replace(especial, normal)
    return texto


def encontrar_mejor_coincidencia(campo, columnas):
    """
    Encuentra la mejor coincidencia para un campo en la lista de columnas.
    Usa normalización de texto para hacer coincidencias más flexibles.
    """
    campo_norm = normalizar_texto(campo.lower())

    # Buscar coincidencia exacta primero
    for col in columnas:
        if normalizar_texto(col.lower()) == campo_norm:
            return col

    # Si no hay coincidencia exacta, buscar coincidencia parcial
    posibles_coincidencias = []
    for col in columnas:
        col_norm = normalizar_texto(col.lower())
        # Verificar si es una subcadena o viceversa
        if campo_norm in col_norm or col_norm in campo_norm:
            posibles_coincidencias.append((col, max(len(campo_norm), len(col_norm)) -
                                           abs(len(campo_norm) - len(col_norm))))

    # Ordenar por puntaje (mayor es mejor)
    if posibles_coincidencias:
        posibles_coincidencias.sort(key=lambda x: x[1], reverse=True)
        return posibles_coincidencias[0][0]

    return None

def consolidar_atributos(gdf, año, campos_temporales, logger):
    """
    Agrega un sufijo (año) a los campos definidos para que sean únicos al consolidar.
    Implementa búsqueda flexible para campos con caracteres especiales suecos.
    """
    gdf = gdf.copy()
    sufijo = f"_{año}"

    # Procesar cada campo temporal
    for campo in campos_temporales:
        # Verificar si el campo existe directamente
        if campo in gdf.columns:
            nuevo_nombre = campo + sufijo
            gdf = gdf.rename(columns={campo: nuevo_nombre})
        else:
            # Buscar la mejor coincidencia
            mejor_coincidencia = encontrar_mejor_coincidencia(campo, gdf.columns)

            if mejor_coincidencia:
                nuevo_nombre = mejor_coincidencia + sufijo
                gdf = gdf.rename(columns={mejor_coincidencia: nuevo_nombre})
                logger.log("INFO", "atributo", año, None, None,
                           f"Campo '{campo}' coincidido con '{mejor_coincidencia}'")
            else:
                logger.log("WARNING", "atributo", año, None, None,
                           f"El campo '{campo}' no está presente ni se encontró coincidencia")

    return gdf

# Módulo de Identificación de Nodos
def identificar_nodos(gdf):
    """
    A partir de un GeoDataFrame de segmentos, identifica:
      - Intersecciones (puntos donde convergen 2 o más segmentos).
      - Nodos finales (inicios y fines de cada segmento).
    Devuelve dos GeoDataFrames: uno para nodos y otro para segmentos no conectados.
    """
    nodos = []
    nodos_dict = {}  # Para agrupar puntos y luego clasificar intersecciones
    segmentos_no_conectados = []

    for idx, row in gdf.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue

        # Verificar que la geometría tiene coordenadas
        if len(geom.coords) < 2:
            continue

        # Se obtiene el primer y último punto del segmento
        inicio = geom.coords[0]
        fin = geom.coords[-1]

        # Usamos una tupla redondeada como clave para evitar problemas de precisión
        inicio_key = (round(inicio[0], 6), round(inicio[1], 6))
        fin_key = (round(fin[0], 6), round(fin[1], 6))

        for punto_key, punto in [(inicio_key, inicio), (fin_key, fin)]:
            if punto_key in nodos_dict:
                nodos_dict[punto_key]['count'] += 1
                nodos_dict[punto_key]['ELEMENT_ID'].append(row.get("ELEMENT_ID", None))
            else:
                nodos_dict[punto_key] = {
                    'count': 1,
                    'coords': punto,  # Guardamos las coordenadas originales
                    'ELEMENT_ID': [row.get("ELEMENT_ID", None)]
                }

    # Identificar segmentos no conectados (aquellos cuyos nodos tienen count=1)
    for idx, row in gdf.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty or len(geom.coords) < 2:
            continue

        inicio = geom.coords[0]
        fin = geom.coords[-1]
        inicio_key = (round(inicio[0], 6), round(inicio[1], 6))
        fin_key = (round(fin[0], 6), round(fin[1], 6))

        if nodos_dict[inicio_key]['count'] == 1 and nodos_dict[fin_key]['count'] == 1:
            segmentos_no_conectados.append({
                'geometry': geom,
                'ELEMENT_ID': row.get("ELEMENT_ID", None)
            })

    # Clasificar nodos
    for coords_key, info in nodos_dict.items():
        if info['count'] > 1:
            tipo = "Intersección"
        else:
            tipo = "Nodo_final"
        nodos.append({
            'geometry': Point(info['coords']),
            'tipo': tipo,
            'conteo': info['count'],
            'elementos': info['ELEMENT_ID']
        })

    gdf_nodos = gpd.GeoDataFrame(nodos, geometry='geometry', crs=gdf.crs)
    gdf_segmentos_aislados = gpd.GeoDataFrame(segmentos_no_conectados, geometry='geometry',
                                              crs=gdf.crs) if segmentos_no_conectados else None

    return gdf_nodos, gdf_segmentos_aislados


# Módulo de Construcción del Grafo
def construir_grafo(gdf, gdf_nodos):
    """
    Crea un grafo dirigido utilizando NetworkX.
      - Los nodos se toman del GeoDataFrame de nodos.
      - Las aristas corresponden a cada segmento, asignándoles atributos adicionales.
      - Se respeta la dirección de circulación según el atributo 'DIRECTION'.
    """
    G = nx.DiGraph()

    # Crear un diccionario para búsqueda rápida de nodos por coordenadas
    nodos_por_coords = {}
    for idx, row in gdf_nodos.iterrows():
        # Usamos un ID numérico para los nodos en lugar de WKT
        node_id = f"node_{idx}"
        coords = (row.geometry.x, row.geometry.y)
        coords_key = (round(coords[0], 6), round(coords[1], 6))

        # Almacenar en diccionario para buscar por coordenadas
        nodos_por_coords[coords_key] = node_id

        # Agregar al grafo - Convertir coordenadas a strings para compatibilidad con GraphML
        G.add_node(node_id,
                   tipo=row['tipo'],
                   coord_x=float(coords[0]),  # Guardar como números separados
                   coord_y=float(coords[1]))  # en lugar de una tupla

    # Agregar aristas tomando cada segmento
    for idx, row in gdf.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty or len(geom.coords) < 2:
            continue

        # Obtener coordenadas de inicio y fin
        inicio = geom.coords[0]
        fin = geom.coords[-1]

        # Crear claves redondeadas para búsqueda
        inicio_key = (round(inicio[0], 6), round(inicio[1], 6))
        fin_key = (round(fin[0], 6), round(fin[1], 6))

        # Buscar los IDs de nodos correspondientes
        if inicio_key not in nodos_por_coords or fin_key not in nodos_por_coords:
            continue  # Saltar si no encontramos los nodos

        inicio_id = nodos_por_coords[inicio_key]
        fin_id = nodos_por_coords[fin_key]

        # Verificar la dirección, por ejemplo, suponiendo que el campo 'DIRECTION' indica "forward" o "reverse"
        direccion = row.get("DIRECTION", "forward")
        if direccion.lower() == "reverse":
            source, target = fin_id, inicio_id
        else:  # Por defecto, forward
            source, target = inicio_id, fin_id

        # Filtrar atributos para evitar problemas con la geometría y convertir tipos no compatibles
        atributos = {}
        for k, v in row.items():
            if k != 'geometry':
                # Convertir tipos no compatibles con GraphML
                if isinstance(v, tuple):
                    # No agregar tuplas
                    continue
                elif isinstance(v, (list, dict)):
                    # Convertir a string si es una lista o diccionario
                    atributos[k] = str(v)
                else:
                    atributos[k] = v

        # Agregar ID del segmento como atributo
        atributos['segment_id'] = f"segment_{idx}"
        G.add_edge(source, target, **atributos)

    return G

# Módulo de Análisis Topológico
def analizar_grafo(G):
    """
    Realiza análisis topológico sobre el grafo:
      - Determina componentes conexas, nodos aislados.
      - Calcula grados de entrada y salida.
      - Detecta ciclos.
    """
    reporte = {}
    # Componentes (en un grafo dirigido se pueden analizar fuertemente o débilmente conexos)
    componentes = list(nx.weakly_connected_components(G))
    reporte['n_componentes'] = len(componentes)
    reporte['tamaño_componentes'] = [len(comp) for comp in componentes]

    # Nodos aislados (sin aristas de entrada y salida)
    aislados = [n for n in G.nodes() if G.in_degree(n) == 0 and G.out_degree(n) == 0]
    reporte['n_aislados'] = len(aislados)

    # Grados de entrada y salida
    grados_entrada = dict(G.in_degree())
    grados_salida = dict(G.out_degree())
    reporte['max_grado_entrada'] = max(grados_entrada.values()) if grados_entrada else 0
    reporte['max_grado_salida'] = max(grados_salida.values()) if grados_salida else 0

    # Detección de ciclos: se limita a ciclos pequeños para evitar problemas con grafos grandes
    try:
        # Intentamos buscar ciclos, limitando a 1000 ciclos como máximo
        ciclos = []
        for i, ciclo in enumerate(nx.simple_cycles(G)):
            if i >= 1000:
                break
            ciclos.append(ciclo)
        reporte['n_ciclos'] = len(ciclos)
    except nx.NetworkXNoCycle:
        reporte['n_ciclos'] = 0
    except Exception as e:
        reporte['error_ciclos'] = str(e)
        reporte['n_ciclos'] = "Error al calcular"

    return reporte


# Módulo de Visualización
def visualizar_grafo(G, gdf=None):
    """
    Se pueden generar dos tipos de visualizaciones:
      1. Grafo con NetworkX utilizando matplotlib.
      2. Mapa interactivo con Folium, si se provee un GeoDataFrame.
    """
    # Verificar que el grafo tiene nodos
    if len(G.nodes()) == 0:
        print("El grafo no tiene nodos para visualizar.")
        return

    # Visualización con NetworkX y matplotlib
    plt.figure(figsize=(12, 8))
    try:
        # Usar posiciones geográficas si están disponibles - ahora usando coord_x y coord_y
        pos = {node: (data.get('coord_x', 0), data.get('coord_y', 0))
               for node, data in G.nodes(data=True)}
        nx.draw(G, pos, with_labels=False, node_size=50, arrowstyle='->', arrowsize=10,
                node_color='blue', edge_color='gray')
        plt.title("Grafo Vial")
        plt.savefig("grafo_vial.png")
        print("Grafo guardado como grafo_vial.png")
        plt.close()
    except Exception as e:
        print(f"Error al visualizar con matplotlib: {e}")

    # Visualización con Folium: si se tiene un GeoDataFrame con nodos o segmentos
    if gdf is not None and not gdf.empty:
        try:
            # Centrar el mapa en la media de las coordenadas
            centro = [gdf.geometry.centroid.y.mean(), gdf.geometry.centroid.x.mean()]
            m = folium.Map(location=centro, zoom_start=12)

            # Agregar cada geometría
            for _, row in gdf.iterrows():
                if row.geometry is None or row.geometry.is_empty:
                    continue

                # Estilo según el tipo (si es un nodo)
                if 'tipo' in row:
                    if row['tipo'] == "Intersección":
                        color = 'red'
                    else:
                        color = 'blue'
                else:
                    color = 'green'  # Por defecto para segmentos

                # Añadir al mapa
                folium.GeoJson(
                    row.geometry.__geo_interface__,
                    style_function=lambda x, color=color: {
                        'color': color,
                        'weight': 2,
                        'fillOpacity': 0.7
                    }
                ).add_to(m)

            # Mostrar el mapa
            m.save("mapa_interactivo.html")
            print("Mapa interactivo exportado a mapa_interactivo.html")
        except Exception as e:
            print(f"Error al crear el mapa con Folium: {e}")

# Exportar red a geopackage
# Estas funciones deben estar FUERA de la función principal para que puedan ser pickleable
def procesar_conexiones(args):
    """Procesa las conexiones de un segmento con sus nodos."""
    idx, row, nodos_por_coords = args
    if row.geometry is None or row.geometry.is_empty or len(row.geometry.coords) < 2:
        return None, None, None

    geom = row.geometry
    inicio = geom.coords[0]
    fin = geom.coords[-1]
    inicio_key = (round(inicio[0], 6), round(inicio[1], 6))
    fin_key = (round(fin[0], 6), round(fin[1], 6))

    # Buscar los IDs de nodos correspondientes
    inicio_id = nodos_por_coords.get(inicio_key, None)
    fin_id = nodos_por_coords.get(fin_key, None)

    # Verificar la dirección
    direccion = row.get("DIRECTION", "forward")
    segment_id = f"segment_{idx}"

    if direccion.lower() == "reverse":
        return fin_id, inicio_id, segment_id
    else:  # Por defecto, forward
        return inicio_id, fin_id, segment_id


def procesar_nodo(args):
    """Procesa los atributos de un nodo desde el grafo."""
    idx, row, grafo = args
    node_id = f"node_{idx}"

    # Obtener grados y conectividad desde el grafo
    if node_id in grafo:
        in_degree = grafo.in_degree(node_id)
        out_degree = grafo.out_degree(node_id)
        conectividad = in_degree + out_degree
    else:
        in_degree = 0
        out_degree = 0
        conectividad = 0

    return node_id, in_degree, out_degree, conectividad


def procesar_marcadores(args):
    """Procesa los marcadores direccionales para un segmento."""
    idx, row = args
    marcadores_segmento = []

    if row.geometry is None or row.geometry.is_empty or len(row.geometry.coords) < 2:
        return marcadores_segmento

    geom = row.geometry
    direccion = row.get("DIRECTION", "forward")

    # Crear puntos a 1/3 y 2/3 del recorrido para marcadores direccionales
    punto1 = geom.interpolate(0.33, normalized=True)
    punto2 = geom.interpolate(0.67, normalized=True)

    # Calcular ángulo para cada punto
    for punto in [punto1, punto2]:
        min_dist = float('inf')
        closest_i = 0

        # Encontrar el índice del segmento más cercano
        for i in range(len(geom.coords) - 1):
            segmento = LineString([geom.coords[i], geom.coords[i + 1]])
            dist = segmento.distance(punto)
            if dist < min_dist:
                min_dist = dist
                closest_i = i

        # Usar este segmento para calcular el ángulo
        p1 = geom.coords[closest_i]
        p2 = geom.coords[closest_i + 1]

        # Si es reverse, invertir la dirección del ángulo
        if direccion.lower() == "reverse":
            p1, p2 = p2, p1

        # Calcular ángulo
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        angulo = math.degrees(math.atan2(dy, dx)) - 90

        marcadores_segmento.append({
            'geometry': punto,
            'segment_id': row.get('segment_id', f"segment_{idx}"),
            'angulo': angulo,
            'tipo_marker': 'flecha_direccional',
            'direccion': direccion
        })

    return marcadores_segmento


def exportar_red_a_geopackage(gdf_consolidado, gdf_nodos, grafo, ruta_salida="red_vial_completa.gpkg",
                              num_processes=None):
    """
    Exporta la red vial completa a un archivo GeoPackage con indicadores de dirección.
    Versión paralelizada para mejorar el rendimiento.

    Args:
        gdf_consolidado (GeoDataFrame): GeoDataFrame con los segmentos de red.
        gdf_nodos (GeoDataFrame): GeoDataFrame con los nodos de la red.
        grafo (DiGraph): Grafo dirigido de NetworkX con la topología de la red.
        ruta_salida (str): Ruta al archivo GeoPackage de salida.
        num_processes (int, optional): Número de procesos a utilizar. Por defecto usa todos los núcleos disponibles.

    Returns:
        bool: True si la exportación fue exitosa, False en caso contrario.
    """
    # Si no se especifica el número de procesos, usar todos los núcleos disponibles
    if num_processes is None:
        num_processes = mp.cpu_count()

    try:
        # Crear copia para no modificar los originales
        gdf_segmentos = gdf_consolidado.copy()
        gdf_nodos_export = gdf_nodos.copy()

        # Añadir información de conectividad a los segmentos desde el grafo
        # Primero, crear diccionarios que mapeen nodos a segmentos de inicio y fin
        nodos_por_coords = {}
        for idx, row in gdf_nodos_export.iterrows():
            node_id = f"node_{idx}"
            coords = (row.geometry.x, row.geometry.y)
            coords_key = (round(coords[0], 6), round(coords[1], 6))
            nodos_por_coords[coords_key] = node_id

        # PARALELIZACIÓN 1: Procesar conexiones de segmentos
        # Preparar argumentos para incluir el diccionario nodos_por_coords
        args_list = [(idx, row, nodos_por_coords) for idx, row in gdf_segmentos.iterrows()]

        print(f"Procesando conexiones de segmentos en paralelo con {num_processes} procesos...")
        with mp.Pool(processes=num_processes) as pool:
            resultados = list(tqdm(
                pool.imap(procesar_conexiones, args_list),
                total=len(args_list),
                desc="Procesando conexiones"
            ))

        # Desempaquetar resultados
        nodo_origen, nodo_destino, segment_ids = zip(*resultados)

        # Agregar columnas al GeoDataFrame de segmentos
        gdf_segmentos['node_from'] = nodo_origen
        gdf_segmentos['node_to'] = nodo_destino
        gdf_segmentos['segment_id'] = segment_ids

        # NUEVO: Agregar indicadores direccionales (no es necesario paralelizar esto)
        gdf_segmentos['color_inicio'] = "#3333FF"  # Azul oscuro
        gdf_segmentos['color_medio'] = "#33FFFF"  # Cian
        gdf_segmentos['color_fin'] = "#FF3333"  # Rojo claro

        # PARALELIZACIÓN 2: Procesar atributos de nodos
        # Incluir el grafo en los argumentos
        args_list = [(idx, row, grafo) for idx, row in gdf_nodos_export.iterrows()]

        print(f"Procesando atributos de nodos en paralelo con {num_processes} procesos...")
        with mp.Pool(processes=num_processes) as pool:
            resultados = list(tqdm(
                pool.imap(procesar_nodo, args_list),
                total=len(args_list),
                desc="Procesando nodos"
            ))

        # Desempaquetar resultados
        node_ids, in_degrees, out_degrees, conectividades = zip(*resultados)

        # Agregar columnas al GeoDataFrame de nodos
        gdf_nodos_export['node_id'] = node_ids
        gdf_nodos_export['in_degree'] = in_degrees
        gdf_nodos_export['out_degree'] = out_degrees
        gdf_nodos_export['conectividad'] = conectividades

        # PARALELIZACIÓN 3: Crear marcadores direccionales
        args_list = [(idx, row) for idx, row in gdf_segmentos.iterrows()]

        print(f"Procesando marcadores direccionales en paralelo con {num_processes} procesos...")
        with mp.Pool(processes=num_processes) as pool:
            resultados = list(tqdm(
                pool.imap(procesar_marcadores, args_list),
                total=len(args_list),
                desc="Procesando marcadores"
            ))

        # Aplanar la lista de listas de marcadores
        markers = [marker for sublist in resultados for marker in sublist]

        # Exportar a GeoPackage
        print(f"Exportando segmentos a {ruta_salida}...")
        gdf_segmentos.to_file(ruta_salida, layer="segmentos_red", driver="GPKG")

        print(f"Exportando nodos a {ruta_salida}...")
        gdf_nodos_export.to_file(ruta_salida, layer="nodos_red", driver="GPKG", mode="a")

        # Crear y exportar GeoDataFrame con los marcadores direccionales
        if markers:
            gdf_markers = gpd.GeoDataFrame(markers, geometry='geometry', crs=gdf_segmentos.crs)
            print(f"Exportando marcadores direccionales a {ruta_salida}...")
            gdf_markers.to_file(ruta_salida, layer="marcadores_direccionales", driver="GPKG", mode="a")

        print(f"Red vial completa exportada exitosamente a {ruta_salida}")
        return True

    except Exception as e:
        print(f"Error al exportar la red vial a GeoPackage: {e}")
        traceback.print_exc()
        return False

def exportar_red_a_geopackage_old(gdf_consolidado, gdf_nodos, grafo, ruta_salida="red_vial_completa.gpkg"):
    """
    Exporta la red vial completa a un archivo GeoPackage con indicadores de dirección.
    Incluye:
      - Segmentos de red (linestrings) con sus atributos y topología
      - Nodos/intersecciones (puntos) con sus atributos
      - Indicadores direccionales para visualizar el sentido de la vía

    Args:
        gdf_consolidado (GeoDataFrame): GeoDataFrame con los segmentos de red.
        gdf_nodos (GeoDataFrame): GeoDataFrame con los nodos de la red.
        grafo (DiGraph): Grafo dirigido de NetworkX con la topología de la red.
        ruta_salida (str): Ruta al archivo GeoPackage de salida.

    Returns:
        bool: True si la exportación fue exitosa, False en caso contrario.
    """
    try:
        # Crear copia para no modificar los originales
        gdf_segmentos = gdf_consolidado.copy()
        gdf_nodos_export = gdf_nodos.copy()

        # Añadir información de conectividad a los segmentos desde el grafo
        # Primero, crear diccionarios que mapeen nodos a segmentos de inicio y fin
        nodos_por_coords = {}
        for idx, row in gdf_nodos_export.iterrows():
            node_id = f"node_{idx}"
            coords = (row.geometry.x, row.geometry.y)
            coords_key = (round(coords[0], 6), round(coords[1], 6))
            nodos_por_coords[coords_key] = node_id

        # Agregar columnas de nodo origen y destino a los segmentos
        nodo_origen = []
        nodo_destino = []
        segment_ids = []

        for idx, row in tqdm(gdf_segmentos.iterrows(), total=len(gdf_segmentos), desc="Procesando segmentos"):
            geom = row.geometry
            if geom is None or geom.is_empty or len(geom.coords) < 2:
                nodo_origen.append(None)
                nodo_destino.append(None)
                segment_ids.append(None)
                continue

            inicio = geom.coords[0]
            fin = geom.coords[-1]
            inicio_key = (round(inicio[0], 6), round(inicio[1], 6))
            fin_key = (round(fin[0], 6), round(fin[1], 6))

            # Buscar los IDs de nodos correspondientes
            inicio_id = nodos_por_coords.get(inicio_key, None)
            fin_id = nodos_por_coords.get(fin_key, None)

            # Verificar la dirección
            direccion = row.get("DIRECTION", "forward")
            if direccion.lower() == "reverse":
                nodo_origen.append(fin_id)
                nodo_destino.append(inicio_id)
            else:  # Por defecto, forward
                nodo_origen.append(inicio_id)
                nodo_destino.append(fin_id)

            segment_ids.append(f"segment_{idx}")

        # Agregar columnas al GeoDataFrame de segmentos
        gdf_segmentos['node_from'] = nodo_origen
        gdf_segmentos['node_to'] = nodo_destino
        gdf_segmentos['segment_id'] = segment_ids

        # NUEVO: Agregar indicadores direccionales
        # 1. Atributo de gradiente de color
        color_inicio = []
        color_medio = []
        color_fin = []
        for idx, row in tqdm(gdf_segmentos.iterrows(), total=len(gdf_segmentos), desc="Procesando segmentos"):
            direccion = row.get("DIRECTION", "forward")
            # Independientemente de la dirección, establecemos un gradiente
            # Esto facilita la visualización en SIG: del inicio al final de la línea
            color_inicio.append("#3333FF")  # Azul oscuro
            color_medio.append("#33FFFF")  # Cian
            color_fin.append("#FF3333")  # Rojo claro
        gdf_segmentos['color_inicio'] = color_inicio
        gdf_segmentos['color_medio'] = color_medio
        gdf_segmentos['color_fin'] = color_fin

        # Agregar atributos de grado y centralidad a los nodos desde el grafo
        node_ids = []
        in_degrees = []
        out_degrees = []
        conectividades = []

        for idx, row in gdf_nodos_export.iterrows():
            node_id = f"node_{idx}"
            node_ids.append(node_id)

            # Obtener grados y conectividad desde el grafo
            if node_id in grafo:
                in_degrees.append(grafo.in_degree(node_id))
                out_degrees.append(grafo.out_degree(node_id))
                conectividades.append(grafo.in_degree(node_id) + grafo.out_degree(node_id))
            else:
                in_degrees.append(0)
                out_degrees.append(0)
                conectividades.append(0)

        # Agregar columnas al GeoDataFrame de nodos
        gdf_nodos_export['node_id'] = node_ids
        gdf_nodos_export['in_degree'] = in_degrees
        gdf_nodos_export['out_degree'] = out_degrees
        gdf_nodos_export['conectividad'] = conectividades

        # NUEVO: Crear capa de marcadores direccionales
        markers = []
        for idx, row in tqdm(gdf_segmentos.iterrows(), total=len(gdf_segmentos), desc="Procesando segmentos"):
            if row.geometry is None or row.geometry.is_empty or len(row.geometry.coords) < 2:
                continue

            geom = row.geometry
            direccion = row.get("DIRECTION", "forward")

            # Crear puntos a 1/3 y 2/3 del recorrido para marcadores direccionales
            punto1 = geom.interpolate(0.33, normalized=True)
            punto2 = geom.interpolate(0.67, normalized=True)

            # Calcular ángulo de la línea en este punto
            if len(geom.coords) >= 2:
                # Aproximación para calcular el ángulo en el punto
                # Encontrar los segmentos más cercanos a estos puntos
                for punto in [punto1, punto2]:
                    min_dist = float('inf')
                    closest_i = 0

                    # Encontrar el índice del segmento más cercano
                    for i in range(len(geom.coords) - 1):
                        segmento = LineString([geom.coords[i], geom.coords[i + 1]])
                        dist = segmento.distance(punto)
                        if dist < min_dist:
                            min_dist = dist
                            closest_i = i

                    # Usar este segmento para calcular el ángulo
                    p1 = geom.coords[closest_i]
                    p2 = geom.coords[closest_i + 1]

                    # Si es reverse, invertir la dirección del ángulo
                    if direccion.lower() == "reverse":
                        p1, p2 = p2, p1

                    # Calcular ángulo
                    dx = p2[0] - p1[0]
                    dy = p2[1] - p1[1]
                    angulo = math.degrees(math.atan2(dy, dx)) - 90

                    markers.append({
                        'geometry': punto,
                        'segment_id': row.get('segment_id', f"segment_{idx}"),
                        'angulo': angulo,
                        'tipo_marker': 'flecha_direccional',
                        'direccion': direccion
                    })

        # Exportar a GeoPackage
        print(f"Exportando segmentos a {ruta_salida}...")
        gdf_segmentos.to_file(ruta_salida, layer="segmentos_red", driver="GPKG")

        print(f"Exportando nodos a {ruta_salida}...")
        gdf_nodos_export.to_file(ruta_salida, layer="nodos_red", driver="GPKG", mode="a")

        # Crear y exportar GeoDataFrame con los marcadores direccionales
        if markers:
            import geopandas as gpd
            gdf_markers = gpd.GeoDataFrame(markers, geometry='geometry', crs=gdf_segmentos.crs)
            print(f"Exportando marcadores direccionales a {ruta_salida}...")
            gdf_markers.to_file(ruta_salida, layer="marcadores_direccionales", driver="GPKG", mode="a")

        print(f"Red vial completa exportada exitosamente a {ruta_salida}")
        return True

    except Exception as e:
        print(f"Error al exportar la red vial a GeoPackage: {e}")
        import traceback
        traceback.print_exc()
        return False

# Exportar grafo
def exportar_grafo(grafo, prefix="grafo_vial"):
    """
    Exporta el grafo en formatos GraphML y JSON,
    asegurando la compatibilidad de tipos de datos.
    """
    # Para GraphML necesitamos asegurar que todos los valores sean tipos simples
    grafo_export = grafo.copy()

    # Convertir cualquier atributo no compatible en los nodos
    for node, attrs in grafo.nodes(data=True):
        for key, value in list(attrs.items()):
            # Tratar con tipos no compatibles con GraphML
            if value is None:
                grafo_export.nodes[node][key] = "None"  # Convertir None a string "None"
            elif isinstance(value, (tuple, list, dict, set)):
                grafo_export.nodes[node][key] = str(value)

    # Convertir cualquier atributo no compatible en las aristas
    for u, v, attrs in grafo.edges(data=True):
        for key, value in list(attrs.items()):
            # Tratar con valores None o tipos no compatibles con GraphML
            if value is None:
                grafo_export.edges[u, v][key] = "None"  # Convertir None a string "None"
            elif isinstance(value, (tuple, list, dict, set)):
                grafo_export.edges[u, v][key] = str(value)

    try:
        # Intentar exportar en GraphML
        try:
            nx.write_graphml(grafo_export, f"{prefix}.graphml")
            print(f"Grafo exportado en formato GraphML: {prefix}.graphml")
        except Exception as e:
            print(f"No se pudo exportar en formato GraphML: {e}")
            print("Intentando exportar en formato GEXF como alternativa...")
            # GEXF es una alternativa robusta a GraphML
            nx.write_gexf(grafo_export, f"{prefix}.gexf")
            print(f"Grafo exportado en formato GEXF: {prefix}.gexf")

        # Para exportar en JSON, se puede convertir el grafo a diccionario
        import json
        grafo_json = nx.readwrite.json_graph.node_link_data(grafo_export)
        with open(f"{prefix}.json", "w", encoding="utf-8") as f:
            json.dump(grafo_json, f, ensure_ascii=False, indent=2)
        print(f"Grafo exportado en formato JSON: {prefix}.json")

        return True
    except Exception as e:
        print(f"Error al exportar el grafo: {e}")
        import traceback
        traceback.print_exc()
        return False

# Función principal que integra todos los módulos
def main():
    # Carpeta de Geopackages
    ruta = r"G:\My Drive\MSc\Thesis\ÅDT\Data\Lastkajen\Trafik_Yearly"

    # Lista de rutas a los archivos GeoPackage (ejemplo)def exportar_grafo(grafo, prefix="grafo_vial"):
    #     """
    #     Exporta el grafo en formatos GraphML y JSON,
    #     asegurando la compatibilidad de tipos de datos.
    #     """
    #     # Para GraphML necesitamos asegurar que todos los valores sean tipos simples
    #     grafo_export = grafo.copy()
    #
    #     # Convertir cualquier atributo no compatible en los nodos
    #     for node, attrs in grafo.nodes(data=True):
    #         for key, value in list(attrs.items()):
    #             # Tratar con tipos no compatibles con GraphML
    #             if value is None:
    #                 grafo_export.nodes[node][key] = "None"  # Convertir None a string "None"
    #             elif isinstance(value, (tuple, list, dict, set)):
    #                 grafo_export.nodes[node][key] = str(value)
    #
    #     # Convertir cualquier atributo no compatible en las aristas
    #     for u, v, attrs in grafo.edges(data=True):
    #         for key, value in list(attrs.items()):
    #             # Tratar con valores None o tipos no compatibles con GraphML
    #             if value is None:
    #                 grafo_export.edges[u, v][key] = "None"  # Convertir None a string "None"
    #             elif isinstance(value, (tuple, list, dict, set)):
    #                 grafo_export.edges[u, v][key] = str(value)
    #
    #     try:
    #         # Intentar exportar en GraphML
    #         try:
    #             nx.write_graphml(grafo_export, f"{prefix}.graphml")
    #             print(f"Grafo exportado en formato GraphML: {prefix}.graphml")
    #         except Exception as e:
    #             print(f"No se pudo exportar en formato GraphML: {e}")
    #             print("Intentando exportar en formato GEXF como alternativa...")
    #             # GEXF es una alternativa robusta a GraphML
    #             nx.write_gexf(grafo_export, f"{prefix}.gexf")
    #             print(f"Grafo exportado en formato GEXF: {prefix}.gexf")
    #
    #         # Para exportar en JSON, se puede convertir el grafo a diccionario
    #         import json
    #         grafo_json = nx.readwrite.json_graph.node_link_data(grafo_export)
    #         with open(f"{prefix}.json", "w", encoding="utf-8") as f:
    #             json.dump(grafo_json, f, ensure_ascii=False, indent=2)
    #         print(f"Grafo exportado en formato JSON: {prefix}.json")
    #
    #         return True
    #     except Exception as e:
    #         print(f"Error al exportar el grafo: {e}")
    #         import traceback
    #         traceback.print_exc()
    #         return False
    archivos = [
        "Trafik_ÅDT_01_01_2020_1673.gpkg",
        "Trafik_ÅDT_01_01_2021_1674.gpkg",
        "Trafik_ÅDT_01_01_2022_1675.gpkg",
        # Agregar las demás rutas (aprox. 20 archivos)
    ]

    lista_rutas = [os.path.join(ruta, archivo) for archivo in archivos]

    # Definir los campos que deben consolidarse con sufijo de año
    campos_temporales = [
        'Adt_axelpar', 'Adt_samtliga_fordon', 'Adt_tunga_fordon',
        'Adt_latta_fordon_06_18', 'Adt_latta_fordon_18_22', 'Adt_latta_fordon_22_06',
        'Adt_medeltunga_fordon_06_18', 'Adt_medeltunga_fordon_18_22', 'Adt_medeltunga_fordon_22_06',
        'Adt_tunga_fordon_06_18', 'Adt_tunga_fordon_18_22', 'Adt_tunga_fordon_22_06',
        'Avsnittsidentitet',
        'Matarsperiod', 'Matmetod',
        'Mc_floden',
        'Osakerhet_axelpar', 'Osakerhet_samtliga_fordon', 'Osakerhet_tunga_fordon'
    ]

    try:
        # Paso 1: Unificar capas
        print("Iniciando la consolidación de capas...")
        gdf_consolidado = unir_capas(lista_rutas, campos_temporales)
        if gdf_consolidado is None or gdf_consolidado.empty:
            print("No se pudo consolidar la información de las capas. Revise los errores en el log.")
            return

        # Paso 2: Identificación de nodos e intersecciones
        print("Identificando nodos e intersecciones...")
        gdf_nodos, gdf_segmentos_aislados = identificar_nodos(gdf_consolidado)

        # Paso 3: Construcción del grafo vial
        print("Construyendo el grafo vial dirigido...")
        grafo = construir_grafo(gdf_consolidado, gdf_nodos)

        # Paso 4: Análisis topológico del grafo
        print("Realizando análisis topológico...")
        reporte_topologico = analizar_grafo(grafo)
        print("Reporte Topológico:", reporte_topologico)

        # Paso 5: Exportación del grafo en formatos GraphML y JSON
        print("Exportando el grafo vial...")
        exportar_grafo(grafo)

        # Paso 6: Exportar la red completa a GeoPackage
        print("Exportando la red vial completa a GeoPackage...")
        exportar_red_a_geopackage(gdf_consolidado, gdf_nodos, grafo, "red_vial_completa.gpkg")

        # Paso 7: Visualización de la red
        print("Visualizando la red vial...")
        visualizar_grafo(grafo, gdf_nodos)

        # Exportar segmentos aislados si existen
        if gdf_segmentos_aislados is not None and not gdf_segmentos_aislados.empty:
            gdf_segmentos_aislados.to_file("segmentos_aislados.gpkg", layer="segmentos_aislados", driver="GPKG")
            print("Segmentos no conectados exportados.")

        print("Proceso completado exitosamente.")

    except Exception as e:
        print(f"Error en la ejecución principal: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()