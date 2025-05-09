# pipeline.py: script principal para correr all el flujo

import time
import argparse
import sys
import logging
from configs.settings import Paths, Pipeline, Filenames, Regex, Fields, Layer
from export_utils import GeoPackageExporter, ReportLogger
from data_ingestion import GeoPackageHandler
from network_processing import *
from graph_tools import GraphBuilder, GraphAnalyzer, GraphExporter
from map_visualization import FoliumMapBuilder
from project_utils import process_aadt_columns

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_pipeline(lastkajen_dir=None, input_dir=None, output_dir=None, tolerance=None, strategies=None):

    lastkajen_dir = lastkajen_dir or str(Paths.LASTKAJEN_GEOPACKAGES_DIR)
    input_dir = input_dir or str(Paths.INPUT_DIR)
    output_dir = output_dir or str(Paths.OUTPUT_DIR)
    tolerance = tolerance if tolerance is not None else Pipeline.NODE_MATCH_TOLERANCE

    report_logger = ReportLogger()
    attr_cons = AttributeManager()
    node_id = NodeIdentifier()

    if Pipeline.PHASE_BLEND_LASTKAJEN_GEOPACKAGES:
        logging.info("Pipeline: Procesamiento inicial")
        
        # Importacion de librerias y funciones necesarias
        from pathlib import Path
        import re
        import pandas as pd
        import geopandas as gpd

        # Ciclo for para la lectura de los geopackages de Lastkajen a lo largo de los años
        gdf_list = []
        for ruta in Path(lastkajen_dir).glob("*.gpkg"):
            year_match = re.findall(Regex.YEAR_REGEX, str(ruta))[-2] ## Extraer el año del nombre del archivo
            year = int(year_match) if year_match else None
            if year in Pipeline.YEARS_TO_ASSESS: ## Filtrar por años de interés
                handler = GeoPackageHandler(ruta) 
                gdf = handler.read_layer(layer_name=Layer.GPKG_LAYER_NAME, force_2d=True) ## Leer la capa de tráfico
                gdf["year"] = year ## Agregar el año a la capa de tráfico
                clip = GeoPackageHandler(Paths.MOBILE_POLYGONS_GEOPACKAGE_DIR).read_layer(layer_name="mobildatapolygoner_granser")
                gdf = handler.clip_geopackage(gdf, clip_geom=clip) ## Recortar la capa de tráfico con la capa de polígonos móviles
                gdf = gdf.drop(columns=Fields.DROP_FIELDS_LASTKAJEN, inplace=False) ## Eliminar campos innecesarios
                gdf = attr_cons.consolidate(gdf, year) 
                gdf_list.append(gdf)  # Agregar el GeoDataFrame a la lista
        gdf_all = gpd.GeoDataFrame(pd.concat(gdf_list, ignore_index=True))  # Combinar todos los GeoDataFrames en uno solo
        logging.info("Pipeline: Procesamiento inicial: concatenacion historico Lastkajen completado")


        # Proceso de limpieza de geometría, identificacion de segmentos homologos e identificación de nodos
        cleaner = GeometryCleaner()
        # Se limpian geometrias nulas o vacias
        gdf_cleaned = cleaner.clean(gdf_all)  # Limpiar geometría
        # Se inician los algoritmos de blending de segmentos e identificación de nodos
        # Se ejecutan todas las estrategias de blending para identificar los segmentos homologos de cada año
        #  gdf_pre_blend = cleaner.find_homologs(gdf_cleaned, strategies=Pipeline.STRATEGIES_BLENDING)
        #  gdf_pre_blend = cleaner.find_homologs_parallel(gdf_cleaned, strategies=Pipeline.STRATEGIES_BLENDING)
        logging.info("Pipeline: Procesamiento inicial: inicia proceso de identificacion de links homologos")
        gdf_pre_blend = cleaner.find_homologs_joblib(gdf_cleaned, n_jobs=6)
        logging.info("Pipeline: Procesamiento inicial: termina proceso de identificacion de links homologos")
        # Se hace el blending de segmentos homologos
        gdf_blend = cleaner.blend_geometries(gdf_pre_blend, Fields.TEMPORAL_FIELDS)

        # Añadir los blend_id a los datos originales (si aún no lo hiciste antes de exportar)
        #  gdf_all_with_blend = gdf_all.merge(gdf_blend[[id_col, "blend_id"]].drop_duplicates(), on=id_col, how="left")
        # Se determinan las intersecciones, nodos finales y segmentos huérfanos
        gdf_nodes, orphan = node_id.identify(gdf_blend)
        # Se eliminan los segmentos huérfanos de la red
        gdf_blend = node_id.remove_orphans(gdf=gdf_blend, orphan_segs=orphan)

        logger.info(f"Geopackage de segmentos unificado con {len(gdf_blend)} elementos")
        
        # Exportar segmentos huérfanos
        try:
            orphan.to_file(f"{output_dir}/segmentos_huerfanos.gpkg", layer='huerfanos', driver='GPKG')
            logger.info(f"{len(orphan)} Segmentos huérfanos exportados a {output_dir}/segmentos_huerfanos.gpkg")
        except Exception as e:
            logger.error(f"Error al exportar segmentos huérfanos: {e}")

        # TODO implementar match_segments
        # gdf_links, gdf_nodes = match_segments(gdf_blend, gdf_nodes, node_tolerance=tolerance)

        # TODO Codificar Matarsmetod, date AAAA-MM, si coincide con el sufijo dejar, 
        # sino borrar independiente del matarmetod Stickprovsmätning
        
        # Exportar los blended segmentos y nodos a GeoPackages
        years_range = f"{Pipeline.YEARS_TO_ASSESS[0]}_{Pipeline.YEARS_TO_ASSESS[-1]}"

        # 1. Nodos; 2. Segmentos preblended; 3. Segmentos blended
        lskj_geopackage_name = Filenames.LASTKAJEN_PROCESSED + "_" + years_range
        export_path = Paths.GEOPACKAGES_DIR / lskj_geopackage_name

        layer_names = {
            "nodes": "nodos_red_" + "_" + years_range,
            "links_pre_blend": "links_pre_blend_" + "_" + years_range,
            "links_blended": "links_blended_" + "_" + years_range,
            }
        
        export = GeoPackageExporter(export_path)
        export.export_nodes(gdf_nodes, layer=layer_names["nodes"])
        export.export_segments(gdf_pre_blend, layer=layer_names["links_pre_blend"])
        export.export_segments(gdf_blend, layer=layer_names["links_blended"])

        logger.info(f"GeoPackage {lskj_geopackage_name} exportado en {export_path} Inspeccion manual necesaria.")
        report_logger.export()

    if Pipeline.PHASE_LINK_LASTKAJEN_TO_EMME:
        logging.info("Pipeline: Union de Lastkajen con red Emme.")
        from pathlib import Path
        import re
        import pandas as pd
        import geopandas as gpd

        # lskj_geopackage_name, clip
        # ---Temporales aqui para solo ejecutar el Pipeline 2 ---
        years_range = f"{Pipeline.YEARS_TO_ASSESS[0]}_{Pipeline.YEARS_TO_ASSESS[-1]}"

        layer_names = {
            "nodes": "nodos_red_" + "_" + years_range,
            "links_pre_blend": "links_pre_blend_" + "_" + years_range,
            "links_blended": "links_blended_" + "_" + years_range,
            }

        # 1. Nodos; 2. Segmentos preblended; 3. Segmentos blended
        lskj_geopackage_name = Filenames.LASTKAJEN_PROCESSED + "_" + years_range
        export_path = Paths.GEOPACKAGES_DIR / lskj_geopackage_name

        clip = GeoPackageHandler(Paths.MOBILE_POLYGONS_GEOPACKAGE_DIR).read_layer(layer_name="mobildatapolygoner_granser")

        # --- Termina temporales ----

        # Aquí se implementaría la lógica de vinculación a Emme
        gdf_lastkajen = GeoPackageHandler(Paths.GEOPACKAGES_DIR / lskj_geopackage_name).read_layer(layer_name=layer_names["links_blended"])
        gdf_links_emme = GeoPackageHandler(Paths.EMME_GEOPACKAGE_DIR).read_layer(layer_name="emme_links_ready_to_join", force_2d=True)
        gdf_nodes_emme = GeoPackageHandler(Paths.EMME_GEOPACKAGE_DIR).read_layer(layer_name="emme_nodes_ready_to_join", force_2d=True)

        # Se eliminan y reorganizan los campos innecesarios de la capa de emme 
        gdf_links_emme = gdf_links_emme.drop(columns=Fields.DROP_FIELDS_EMME, inplace=False)

        # Nuevo: usar LayerMerger que selecciona el match con score más alto
        processing_join = LayerMerger()
        logging.info("Pipeline: Union de Lastkajen con red Emme: Iniciando la unión de capas")
        gdf_links_join = processing_join.merge_layers(gdf_links_emme, gdf_lastkajen)
   
        # processing_join_parallel = LayerMergerParallel()
        # gdf_links_join = processing_join_parallel.merge_layers(gdf_links_emme, gdf_lastkajen)

        
        identify_nodes = NodeIdentifier()
        logging.info("Pipeline: Union de Lastkajen con red Emme: Iniciando la identificación y clasificacion de nodos")
        gdf_nodes_join, orphans = identify_nodes.identify(gdf_links_join, auto_explode_multilines=False)
        gdf_nodes_join = identify_nodes.assign_border_nodes(gdf_nodes_join, clip)
        gdf_nodes_join = identify_nodes.heredar_IDs(gdf_nodes_join, gdf_nodes_emme, max_dist=5)
        gdf_nodes_join = identify_nodes.remove_orphans(gdf=gdf_nodes_join, orphan_segs=orphans)

        #Aqui se verifica la topologia de los links, y se asigna el JNODE de los links que quedaron apuntando a un border_node. Esto es necesario
        # porque cuando se hizo el clip, el link que se corto, quedo apuntando a un punto que ya no existe, y apenas el nuevo punto al que este apunta
        # se crea mediante assign_border_nodes
        gdf_links_join = control_topology(gdf_links_join, gdf_nodes_join)

        # Exportar joined emme geopackages
        logging.info("Pipeline: Union de Lastkajen con red Emme: Exportando los geopackage")
        years_range = f"{Pipeline.YEARS_TO_ASSESS[0]}_{Pipeline.YEARS_TO_ASSESS[-1]}"

        # 1. Nodos; 2. Union EMME to Lastkajen; 3. Union Lastkajen to EMME
        emme_geopackage_name = Filenames.EMME_PROCESSED + "_" + years_range
        export_path = Paths.GEOPACKAGES_DIR / emme_geopackage_name

        layer_names = {
            "nodes": "nodes_emme" + "_" + years_range,
            "links_emme": "links_emme" + "_" + years_range,
            "links_emme_inv": "links_emme_inv" + "_" + years_range
            }

        export = GeoPackageExporter(export_path)
        export.export_nodes(gdf_nodes_join, layer=layer_names["nodes"])
        export.export_segments(gdf_links_join, layer=layer_names["links_emme"])
        #  export.export_segments(gdf_join_inv, layer=layer_names["links_emme_inv"])
        logging.info(f"Se ha exportado el Geopackage {emme_geopackage_name} en {export_path}.")
        
        report_logger.export()

    if Pipeline.PHASE_HANDLING_TOPOLOGY:
        logging.info("Pipeline: handling topology.")

        # 1. Network_Links; 2. TAZ_Links; 3. Auxiliar_Links
        # 4. Network_Nodes; 5. TAZ_Nodes; 6. Auxiliar_Nodes
        # 7. ODM

        # ───── Importaciones ─────
        from pathlib import Path
        import pandas as pd
        import geopandas as gpd

        # ───── Carga de datos ─────
        try:
            # Directorio de red procesada
            REAL_NETWORK_DIR = next(Paths.GEOPACKAGES_DIR.glob(f"*{Filenames.EMME_PROCESSED}*.gpkg"), None)

            # TODO: obtener nombres de capa automáticamente
            REAL_NETWORK_NODES_LAYER = "nodes_emme_2000_2024"  # 4
            REAL_NETWORK_LINKS_LAYER = "links_emme_2000_2024"  # 1

            # 7 - Matriz OD
            ODM = pd.read_csv(Paths.ODM, sep=";", encoding="utf-8")

            # 4 - Network Nodes
            Network_Nodes = GeoPackageHandler(REAL_NETWORK_DIR).read_layer(layer_name="nodes_emme_2000_2024")

            # 1 - Network Links
            Network_Links = GeoPackageHandler(REAL_NETWORK_DIR).read_layer(layer_name="links_emme_2000_2024")

            # 5 y 6 - TAZ y nodos auxiliares
            TAZs_folder = Paths.DATA_DIR / 'TAZ_Layers'
            Auxiliar_Nodes = GeoPackageHandler(TAZs_folder / 'Auxiliar_TAZ').read_layer()
            TAZ_Nodes = GeoPackageHandler(TAZs_folder / 'TAZ').read_layer()

        except FileNotFoundError as e:
            logger.error(f"Error al cargar archivos de red, TAZ o ODM: {e}")
            sys.exit(1)

        # Eliminar los dobles sentidos de Network
        # Crear columnas con los pares ordenados
        Network_Links['node_min'] = Network_Links[['INODE', 'JNODE']].min(axis=1)
        Network_Links['node_max'] = Network_Links[['INODE', 'JNODE']].max(axis=1)

        # Eliminar duplicados basados en los pares ordenados
        Network_Links = Network_Links.drop_duplicates(subset=['node_min', 'node_max'])

        # Opcional: eliminar columnas auxiliares si ya no las necesitas
        Network_Links = Network_Links.drop(columns=['node_min', 'node_max'])

        # Anadir tipo a cada capa
        Network_Links['edge_type'] = 'road'
        TAZ_Nodes['node_type'] = 'TAZ'
        Auxiliar_Nodes['node_type'] = 'aux'
        
        new_int_nodes = []
        new_links = []

        # Arreglar ID de nodos
        Auxiliar_Nodes['ID'] = Auxiliar_Nodes["ID"] + int(2000)

        # Conectar TAZ nodes a la red.
        for idx, taz in TAZ_Nodes.iterrows():
            nearest_link_idx = Network_Links.distance(taz.geometry).idxmin()
            link_geom, start_node_idx, end_node_idx = Network_Links.loc[nearest_link_idx,  ["geometry", 'INODE', 'JNODE']]
            nearest_point = link_geom.interpolate(link_geom.project(taz.geometry))

            new_node_id = int(idx+1000)
            new_int_nodes.append({
                "ID": new_node_id,
                "geometry": nearest_point,
                "node_type": "Intersection",
                'X': nearest_point.x,
                'Y': nearest_point.y
            })

            start = link_geom.coords[0]
            end = link_geom.coords[-1]
            inter = nearest_point.coords[0]

            link1 = LineString([start, inter])
            link2 = LineString([inter, end])

            #INODE, JNODE, LENGTH
            new_links.append({"geometry": link1, 'INODE': int(start_node_idx), 'JNODE': new_node_id, 
                              'LENGTH': link1.length/1000, "edge_type": "road", "LOG_DIRECTION": 0})
            new_links.append({"geometry": link2, 'INODE': new_node_id, 'JNODE': int(end_node_idx), 
                              'LENGTH': link2.length/1000, "edge_type": "road", "LOG_DIRECTION": 0})

            logical_link = LineString([taz.geometry, nearest_point])
            new_links.append({"geometry": logical_link, 'INODE': new_node_id, 'JNODE': taz['ID'],
                              "LENGTH": logical_link.length/1000, "edge_type": "taz_link", "LOG_DIRECTION": 0})

        # Conectar AUX Nodes a la red
        for idx, aux in Auxiliar_Nodes.iterrows():
            nearest_node_idx = Network_Nodes.distance(aux.geometry).idxmin()
            nearest_node_geom, nearest_node_id = Network_Nodes.loc[nearest_node_idx, ["geometry", "ID"]]
            aux_link = LineString([aux.geometry.coords[0], nearest_node_geom.coords[0]])
            new_links.append({"geometry": aux_link, 'INODE': aux['ID'], 'JNODE': nearest_node_id,
                              'LENGTH': aux_link.length/1000, "edge_type": "aux_link", 'LOG_DIRECTION': 0})
        
        # Tratamiento de las columnas de AADT
        Network_Links = process_aadt_columns(Network_Links)
        #1. Agrupar las columnad por su sufijo _AAAA. 
        #2. Coger ano por ano la columna 'Matarsperiod', 'Matmetod'. Coger el Matarsperiod que es un 
        # entero cuyos 4 primeros numero simbolizan el ano. Verificar que este sea igual que el 
        # sufijo AAAA de la columna. En caso de que no coincidan, asignar 0 a todas las columnas. En caso de que si, dejar la columna 'Matarsperiod' con los 4 primeros numeros, y dejar las otras columnas intactas.
        #3. De los registros que queden identificar que "Matmetod" sea igual a "Stickprovsmätning". Aquellos registros que no cumplan. Asignar 0 a todas las columnas,

        # Crear GeoDataFrames finales
        nodes_all = gpd.GeoDataFrame(pd.concat([Network_Nodes,
                                            gpd.GeoDataFrame(new_int_nodes, crs=Network_Nodes.crs), 
                                            Auxiliar_Nodes,
                                            TAZ_Nodes]))
        
        links_all = gpd.GeoDataFrame(pd.concat([Network_Links, 
                                            gpd.GeoDataFrame(new_links, crs=Network_Links.crs)]))



        # --- Guardar resultados
        final_geopackage_name = Filenames.FINAL_NETWORK
        export_path = Paths.GEOPACKAGES_DIR / final_geopackage_name


        # --- Como Pickle

        #TODO Exportacion de Pickle

        # --- Como Geopackage
        export = GeoPackageExporter(export_path)
        export.export_nodes(nodes_all, layer='nodes')
        export.export_segments(links_all, layer='links')
            
        # Fabricas 2 y 5

        # Layers
        # En Networks Links se crea nueva columna que se llame como tipo de link, en este caso REAL LINK. Para Aux LOGICAL_AUX_NODE, para TAZs LOGICAL_TAZ
        
        # Se coge TAZs Nodes y se le añade la columna del tipo de link, posteriormente se inicializa algoritmos que lo une con la parte de la red mas cercana.
        # Esto significa que se crea un link entre el nodo y el segmento de la red mas cercano. y por tanto se debe crear un nuevo tipo de nodo de tipo interseccion. 
        # que hara que cambie la topologia de la red. Habra que dividir el link justo donde se cree el nodo, (link que tendra los mismos features salvo su ID diferente) 
        # y a estos links tendra que cambiarsele la topologia, puesto que ya uno de sus extreemos corresponde a este nuevo nodo creado.
        
        # Se coge AUX nodes y se hace la misma logica, salvo con la diferencia que se tiene que encontrar el nodo de Network mas cercano. 
        # A este link se le hace la topologia segun la logica de los links de Network

        #Con esto se tiene una Network que contiene los links de Network de tipo REAL, los links de TAZs que son de tipo LOGICAL_LINK_TAZ y los links de AUX que son de tipo LOGICAL_AUX_LINK.
        #Con este se tiene nodos que contiene los nodos de Network que son de tipo Interseccion o Nodo_Final, los nodos de TAZs que son de tipo LOGICAL_TAZ y los nodos de AUX que son de tipo LOGICAL_AUX_NODE.

        # ELEMENT, NODES , LINKS
        # NETWORK,   X,   ,  X
        # TAzs,      X,   ,  TODO
        # AUX_NODE,  X,   ,  TODO
        # ODM
        x = 1


    if Pipeline.PHASE_GRAPH_ANALYSIS:
        # Construcción del grafo y análisis
        logger.info("Construyendo y exportando grafo")
        G = GraphBuilder().build(gdf_links, gdf_nodes)
        GraphExporter(output_dir + "/grafo_vial").export_graphml(G)
        report = GraphAnalyzer().analyze(G)

    if Pipeline.PHASE_VISUALIZATION:
        # Visualización
        folium_map = FoliumMapBuilder().build([gdf_links.to_crs(epsg=4326), gdf_nodes.to_crs(epsg=4326)])
        folium_map.save(f"{output_dir}/mapa_interactivo.html")

    report_logger.export()
    logger.info("Pipeline finalizado")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Pipeline GIS de red vial")
    parser.add_argument('--lastkajen-dir', default=None, help='Directorio de Geopackages Lastkajen')
    parser.add_argument('--input-dir', default=None, help='Directorio de entrada')
    parser.add_argument('--output-dir', default=None, help='Directorio de salida')
    parser.add_argument('--tolerance', type=float, default=None, help='Tolerancia de matching')
    args = parser.parse_args()

    run_pipeline(args.lastkajen_dir, args.input_dir, args.output_dir, args.tolerance)