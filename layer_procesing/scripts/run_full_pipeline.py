# scripts/run_full_pipeline.py
import argparse
import logging
import re
import pandas as pd
import geopandas as gpd # Asegurarse de importar geopandas
import os # Importar os para verificar la existencia de archivos

from configs.logging_config import setup_logging
from configs.settings import (
    DATA_DIR, TEMPORAL_FIELDS, PRELIMINAR_GEOPACKAGES, CORRECTED_GEOPACKAGES,
    INPUT_DIR, OUTPUT_DIR, LOAD_CORRECTED_GEOPACKAGES, PRELIMINAR_NODES_FILE, PRELIMINAR_LINKS_FILE,
    CORRECTED_NODES_FILE, CORRECTED_LINKS_FILE, NODE_MATCH_TOLERANCE, YEARS_TO_ASSESS, LASTKAJEN_GEOPACKAGES_DIR
)
from data_ingestion.reader import GeoPackageReader
from data_ingestion.metadata import GeoPackageMetadata
from processing.geometry import GeometryCleaner
from processing.attributes import AttributeConsolidator
from processing.matching import match_segments # Asumimos que esta funci\u00F3n ya tiene la l\u00F3gica mejorada
from processing.nodes import NodeIdentifier
# Importar la nueva función para actualizar la topología de nodos
from processing.topology_updater import update_node_topology # Necesitaremos crear este archivo/funci\u00F3n

from graph.builder import GraphBuilder
from graph.analyzer import GraphAnalyzer
from graph.exporter import GraphExporter
from export.geopackage_exporter import GeoPackageExporter
from export.report_logger import ReportLogger
from visualization.networkx_viz import NetworkXPlotter
from visualization.folium_viz import FoliumMapBuilder


setup_logging()


def main():
    parser = argparse.ArgumentParser(description='Ejecuta pipeline completo de red vial GIS')
    parser.add_argument('--lastkajen-geopackage-dir', default=LASTKAJEN_GEOPACKAGES_DIR, help='Directorio de Geopackages Lastkajen')
    parser.add_argument('--preliminar-geopackage-dir', default=PRELIMINAR_GEOPACKAGES, help='Directorio de Geopackages Lastkajen')
    parser.add_argument('--corrected-geopackage-dir', default=CORRECTED_GEOPACKAGES, help='Directorio con GeoPackages anuales')
    parser.add_argument('--input-dir', default=INPUT_DIR, help='Directorio de lectura de datos de entradas')
    parser.add_argument('--output-dir', default=OUTPUT_DIR, help='Directorio de salida para archivos generados')
    parser.add_argument('--tolerance', type=float, default=NODE_MATCH_TOLERANCE, help='Tolerancia de matching geométrico')
    parser.add_argument('--year-to-assess', default=YEARS_TO_ASSESS, help='Cuales son los años que se van a tomar de los geopackages')
    args = parser.parse_args()

    logger = logging.getLogger(__name__)
    report_logger = ReportLogger()

    # Definir rutas completas a los archivos corregidos esperados
    corrected_links_path = PRELIMINAR_GEOPACKAGES / CORRECTED_LINKS_FILE
    corrected_nodes_path = PRELIMINAR_GEOPACKAGES / CORRECTED_NODES_FILE

    gdf_blend_topology = None
    gdf_nodes_topology = None

    # --- Control del Flujo de la Pipeline ---
    if LOAD_CORRECTED_GEOPACKAGES:
        logger.info("LOAD_CORRECTED_GEOPACKAGES es True. Intentando cargar datos corregidos...")
        # Cargar GeoPackages corregidos manualmente
        if os.path.exists(corrected_links_path) and os.path.exists(corrected_nodes_path):
            logger.info(f"Cargando datos corregidos desde {corrected_links_path} y {corrected_nodes_path}")
            try:
                # Cargar usando geopandas.read_file
                gdf_blend_topology = gpd.read_file(corrected_links_path)
                gdf_nodes_topology = gpd.read_file(corrected_nodes_path)

                # Validar que las columnas de topología existen después de cargar
                if 'start_node_id_logico' not in gdf_blend_topology.columns or \
                   'end_node_id_logico' not in gdf_blend_topology.columns or \
                   'direccion_logica' not in gdf_blend_topology.columns:
                    logger.error("Los archivos corregidos de links no tienen las columnas de topologíaa esperadas. Asegúrese de que las correcciones mantuvieron estas columnas.")
                    return # Salir si los datos cargados no son válidos

                # Recalcular la topología de nodos basada en los links corregidos
                logger.info("Actualizando topología de nodos basada en links corregidos...")
                gdf_nodes_topology = update_node_topology(gdf_blend_topology, gdf_nodes_topology)
                logger.info("Topología de nodos actualizada.")


            except Exception as e:
                logger.error(f"Error al cargar o procesar los GeoPackages corregidos: {e}")
                return # Salir en caso de error de carga/procesamiento
        else:
            logger.error(f"Archivos corregidos no encontrados en {args.input_dir}. Aseg\u00FArese de que '{CORRECTED_LINKS_FILE}' y '{CORRECTED_NODES_FILE}' existen.")
            logger.info("Por favor, ejecute con LOAD_CORRECTED_GEOPACKAGES = False primero para generar los archivos para correcci\u00F3n.")
            return # Salir si los archivos no existen

    else: # LOAD_CORRECTED_GEOPACKAGES is False
        logger.info("LOAD_CORRECTED_GEOPACKAGES es False. Ejecutando procesamiento inicial...")
        # --- Pasos de Procesamiento Inicial (si no se cargan datos corregidos) ---

        # Listar archivos gpkg
        import glob
        rutas = glob.glob(f"{args.lastkajen_geopackage_dir}/*.gpkg")
        gdf_list = []

        for ruta in rutas:
            year_match = re.search(r"\d{4}", ruta)
            year = int(year_match.group()) if year_match else None
            if year in args.year_to_assess:
                reader = GeoPackageReader(ruta)
                try:
                    gdf = reader.read_traffic_layer()
                    gdf_list.append((year, gdf))
                except Exception as e:
                    report_logger.log('ERROR', 'lectura', year, None, None, str(e))

        # Consolidar anual
        consolidated = []
        geom_cleaner = GeometryCleaner()
        attr_cons = AttributeConsolidator()
        for year, gdf in gdf_list:
            clean_gdf = geom_cleaner.clean(gdf)
            cons_gdf = attr_cons.consolidate(clean_gdf, year)
            consolidated.append(cons_gdf)
        from geopandas import GeoDataFrame
        gdf_all = GeoDataFrame(pd.concat(consolidated, ignore_index=True))

        # Blending de segmentos
        # Usar los thresholds definidos en settings.py si es necesario
        gdf_blend = geom_cleaner.blend_duplicates(gdf=gdf_all,
                                                id_col="ELEMENT_ID",
                                                strategies=["exact", "distance", "diff", "overlap_ratio", "buffer_overlap"],
                                                thresholds={
                                                      "distance": 1.5,
                                                      "diff": 4,
                                                      "overlap_ratio": 0.5,
                                                      "buffer_overlap": 1.0
                                                    }
                                                )

        # Identificación de nodos
        node_id = NodeIdentifier()
        gdf_nodes, orphan_segs = node_id.identify(gdf_blend)
        report_logger.log('INFO', 'nodos', None, None, None, f'Nodos creados: {len(gdf_nodes)}')

        # Identificación de las direcciones de los segmentos (lógica mejorada)
        logger.info("Ejecutando match_segments para inferencia inicial de topolog\u00EDa...")
        gdf_blend_topology , gdf_nodes_topology = match_segments(gdf_blend,
                                                                 gdf_nodes,
                                                                 node_tolerance=args.tolerance)
        logger.info("match_segments completado.")

        # Exportar GeoPackages preliminares para correcci\u00F3n manual
        logger.info(f"Exportando GeoPackages preliminares a {args.input_dir} para corrección manual...")
        # Inicializar GeoPackageExporter con la ruta completa al archivo de salida
        preliminary_links_path = PRELIMINAR_GEOPACKAGES / PRELIMINAR_LINKS_FILE
        preliminary_nodes_path = PRELIMINAR_GEOPACKAGES / PRELIMINAR_NODES_FILE

        # Usar el exportador para segmentos y nodos
        gpkg_exp_preliminar_links = GeoPackageExporter(preliminary_links_path)
        gpkg_exp_preliminar_nodes = GeoPackageExporter(preliminary_nodes_path)

        try:
            # Exportar links como capa 'segmentos_red' en el archivo CORRECTED_LINKS_FILE
            gpkg_exp_preliminar_links.export_segments(gdf_blend_topology, layer_name='segmentos_red')
            # Exportar nodos como capa 'nodos_red' en el archivo CORRECTED_NODES_FILE
            gpkg_exp_preliminar_nodes.export_nodes(gdf_nodes_topology, layer_name='nodos_red') # Exportar nodos también si es necesario para el revisor

            logger.info(f"GeoPackages preliminares '{PRELIMINAR_LINKS_FILE}' y '{PRELIMINAR_NODES_FILE}' exportados exitosamente a {args.input_dir}.")
            logger.info("Por favor, corrija manualmente estos archivos en un SIG y luego ejecute el script con LOAD_CORRECTED_GEOPACKAGES = True.")
        except Exception as e:
             report_logger.log('ERROR', 'exportacion_preliminar', None, None, None, str(e))
             logger.error(f"Error al exportar GeoPackages preliminares: {e}")

        # Detener la ejecución aquí para permitir la corrección manual
        return


    # --- Continuar la Pipeline (si se cargaron datos corregidos) ---
    logger.info("Continuando la pipeline con datos de topología.")

    # Carga de la layer de ZATs

    # Conectar a la red

    # Actualizar topología

    # Graph creation and export

    # Construcci\u00F3n del grafo
    logger.info("Construyendo grafo NetworkX...")
    gb = GraphBuilder()
    G = gb.build(gdf_blend_topology, gdf_nodes_topology)
    logger.info(f"Grafo construido con {G.number_of_nodes()} nodos y {G.number_of_edges()} aristas.")

    # Exportar grafo
    logger.info(f"Exportando grafo a {args.output_dir}...")
    # GraphML
    """ge = GraphExporter(args.output_dir)
    ge.export_graphml(G) # Descomentar si se necesita exportar a GraphML"""
    # Json
    """ge.export_json(G)
    logger.info("Grafo exportado.")"""

    # Análisis topológico (Descomentar si se necesita)
    # logger.info("Realizando análisis topológico")
    # ga = GraphAnalyzer()
    # report = ga.analyze(G)
    # logger.info("An\u00E1lisis topol\u00F3gico completado.")


    # Visualización con Folium
    """logger.info("Generando mapa interactivo con Folium...")
    # Asegurarse de que los GeoDataFrames est\u00E1n en WGS84 para Folium
    gdf_nodes_wgs84 = gdf_nodes_topology.to_crs(epsg=4326)
    gdf_blend_wgs84 = gdf_blend_topology.to_crs(epsg=4326)

    fm = FoliumMapBuilder()
    m = fm.build([gdf_blend_wgs84, gdf_nodes_wgs84])
    map_output_path = os.path.join(args.output_dir, "folium_mapa_interactivo.html")
    m.save(map_output_path)
    logger.info(f"Mapa interactivo guardado en {map_output_path}.")"""

    """
    # Exportar segmentos huérfanos (Si se ejecuta el procesamiento inicial)
    if 'orphan_segs' in locals() and orphan_segs is not None:
        orphan_segs.to_file(f"{args.output_dir}/segmentos_huerfanos.gpkg", layer='hu\u00E9rfanos', driver='GPKG')
        logger.info(f"Segmentos hu\u00E9rfanos exportados a {args.output_dir}/segmentos_huerfanos.gpkg")
    """

    # Exportar reporte errores
    report_logger.export()
    logger.info('Pipeline completo finalizado')


if __name__ == '__main__':
    main()
