# pipeline.py: script principal para correr all el flujo

import argparse
import sys
import logging
from configs.settings import Paths, Pipeline, Filenames, Regex, Fields, Layer
from export_utils import GeoPackageExporter, ReportLogger
from data_ingestion import GeoPackageHandler
from network_processing import *
from graph_tools import GraphBuilder, GraphAnalyzer, GraphExporter
from map_visualization import FoliumMapBuilder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_pipeline(lastkajen_dir=None, input_dir=None, output_dir=None, tolerance=None, strategies=None):
    lastkajen_dir = lastkajen_dir or str(Paths.LASTKAJEN_GEOPACKAGES_DIR)
    input_dir = input_dir or str(Paths.INPUT_DIR)
    output_dir = output_dir or str(Paths.OUTPUT_DIR)
    tolerance = tolerance if tolerance is not None else Pipeline.NODE_MATCH_TOLERANCE

    report_logger = ReportLogger()
    geom_cleaner = GeometryCleaner()
    attr_cons = AttributeConsolidator()
    node_id = NodeIdentifier()

    corrected_links_path = Paths.GEOPACKAGES_DIR / Filenames.CORRECTED_LINKS_FILE
    corrected_nodes_path = Paths.GEOPACKAGES_DIR / Filenames.CORRECTED_NODES_FILE


    if Pipeline.PHASE_BLEND_LASTKAJEN_GEOPACKAGES:
        logging.info("Pipeline: Procesamiento inicial")
        from pathlib import Path
        import re
        import pandas as pd
        import geopandas as gpd
        gdf_list = []
        for ruta in Path(lastkajen_dir).glob("*.gpkg"):
            year_match = re.findall(Regex.YEAR_REGEX, str(ruta))[-2]
            year = int(year_match) if year_match else None
            if year in Pipeline.YEARS_TO_ASSESS:
                handler = GeoPackageHandler(ruta)
                gdf = handler.read_layer(layer_name=Layer.GPKG_LAYER_NAME, force_2d=True)
                clip = GeoPackageHandler(Paths.MOBILE_POLYGONS_GEOPACKAGE_DIR).read_layer(layer_name="mobildatapolygoner_granser")
                gdf = handler.clip_geopackage(gdf, clip_geom=clip)
                gdf = gdf.drop(columns=Fields.DROP_FIELDS_LASTKAJEN, inplace=False)
                gdf = geom_cleaner.clean(gdf)
                gdf = attr_cons.consolidate(gdf, year)
                gdf_list.append(gdf)
        gdf_all = gpd.GeoDataFrame(pd.concat(gdf_list, ignore_index=True))
        gdf_blend = geom_cleaner.blend_duplicates(gdf_all, strategies=strategies)
        gdf_nodes, orphan = node_id.identify(gdf_blend)

        logger.info(f"Geopackage de segmentos unificado con {len(gdf_blend)} elementos")
        # Exportar segmentos huérfanos
        try:
            orphan.to_file(f"{output_dir}/segmentos_huerfanos.gpkg", layer='huerfanos', driver='GPKG')
            logger.info(f"{len(orphan)} Segmentos huérfanos exportados a {output_dir}/segmentos_huerfanos.gpkg")
        except Exception as e:
            logger.error(f"Error al exportar segmentos huérfanos: {e}")

        # TODO implementar match_segments
        # gdf_links, gdf_nodes = match_segments(gdf_blend, gdf_nodes, node_tolerance=tolerance)
        years_range = f"{Pipeline.YEARS_TO_ASSESS[0]}_{Pipeline.YEARS_TO_ASSESS[-1]}"
        exporter = GeoPackageExporter(Paths.GEOPACKAGES_DIR / (Filenames.PRELIMINAR_LINKS_FILE + "_" + years_range + ".gpkg"))
        # TODO should be gdf_links
        exporter.export_segments(gdf_blend)
        exporter = GeoPackageExporter(Paths.GEOPACKAGES_DIR / (Filenames.PRELIMINAR_NODES_FILE + "_" + years_range + ".gpkg"))
        exporter.export_nodes(gdf_nodes)
        logger.info(f"GeoPackages {Filenames.PRELIMINAR_LINKS_FILE} y {Filenames.PRELIMINAR_NODES_FILE} exportados. Inspeccion manual necesaria.")
        report_logger.export()

    if Pipeline.PHASE_LINK_LASTKAJEN_TO_EMME:
        logging.info("Pipeline: Union de Lastkajen con red Emme.")
        from pathlib import Path
        import re
        import pandas as pd
        import geopandas as gpd

        years_range = f"{Pipeline.YEARS_TO_ASSESS[0]}_{Pipeline.YEARS_TO_ASSESS[-1]}"

        # Aquí se implementaría la lógica de vinculación a Emme
        gdf_lastkajen = GeoPackageHandler(Paths.GEOPACKAGES_DIR / "blended_links_from_lastkajen_2000_2024.gpkg").read_layer()
        gdf_emme = GeoPackageHandler(Paths.EMME_GEOPACKAGE_DIR).read_layer()
        
        processing_join = LayerMerger()
        gdf_join = processing_join.merge_layers(gdf_lastkajen, gdf_emme)
        
        exporter = GeoPackageExporter(Paths.GEOPACKAGES_DIR / (Filenames.JOINED_EMME_LINKS_FILE + "_" + years_range + ".gpkg"))
        exporter.export_segments(gdf_join, layer='joined_emme_network_links')
        logging.info(f"Se ha exportado el Geopackage {Filenames.JOINED_EMME_LINKS_FILE}")
        
        report_logger.export()

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

    run_pipeline(args.lastkajen_dir, args.input_dir, args.output_dir, args.tolerance, strategies=Pipeline.STRATEGIES)
