# scripts/inspect_orphans.py
import argparse
import logging
from configs.logging_config import setup_logging
from data_ingestion.reader import GeoPackageReader
from processing.nodes import NodeIdentifier


@setup_logging()
def main():
    parser = argparse.ArgumentParser(description='Detecta y exporta segmentos huérfanos')
    parser.add_argument('input_file', help='GeoPackage consolidado de la red vial')
    parser.add_argument('--output', default='segmentos_huerfanos.gpkg', help='Ruta de salida para huérfanos')
    args = parser.parse_args()

    logger = logging.getLogger(__name__)
    reader = GeoPackageReader(args.input_file)
    try:
        gdf = reader.read_traffic_layer()
    except Exception as e:
        logger.error(f'Error leyendo archivo: {e}')
        return

    node_id = NodeIdentifier()
    _, orphan_segs = node_id.identify(gdf)

    if orphan_segs is None or orphan_segs.empty:
        logger.info('No se encontraron segmentos huérfanos.')
    else:
        orphan_segs.to_file(args.output, layer='huérfanos', driver='GPKG')
        logger.info(f'Segmentos huérfanos exportados en {args.output}')

if __name__ == '__main__':
    main()
