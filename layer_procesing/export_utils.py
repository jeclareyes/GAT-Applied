# export_utils.py: combinación de geopackage_exporter.py y report_logger.py

import logging
import geopandas as gpd
from configs.settings import Paths, Log


logger = logging.getLogger(__name__)


class GeoPackageExporter:
    def __init__(self, filepath=None):
        self.filepath = Paths.OUTPUT_DIR / 'red_vial_completa.gpkg' if filepath is None else filepath

    def export_segments(self, gdf, layer='segmentos_red'):
        self._export_layer(gdf, layer, mode='w')

    def export_nodes(self, gdf, layer='nodos_red'):
        self._export_layer(gdf, layer, mode='a')

    def export_markers(self, gdf, layer='marcadores_direccionales'):
        if gdf is None or gdf.empty:
            logger.info("No hay marcadores direccionales para exportar.")
            return
        self._export_layer(gdf, layer, mode='a')

    def export_all(self, segments, nodes, markers=None):
        self.export_segments(segments)
        self.export_nodes(nodes)
        if markers is not None:
            self.export_markers(markers)
        logger.info(f"Exportación completa al GeoPackage {self.filepath} finalizada.")

    def _export_layer(self, gdf, layer, mode='w'):
        try:
            gdf.to_file(self.filepath, layer=layer, driver='GPKG', mode=mode)
            logger.info(f"Capa '{layer}' exportada a {self.filepath}")
        except Exception as e:
            logger.error(f"Error exportando capa '{layer}': {e}")
            raise


class ReportLogger:
    def __init__(self, logfile=None):
        self.logfile = logfile or Log.LOG_FILE
        self.entries = []

    def log(self, level, category, year=None, element_id=None, coords=None,
            description='', tolerance=None):
        entry = {
            'level': level,
            'category': category,
            'year': year,
            'element_id': element_id,
            'coords': coords,
            'description': description,
            'tolerance': tolerance
        }
        self.entries.append(entry)

        msg = f"{category} | Año: {year} | ELEMENT_ID: {element_id} | Coords: {coords} | {description}"
        if level == 'ERROR':
            logger.error(msg)
        elif level == 'WARNING':
            logger.warning(msg)
        else:
            logger.info(msg)

    def export(self):
        try:
            with open(self.logfile, 'w', encoding='utf-8') as f:
                for e in self.entries:
                    line = (
                        f"{e['level']} | {e['category']} | "
                        f"Año: {e['year']} | ELEMENT_ID: {e['element_id']} | "
                        f"Coords: {e['coords']} | "
                        f"Tolerancia: {e['tolerance']} | {e['description']}\n"
                    )
                    f.write(line)
            logger.info(f"Reporte de errores exportado a {self.logfile}")
        except Exception as ex:
            logger.error(f"Error exportando reporte de errores: {ex}")
