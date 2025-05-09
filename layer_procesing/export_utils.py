# export_utils.py: combinación de geopackage_exporter.py y report_logger.py

import logging
import geopandas as gpd
from configs.settings import Paths, Log
from pathlib import Path


logger = logging.getLogger(__name__)


class GeoPackageExporter:
    """
    Clase encargada de exportar GeoDataFrames a GeoPackage, incluyendo exportación automática por año.
    """
    def __init__(self, filepath=None):
        self.filepath = Path(filepath) if filepath else (Paths.OUTPUT_DIR / 'red_vial_completa.gpkg')
        # Asegurar que el directorio existe
        self.filepath.parent.mkdir(parents=True, exist_ok=True)

        # Asegurarse de que la extensión sea .gpkg
        if self.filepath.suffix.lower() != '.gpkg':
            self.filepath = self.filepath.with_suffix('.gpkg')

    def export_segments(self, gdf, layer='segmentos_red'):
        self._export_layer(gdf, layer, mode='w')

    def export_nodes(self, gdf, layer='nodos_red'):
        self._export_layer(gdf, layer, mode='w')

    def export_markers(self, gdf, layer='marcadores_direccionales'):
        if gdf is None or gdf.empty:
            logger.info("No hay marcadores direccionales para exportar.")
            return
        self._export_layer(gdf, layer, mode='w')

    def export_by_year(self, gdf, year_col='year', id_col=None, start_mode='w'):
        """
        Exporta automáticamente capas separadas por cada año presente en `year_col`.

        Args:
            gdf (GeoDataFrame): Datos originales con columna de año.
            year_col (str): Nombre de la columna de año.
            id_col (str): Columna de identificación para incluir en cada capa (opcional).
            start_mode (str): Modo inicial ('w' para sobrescribir, 'a' para anexar).
        """
        if year_col not in gdf.columns:
            raise ValueError(f"El GeoDataFrame no contiene la columna '{year_col}'")

        years = sorted(gdf[year_col].dropna().unique())
        for year in years:
            subset = gdf[gdf[year_col] == year].copy()
            layer_name = f"lineas_{year}"
            # si id_col se proporciona, garantizar su presencia
            if id_col and id_col not in subset.columns:
                logger.warning(f"id_col '{id_col}' no encontrado en subset año {year}")
            self._export_layer(subset, layer_name, mode="w")
            logger.info(f"Exportada capa {layer_name} con {len(subset)} features.")

    def _export_layer(self, gdf, layer, mode='w'):
        """
        Método interno para escribir una capa en el GeoPackage.

        mode 'w': crea/reescribe GeoPackage, 'a' anexa capa.
        """
        try:
            gdf.to_file(self.filepath, layer=layer, driver='GPKG', mode=mode)
            logger.info(f"Capa '{layer}' exportada a {self.filepath} (mode={mode})")
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
