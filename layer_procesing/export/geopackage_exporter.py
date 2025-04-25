# export/geopackage_exporter.py
import logging
import geopandas as gpd
from configs.settings import OUTPUT_DIR

class GeoPackageExporter:
    """
    Exporta capas de segmentos, nodos y marcadores direccionales a un GeoPackage.
    """
    def __init__(self, filepath=None):
        # Por defecto, usa OUTPUT_DIR/red_vial_completa.gpkg
        self.filepath = OUTPUT_DIR / 'red_vial_completa.gpkg' if filepath is None else filepath

    def export_segments(self, gdf_segments, layer_name='segmentos_red'):
        """Exporta GeoDataFrame de segmentos."""
        try:
            gdf_segments.to_file(
                self.filepath,
                layer=layer_name,
                driver='GPKG'
            )
            logger.info(f"Segmentos exportados en capa '{layer_name}' de {self.filepath}")
        except Exception as e:
            logger.error(f"Error exportando segmentos: {e}")
            raise

    def export_nodes(self, gdf_nodes, layer_name='nodos_red'):
        """Exporta GeoDataFrame de nodos."""
        try:
            gdf_nodes.to_file(
                self.filepath,
                layer=layer_name,
                driver='GPKG',
                mode='a'
            )
            logger.info(f"Nodos exportados en capa '{layer_name}' de {self.filepath}")
        except Exception as e:
            logger.error(f"Error exportando nodos: {e}")
            raise

    def export_markers(self, gdf_markers, layer_name='marcadores_direccionales'):
        """Exporta GeoDataFrame de marcadores direccionales."""
        if gdf_markers is None or gdf_markers.empty:
            logger.info("No hay marcadores direccionales para exportar.")
            return
        try:
            gdf_markers.to_file(
                self.filepath,
                layer=layer_name,
                driver='GPKG',
                mode='a'
            )
            logger.info(f"Marcadores exportados en capa '{layer_name}' de {self.filepath}")
        except Exception as e:
            logger.error(f"Error exportando marcadores: {e}")
            raise

    def export_all(self, gdf_segments, gdf_nodes, gdf_markers=None):
        """Realiza la exportación completa de la red vial a GeoPackage."""
        self.export_segments(gdf_segments)
        self.export_nodes(gdf_nodes)
        if gdf_markers is not None:
            self.export_markers(gdf_markers)
        logger.info(f"Exportación completa al GeoPackage {self.filepath} finalizada.")


logger = logging.getLogger(__name__)
