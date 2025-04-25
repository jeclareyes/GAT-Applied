# data_ingestion/reader.py
import fiona
import geopandas as gpd
import logging
from pathlib import Path
from configs.settings import GPKG_LAYER_NAME, DEFAULT_GEOMETRY_TOLERANCE
from data_ingestion.exceptions import LayerNotFoundError, GeoPackageReadError

logger = logging.getLogger(__name__)


class GeoPackageReader:
    """
    Clase para listar y leer capas de GeoPackage.
    """

    def __init__(self, filepath):
        self.filepath = Path(filepath)
        if not self.filepath.exists():
            raise GeoPackageReadError(filepath, "File does not exist")

    def list_layers(self):
        """
        Devuelve la lista de capas disponibles en el GeoPackage.
        """
        try:
            layers = fiona.listlayers(str(self.filepath))
            logger.debug(f"Capas encontradas en {self.filepath}: {layers}")
            return layers
        except Exception as e:
            logger.error(f"Error listando capas: {e}")
            raise GeoPackageReadError(self.filepath, e)

    def read_traffic_layer(self, layer_name=None, force_2d=True):
        """
        Lee la capa de tráfico con validación de existencia.
        """
        layer = layer_name or GPKG_LAYER_NAME
        layers = self.list_layers()
        if layer not in layers:
            logger.error(f"Capa {layer} no encontrada en {self.filepath}")
            raise LayerNotFoundError(layer, self.filepath)
        try:
            gdf = gpd.read_file(
                str(self.filepath),
                layer=layer,
                force_2d=force_2d
            )
            logger.info(f"Cargada capa '{layer}' de {self.filepath}, registros: {len(gdf)}")
            return gdf
        except Exception as e:
            logger.error(f"Error leyendo GeoPackage: {e}")
            raise GeoPackageReadError(self.filepath, e)
