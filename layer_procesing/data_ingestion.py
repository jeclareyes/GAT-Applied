# data_ingestion.py (versión simplificada de reader.py, metadata.py y exceptions.py)

import os
import json
import fiona
import logging
import geopandas as gpd
from pathlib import Path
from datetime import datetime
from configs.settings import Layer

logger = logging.getLogger(__name__)

# --- Excepciones personalizadas ---

class DataIngestionError(Exception):
    pass

class LayerNotFoundError(DataIngestionError):
    def __init__(self, layer_name, filepath):
        message = f"Layer '{layer_name}' not found in GeoPackage: {filepath}"
        super().__init__(message)

class GeoPackageReadError(DataIngestionError):
    def __init__(self, filepath, original_exception):
        message = f"Error reading GeoPackage {filepath}: {original_exception}"
        super().__init__(message)


# --- Clase unificada para lectura y metadatos ---

class GeoPackageHandler:
    def __init__(self, filepath):
        self.filepath = Path(filepath)

        # Asegurarse de que la extensión sea .gpkg
        if self.filepath.suffix.lower() != '.gpkg':
            self.filepath = self.filepath.with_suffix('.gpkg')

        if not self.filepath.exists():
            raise GeoPackageReadError(filepath, "File does not exist")
        self.metadata = {}

    def list_layers(self):
        try:
            layers = fiona.listlayers(str(self.filepath))
            logger.debug(f"Capas en {self.filepath}: {layers}")
            return layers
        except Exception as e:
            raise GeoPackageReadError(self.filepath, e)

    def read_layer(self, layer_name=None, force_2d=True):
        layer = layer_name
        if layer not in self.list_layers():
            layers = self.list_layers()
            if len(layers) == 1:
                layer_name = layers[0]
            else:
                raise LayerNotFoundError(layer, self.filepath)
        try:
            gdf = gpd.read_file(str(self.filepath), layer=layer_name, force_2d=force_2d)
            #  logger.info(f"Capa '{layer_name}' cargada desde {self.filepath}, registros: {len(gdf)}")
            return gdf
        except Exception as e:
            raise GeoPackageReadError(self.filepath, e)
        
    def clip_geopackage(self, gdf, clip_geom):
        try:
            gdf_clipped = gpd.clip(gdf, clip_geom)
            #  logger.info(f"Capa recortada con geometría de recorte, registros: {len(clip_geom)}")
            return gdf_clipped
        except Exception as e:
            raise GeoPackageReadError(self.filepath, e)

    def extract_metadata(self):
        self._extract_via_fiona()
        self._extract_sidecar()
        self._validate_date()
        return self.metadata

    def _extract_via_fiona(self):
        try:
            with fiona.open(self.filepath) as src:
                self.metadata.update({
                    'driver': src.driver,
                    'crs': src.crs,
                    'schema': src.schema,
                    'bounds': src.bounds,
                    'layer_count': len(fiona.listlayers(self.filepath))
                })
        except Exception as e:
            logger.warning(f"No se pudo extraer metadatos via fiona: {e}")

    def _extract_sidecar(self):
        sidecar = self.filepath.with_suffix('.json')
        if sidecar.exists():
            try:
                with open(sidecar, 'r', encoding='utf-8') as f:
                    self.metadata.update(json.load(f))
            except Exception as e:
                logger.warning(f"Error leyendo sidecar {sidecar}: {e}")

    def _validate_date(self, key='creation_date'):
        date_str = self.metadata.get(key)
        if not date_str:
            return
        try:
            dt = datetime.fromisoformat(date_str)
            self.metadata[key] = dt.isoformat()
        except ValueError:
            logger.warning(f"Fecha inválida en metadato '{key}': {date_str}")