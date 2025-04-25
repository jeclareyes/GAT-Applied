# data_ingestion/metadata.py
import os
import json
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class GeoPackageMetadata:
    """
    Extrae y valida metadatos de un GeoPackage (via JSON sidecar o fiona meta).
    """

    def __init__(self, filepath):
        self.filepath = filepath
        self.metadata = {}

    def extract_via_fiona(self):
        """
        Usa fiona para leer metadatos básicos.
        """
        try:
            import fiona
            with fiona.open(self.filepath) as src:
                self.metadata['driver'] = src.driver
                self.metadata['crs'] = src.crs
                self.metadata['schema'] = src.schema
                self.metadata['bounds'] = src.bounds
                self.metadata['layer_count'] = len(fiona.listlayers(self.filepath))
            logger.debug(f"Metadatos extraídos de {self.filepath} via fiona")
        except Exception as e:
            logger.warning(f"No se pudo extraer metadatos via fiona: {e}")
        return self.metadata

    def extract_sidecar(self):
        """
        Busca un archivo .json con metadatos junto al GeoPackage.
        """
        sidecar = os.path.splitext(self.filepath)[0] + '.json'
        if os.path.exists(sidecar):
            try:
                with open(sidecar, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                self.metadata.update(data)
                logger.debug(f"Metadatos cargados desde sidecar {sidecar}")
            except Exception as e:
                logger.warning(f"Error leyendo sidecar JSON {sidecar}: {e}")
        return self.metadata

    def validate_date(self, key='creation_date'):
        """
        Valida que la fecha tenga formato ISO y sea razonable.
        """
        date_str = self.metadata.get(key)
        if not date_str:
            logger.info(f"Metadato {key} no encontrado para {self.filepath}")
            return False
        try:
            dt = datetime.fromisoformat(date_str)
            self.metadata[key] = dt.isoformat()
            return True
        except ValueError:
            logger.error(f"Formato de fecha inválido en {key}: {date_str}")
            return False

    def get_metadata(self):
        """
        Orquesta la extracción completa y validación.
        """
        self.extract_via_fiona()
        self.extract_sidecar()
        self.validate_date()
        return self.metadata
