# processing/attributes.py

import logging  # Módulo para registrar mensajes de log (informativos, advertencias, errores, etc.)
from configs.settings import \
    TEMPORAL_FIELDS  # Lista de campos que representan atributos temporales, definidos en la configuración del proyecto
from utils.text_utils import normalize_text, \
    find_best_match  # Funciones auxiliares para normalizar texto y encontrar coincidencias aproximadas entre strings

# Se crea un logger específico para este módulo, útil para seguimiento y depuración
logger = logging.getLogger(__name__)


class AttributeConsolidator:
    """
    Clase responsable de consolidar y renombrar atributos temporales en un GeoDataFrame (gdf).

    Principalmente, agrega un sufijo con el año (_{year}) a los campos definidos en TEMPORAL_FIELDS,
    asegurando que existan en el DataFrame y que no se sobreescriban nombres al trabajar con múltiples años.
    También trata de encontrar nombres similares si el campo no existe exactamente.
    """

    def __init__(self, temporal_fields=TEMPORAL_FIELDS):
        """
        Inicializa el consolidado con una lista de campos temporales.

        Parámetros:
        - temporal_fields: lista de nombres de campos considerados como atributos temporales.
        """
        self.temporal_fields = temporal_fields

    def consolidate(self, gdf, year):
        """
        Consolida y renombra los campos temporales en el GeoDataFrame, agregando el sufijo _{year}.
        Si el campo no se encuentra directamente, intenta encontrar el nombre más parecido usando fuzzy matching.

        Parámetros:
        - gdf: GeoDataFrame de entrada con columnas que representan atributos.
        - year: Año que se usará como sufijo (_{year}) para los campos temporales.

        Retorna:
        - Una copia del GeoDataFrame con las columnas renombradas si aplica.
        """
        # Se crea una copia del GeoDataFrame original para no modificar el original directamente
        gdf = gdf.copy()

        # Formato del sufijo que se añadirá a los campos (ej: "_2020")
        suffix = f"_{year}"

        # Iteramos sobre todos los campos temporales esperados
        for field in self.temporal_fields:

            # Si el campo existe exactamente en las columnas del gdf, se renombra con sufijo
            if field in gdf.columns:
                gdf.rename(columns={field: field + suffix}, inplace=True)

            else:
                # Si no existe exactamente, se intenta encontrar la mejor coincidencia usando fuzzy matching
                match = find_best_match(field, gdf.columns)

                if match:
                    # Si se encuentra una coincidencia razonable, también se renombra
                    gdf.rename(columns={match: match + suffix}, inplace=True)

                    # Se registra la coincidencia en los logs para referencia
                    logger.info(
                        f"Campo '{field}' mapeado a '{match}' para año {year}"
                    )
                else:
                    # Si no se encuentra ningún campo que coincida, se lanza una advertencia
                    logger.warning(
                        f"Campo '{field}' no encontrado ni coincidente en año {year}"
                    )

        # Retornamos el GeoDataFrame con las modificaciones
        return gdf
