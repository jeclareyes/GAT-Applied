# configs/settings.py

import os
from pathlib import Path

# Directorio base del proyecto
BASE_DIR = Path(__file__).resolve().parent.parent

# Directorios de datos y salidas
LASTKAJEN_GEOPACKAGES_DIR = r"G:\My Drive\MSc\Thesis\ÅDT\Data\Lastkajen\Trafik_Yearly"
PRELIMINAR_GEOPACKAGES = BASE_DIR / "geopackages"
CORRECTED_GEOPACKAGES = BASE_DIR / "geopackages"
DATA_DIR = BASE_DIR / "data"
INPUT_DIR = BASE_DIR / "input"
OUTPUT_DIR = BASE_DIR / "output"
GRAPH_EXPORT_PREFIX = OUTPUT_DIR / "grafo_vial"

# Asegurar que los directorios de input y output existen
os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Nombre de la capa dentro de los GeoPackages
GPKG_LAYER_NAME = "TRAFIK_DK_O_105_Trafik"

# Tolerancias y parámetros estándares
DEFAULT_GEOMETRY_TOLERANCE = 1.0  # en metros
YEAR_REGEX = r"\d{4}"

# Campos temporales a consolidar (from fields.TEMPORAL_FIELDS)
TEMPORAL_FIELDS = [
    'Adt_axelpar', 'Adt_samtliga_fordon', 'Adt_tunga_fordon',
    'Adt_latta_fordon_06_18', 'Adt_latta_fordon_18_22', 'Adt_latta_fordon_22_06',
    'Adt_medeltunga_fordon_06_18', 'Adt_medeltunga_fordon_18_22',
    'Adt_medeltunga_fordon_22_06',
    'Adt_tunga_fordon_06_18', 'Adt_tunga_fordon_18_22', 'Adt_tunga_fordon_22_06',
    'Avsnittsidentitet', 'Matarsperiod', 'Matmetod', 'Mc_floden',
    'Osakerhet_axelpar', 'Osakerhet_samtliga_fordon', 'Osakerhet_tunga_fordon'
]

# Logger
LOG_FILE = OUTPUT_DIR / "reporte_errores.txt"

# Configuración de la pipeline
# Establece a True para cargar GeoPackages corregidos manualmente desde INPUT_DIR
# Establece a False para ejecutar el procesamiento inicial y exportar para corrección
LOAD_CORRECTED_GEOPACKAGES = True

# Nombres de archivo para los GeoPackages a corregir
PRELIMINAR_LINKS_FILE = 'preliminar_links_topology.gpkg'
PRELIMINAR_NODES_FILE = 'preliminar_nodes_topology.gpkg'
CORRECTED_LINKS_FILE = 'corrected_links_topology.gpkg'
CORRECTED_NODES_FILE = 'corrected_links_topology.gpkg'

# Tolerancia de matching geom\u00E9trico para match_segments (en unidades del CRS)
# Ajusta este valor seg\u00FAn la precisi\u00F3n de tus datos y CRS
NODE_MATCH_TOLERANCE = 5.0 # Ejemplo: 5 metros si el CRS est\u00E1 en metros

# A\u00F1os a procesar de los GeoPackages anuales
YEARS_TO_ASSESS = [2020, 2021, 2022]

