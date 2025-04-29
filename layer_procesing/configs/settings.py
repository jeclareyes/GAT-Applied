# configs/settings.py

import os
from pathlib import Path
import yaml

class Paths:
    
    __init__ = None  # Evitar la creación de instancias de esta clase
    with open("configs/directories.yaml", "r") as file:
        directories = yaml.safe_load(file)

    BASE_DIR = Path(__file__).resolve().parent.parent
    DATA_DIR = BASE_DIR / "data"
    INPUT_DIR = DATA_DIR / "input"
    OUTPUT_DIR = DATA_DIR / "output"
    GRAPH_EXPORT_PREFIX = OUTPUT_DIR / "Graph"
    LASTKAJEN_GEOPACKAGES_DIR = r"G:\My Drive\MSc\Thesis\ÅDT\Data\Lastkajen\Trafik_Yearly" or directories["LASTKAJEN_GEOPACKAGES_TO_MERGE"]
    EMME_GEOPACKAGE_DIR = directories["EMME_GEOPACKAGE_TO_MERGE"]
    PRELIMINAR_GEOPACKAGES = DATA_DIR / "geopackages"
    CORRECTED_GEOPACKAGES = DATA_DIR / "geopackages"

    # Crear directorios si no existen
    os.makedirs(INPUT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)


class Layer:
    __init__ = None  # Evitar la creación de instancias de esta clase
    with open("configs/directories.yaml", "r") as file:
        directories = yaml.safe_load(file)
    GPKG_LAYER_NAME = "TRAFIK_DK_O_105_Trafik" or directories["LASTKAJEN_LAYERS_NAME"]

# Tolerancias y parámetros estándares
DEFAULT_GEOMETRY_TOLERANCE = 5.0  # en metros
YEAR_REGEX = r"\d{4}"

class Fields:
    TEMPORAL_FIELDS = [
        'Adt_axelpar', 'Adt_samtliga_fordon', 'Adt_tunga_fordon',
        'Adt_latta_fordon_06_18', 'Adt_latta_fordon_18_22', 'Adt_latta_fordon_22_06',
        'Adt_medeltunga_fordon_06_18', 'Adt_medeltunga_fordon_18_22',
        'Adt_medeltunga_fordon_22_06',
        'Adt_tunga_fordon_06_18', 'Adt_tunga_fordon_18_22', 'Adt_tunga_fordon_22_06',
        'Matarsperiod', 'Matmetod', 'Mc_floden',
        'Osakerhet_axelpar', 'Osakerhet_samtliga_fordon', 'Osakerhet_tunga_fordon'
    ]
    DROP_FIELDS_LASTKAJEN = ['START_MEASURE', 'END_MEASURE', 'ISHOST',
                             'Adt_samtliga_fordon', 'Adt_tunga_fordon',
                             'Adt_latta_fordon_06_18', 'Adt_latta_fordon_18_22', 'Adt_latta_fordon_22_06',
                             'Adt_medeltunga_fordon_06_18', 'Adt_medeltunga_fordon_18_22', 'Adt_medeltunga_fordon_22_06',
                             'Adt_tunga_fordon_06_18', 'Adt_tunga_fordon_18_22', 'Adt_tunga_fordon_22_06',
                             'Mc_floden',
                             'Osakerhet_axelpar', 'Osakerhet_samtliga_fordon', 'Osakerhet_tunga_fordon']


class Pipeline:
    PHASE_BLEND_LASTKAJEN_GEOPACKAGES = True
    PHASE_ANALYZE_BLENDED_LASTKAJEN_GEOPACKAGE = False
    PHASE_LINK_LASTKAJEN_TO_EMME = False
    NODE_MATCH_TOLERANCE = 5.0
    #YEARS_TO_ASSESS = [2020, 2021, 2022]
    YEARS_TO_ASSESS = [2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009,
                       2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019,
                       2020, 2021, 2022, 2023, 2024]
    STRATEGIES = ["id_match", "exact", "bibuffer_overlap", "similarity_index"]
    #  STRATEGIES = ["exact", "distance", "diff", "overlap_ratio", "buffer_overlap", "bibuffer_overlap", "similarity_index"]


class Filenames:
    PRELIMINAR_LINKS_FILE = 'blended_links_from_lastkajen'
    PRELIMINAR_NODES_FILE = 'blended_nodes_from_lastkajen'
    CORRECTED_LINKS_FILE = 'corrected_links_topology'
    CORRECTED_NODES_FILE = 'corrected_links_topology'


class Regex:
    YEAR_REGEX = r"\d{4}"


class Log:
    LOG_FILE = Paths.OUTPUT_DIR / "Log_Errors.txt"
