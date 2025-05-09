# configs/settings.py

import os
from pathlib import Path
import yaml

class Paths:
    BASE_DIR = Path(__file__).resolve().parent.parent
    try:
        YAML_FILE = BASE_DIR / "configs/directories.yml"
        with open(YAML_FILE, "r") as file:
            directories = yaml.safe_load(file)
            LASTKAJEN_GEOPACKAGES_DIR = directories["LASTKAJEN_GEOPACKAGES_TO_MERGE"]
            EMME_GEOPACKAGE_DIR = directories["EMME_GEOPACKAGES_TO_JOIN"]
            MOBILE_POLYGONS_GEOPACKAGE_DIR = os.path.join(directories["MOBILE_POLIGONS"] , "mobildatapolygoner_granser.gpkg")
            ODM = directories["ODM_FILE"]
    except FileNotFoundError:
        print(f"Se procede sin archivo .yml")
        LASTKAJEN_GEOPACKAGES_DIR = r"G:\My Drive\MSc\Thesis\ÅDT\Data\Lastkajen\Trafik_Yearly"
        # Proporcionar valores predeterminados o manejar el error según sea necesario
    
    DATA_DIR = BASE_DIR / "data"
    INPUT_DIR = DATA_DIR / "input"
    OUTPUT_DIR = DATA_DIR / "output"
    GRAPH_EXPORT_PREFIX = OUTPUT_DIR / "Graph"
    GEOPACKAGES_DIR = DATA_DIR / "geopackages"

    # Crear directorios si no existen
    os.makedirs(INPUT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)


class Layer:
    BASE_DIR = Path(__file__).resolve().parent.parent
    try:
        YAML_FILE = BASE_DIR / "configs/directories.yml"
        with open(YAML_FILE, "r") as file:
            directories = yaml.safe_load(file)
            GPKG_LAYER_NAME = directories["LASTKAJEN_LAYERS_NAME"]
    except FileNotFoundError:
        print(f"Se procede sin archivo .yml")
        GPKG_LAYER_NAME = "TRAFIK_DK_O_105_Trafik"

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
                             'Adt_latta_fordon_06_18', 'Adt_latta_fordon_18_22', 'Adt_latta_fordon_22_06',
                             'Adt_medeltunga_fordon_06_18', 'Adt_medeltunga_fordon_18_22', 'Adt_medeltunga_fordon_22_06',
                             'Adt_tunga_fordon_06_18', 'Adt_tunga_fordon_18_22', 'Adt_tunga_fordon_22_06',
                             'Mc_floden',
                             'Osakerhet_axelpar', 'Osakerhet_samtliga_fordon', 'Osakerhet_tunga_fordon']

    DROP_FIELDS_EMME = ['ID', # ID of the road link. It is a "INODE-JNODE" string, e.g. "1234-5678"
                        # 'INODE', 'JNODE', # Node IDs of the road link. INODE = start node, JNODE = end node
                        # 'LENGTH', # Length of the road link in meters
                        'TYPE', # 1. Road Link, Others: Ferry, road shaft
                        #'LANES', # Number of lanes.
                        # 'VDF', #Function Volumen Delay. 1-76 ordinary road
                        'DATA1', 'DATA2', 'DATA3', # Don't know what they are
                        '@ad_filter', # Filter for ADT calculation. 1 = filter, 0 = no filter
                        '@adlbs', '@adlbu', # ADT for freight (2019)
                        '@adpb', # ADT for passenger cars (2019)
                        '@atk', # fartkamera, 0 = no speed camera, 1 = speed camera 
                        '@fvkl', # Function class. 0–9 according to NVDB specification 
                        # '@hast', # Speed limit.
                        '@juhas', '@jukap', # Adjustment factors for speed and capacity, usually all of them are 1.0
                        '@komun', # Number of commun according to SCB
                        '@lbef', # Population in urban area
                        # '@vkat', # Road category 1: Europaväg, 2: Riksväg, 3: Primär länsväg, 4: Secundär länsväg, 5: tertiär länsväg, 6: ospec.
                        # '@vnr', # Road number
                        '@vstng', # Wildlife fence in % of the road length
                        # '@vtyp', # 4 = ML road, 5 = motorway, 9 = 2lane road, 10 = separation-free country road (ML) # for VDF calculation
                        # 'geometry' # Geometry of the road link
                        ]
    
    ALL_FIELDS_EMME = ['ID', 'INODE', 'JNODE', 'LENGTH', 'TYPE', 'LANES', 'VDF', 'DATA1',
       'DATA2', 'DATA3', '@ad_filter', '@adlbs', '@adlbu', '@adpb', '@atk',
       '@fvkl', '@hast', '@juhas', '@jukap', '@komun', '@lbef', '@vkat',
       '@vnr', '@vstng', 
        'geometry']

class Pipeline:
    PHASE_BLEND_LASTKAJEN_GEOPACKAGES = True
    PHASE_LINK_LASTKAJEN_TO_EMME = True
    PHASE_HANDLING_TOPOLOGY = True
    PHASE_GRAPH_ANALYSIS = False
    PHASE_VISUALIZATION = False


    NODE_MATCH_TOLERANCE = 5.0
    # YEARS_TO_ASSESS = [2020, 2021, 2022]
    YEARS_TO_ASSESS = [2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009,
                       2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019,
                       2020, 2021, 2022, 2023, 2024]
    
    # TODO implementar tambien id_match
    STRATEGIES_BLENDING = ["exact", "bibuffer_overlap", "similarity_index"]
    STRATEGIES_MATCHING = ["bibuffer_overlap", "similarity_index"]
    #  STRATEGIES_JOIN = ["exact", "distance", "diff", "overlap_ratio", "buffer_overlap", "bibuffer_overlap", "similarity_index"]


class Filenames:
    LASTKAJEN_PROCESSED = 'Lastkajen_Processed'
    EMME_PROCESSED = 'Emme_Processed'

    PRELIMINAR_EMME_LINKS_FILE = 'emme_network_links'
    CORRECTED_LINKS_FILE = 'corrected_links_topology'
    CORRECTED_NODES_FILE = 'corrected_links_topology'
    JOINED_EMME_LINKS_FILE = 'joined_emme_network_links'
    JOINED_EMME_NODES_FILE = 'joined_emme_network_nodes'

    FINAL_NETWORK = 'final_network'


class Regex:
    YEAR_REGEX = r"\d{4}"


class Log:
    LOG_FILE = Paths.OUTPUT_DIR / "Log_Errors.txt"