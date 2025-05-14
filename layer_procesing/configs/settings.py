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
    VISUALIZATION_DIR = DATA_DIR

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

class GraphAnalysis:
    X = 1


class Pipeline:
    PHASE_BLEND_LASTKAJEN_GEOPACKAGES = False
    PHASE_LINK_LASTKAJEN_TO_EMME = False
    PHASE_HANDLING_TOPOLOGY = False
    PHASE_GRAPH_ANALYSIS = False
    PHASE_VISUALIZATION = True

    DELETE_DOUBLE_LINKS = False

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
    
# --- GRAPH PARAMETERS ---
import contextily as cx # Solo si se usa para default_basemap_source más adelante
from map_visualization import (
    NodePlotConfig, NodeStyle, NodeLabelConfig, LabelProperties,
    EdgePlotConfig, EdgeStyle, EdgeLabelConfig, EdgeGlobalArrowConfig,
    format_node_taz_salidas_entradas, format_link_attributes_list, format_aadt_label
)


# O from formatters import format_node_taz_salidas_entradas, ...

# --- NOMBRES DE COLUMNAS CLAVE (Recomendación 1) ---
# Define los nombres de las columnas que tu visualizador espera de los GeoDataFrames.
# Esto permite que la clase NetworkVisualizerOSMnx sea más adaptable
# si los nombres de las columnas de entrada cambian.
INPUT_COLUMNS = {
    "node_id_internal": "osmid",  # Nombre de la columna que se usará/creará como índice de nodo en el grafo
    "node_original_id_column": "ID", # Nombre de tu columna ID original en el GDF de nodos
    "node_type_column": "node_type",
    "node_x_coord_column": "X", # O el nombre de tu columna de coordenada X
    "node_y_coord_column": "Y", # O el nombre de tu columna de coordenada Y
    "node_geometry_column": "geometry",

    "edge_from_node_column": "INODE", # Nombre original en tu GDF de arcos para el nodo origen
    "edge_to_node_column": "JNODE",   # Nombre original en tu GDF de arcos para el nodo destino
    "edge_type_column": "edge_type",
    "edge_geometry_column": "geometry",
    "edge_direction_indicator_column": "LOG_DIRECTION", # Para la lógica de flechas
    "edge_key_column": "key" # Nombre de la columna de clave para arcos múltiples (si existe, si no, se puede omitir o crear)
}

# --- ESTILOS DE NODOS ESPECÍFICOS ---
STYLE_NODE_TAZ = NodeStyle(color='orange', size=100, alpha=0.9)
STYLE_NODE_AUX = NodeStyle(color='green', size=50, alpha=0.7)
STYLE_NODE_INTERSECTION = NodeStyle(color='purple', size=30)
STYLE_NODE_BORDER = NodeStyle(color='red', size=35, marker='s') # Matplotlib marker 's' for square

# --- CONFIGURACIÓN DE PLOTEO DE NODOS ---
node_plot_cfg = NodePlotConfig(
    style_mapping={
        'TAZ': STYLE_NODE_TAZ,
        'aux': STYLE_NODE_AUX, # Asegúrate que 'aux' sea el valor exacto en tus datos
        'Intersection': STYLE_NODE_INTERSECTION,
        'Border_Node': STYLE_NODE_BORDER
    },
    default_style=NodeStyle(color="lightgrey", size=15),
    warning_style=NodeStyle(color='magenta', size=60, zorder=10), # zorder alto para que resalten
    label_configs=[
        NodeLabelConfig(
            label_id="taz_name_custom",
            attribute_source='NAMN', # Atributo que contiene el nombre del TAZ
            node_types=['TAZ'],      # Solo aplica a nodos de tipo 'TAZ'
            properties=LabelProperties(font_size=10, font_color='darkblue', y_offset=0.00025)
        ),
        NodeLabelConfig(
            label_id="aux_id_custom",
            attribute_source=INPUT_COLUMNS["node_original_id_column"], # Usando la config de columnas
            node_types=['aux'], # Solo aplica a nodos de tipo 'aux'
            properties=LabelProperties(font_size=7, font_color='darkgreen', y_offset=-0.00018)
        ),
        NodeLabelConfig(
            label_id="taz_io_formatted",
            node_types=['TAZ'],
            formatting_function=format_node_taz_salidas_entradas, # Función importada
            properties=LabelProperties(font_size=7, font_color='black', x_offset=0.00035)
        )
        # Añade más configuraciones de etiquetas de nodos aquí
    ]
)

# --- ESTILOS DE ARCOS ESPECÍFICOS ---
STYLE_EDGE_ROAD = EdgeStyle(color='black', linewidth=2.5, linestyle='-')
STYLE_EDGE_TAZ_LINK = EdgeStyle(color='darkorange', linewidth=1.5, linestyle='--')
STYLE_EDGE_AUX_LINK = EdgeStyle(color='mediumseagreen', linewidth=1.0, linestyle=':')

# --- CONFIGURACIÓN DE PLOTEO DE ARCOS ---
edge_plot_cfg = EdgePlotConfig(
    style_mapping={
        'road': STYLE_EDGE_ROAD,
        'taz_link': STYLE_EDGE_TAZ_LINK,
        'aux_link': STYLE_EDGE_AUX_LINK
    },
    default_style=EdgeStyle(color="silver", linewidth=0.7),
    warning_style=EdgeStyle(color='red', linewidth=2.0),
    arrow_config=EdgeGlobalArrowConfig(arrowstyle='-|>', mutation_scale=25), # Configuración de flechas
    label_configs=[
        EdgeLabelConfig(
            label_id="edge_id_avsnittsidentitet",
            attribute_source='Avsnittsidentitet', # Nombre del atributo para la identidad del tramo
            properties=LabelProperties(font_size=7, font_color='#333333', y_offset=0.00006,
                                     bbox={'facecolor':'white', 'alpha':0.5, 'pad':0.05})
        ),
        EdgeLabelConfig(
            label_id="edge_emme_attributes",
            formatting_function=format_link_attributes_list, # Función importada
            properties=LabelProperties(font_size=6, font_color='navy', y_offset=-0.00006,
                                     bbox={'facecolor':'#EEEEFF', 'alpha':0.6, 'pad':0.05})
        ),
        EdgeLabelConfig(
            label_id="edge_aadt_data",
            formatting_function=format_aadt_label, # Función importada
            properties=LabelProperties(font_size=6, font_color='maroon', y_offset=0.00012,
                                     bbox={'facecolor':'#FFEEEE', 'alpha':0.6, 'pad':0.05})
        )
        # Añade más configuraciones de etiquetas de arcos aquí
    ]
)

# --- CONFIGURACIÓN PARA ETIQUETAS AADT (usada en pipeline.py) (Recomendación 6 - Eliminada redundancia) ---
AADT_DISPLAY_CONFIG = {
    'type': 'both', 
    'total_col_prefix': 'Adt_samtliga_fordon_', 
    'heavy_col_prefix': 'Adt_tunga_fordon_',   
    'total_label_prefix': 'Total: ',
    'heavy_label_prefix': 'Heavy: '
}

# --- PARÁMETROS DE PLOTEO POR DEFECTO (usados en pipeline.py) (Recomendación 6) ---
DEFAULT_PLOT_PARAMS = {
    "figsize": (100, 100), 
    "dpi": 500,
    "selected_year_for_labels": 2022, 
    "bgcolor": "w", # Color de fondo por defecto para el plot
    "show_basemap": True, # Mostrar o no el mapa base por defecto
    # "basemap_source": cx.providers.OpenStreetMap.Mapnik, # Fuente del mapa base por defecto (Recomendación 2 omitida, pero podrías ponerlo aquí si quieres un default simple)
    # "basemap_zoom": "auto", # Zoom del mapa base por defecto
    "save_plot": True, # Guardar o no la figura por defecto
    "close_plot_after_save": True, # Cerrar la figura después de guardarla
    "plot_file_format": "png", # Formato de archivo por defecto
    # "plot_filepath_default_name": "network_visualization", # Nombre base para el archivo guardado (sin extensión)
}

# --- CONFIGURACIÓN DE LEYENDA (Recomendación 3 - Conceptual) ---
# La implementación de la leyenda requiere lógica adicional en NetworkVisualizerOSMnx
LEGEND_CONFIG = {
    "show_legend": False, # Cambiar a True para activar (requiere implementación)
    "location": "best", # Ubicación de la leyenda (ver Matplotlib docs)
    "title": "Leyenda de Red",
    "elements": [ # Lista de tuplas: (objeto_estilo_proxy, "Etiqueta para la leyenda")
                  # Se necesitarían crear 'proxy artists' de Matplotlib para esto.
        # Ejemplo conceptual:
        # ({"marker": "o", "color": STYLE_NODE_TAZ.color, "markersize": STYLE_NODE_TAZ.size / 10}, "Zona TAZ"),
        # ({"marker": "s", "color": STYLE_NODE_BORDER.color, "markersize": STYLE_NODE_BORDER.size / 10}, "Nodo Frontera"),
        # ({"linestyle": STYLE_EDGE_ROAD.linestyle, "color": STYLE_EDGE_ROAD.color, "linewidth": STYLE_EDGE_ROAD.linewidth}, "Carretera"),
    ],
    "font_size": 8,
    "frame_alpha": 0.8, # Transparencia del marco de la leyenda
    "num_columns": 1, # Número de columnas en la leyenda
}

# --- CONTROL DE FLECHAS (Recomendación 5 - Conceptual) ---
# La implementación para un control de flechas selectivo (ej. no flechas para LOG_DIRECTION=0)
# es avanzada y requeriría modificaciones significativas en cómo se dibujan los arcos.
# La EdgeGlobalArrowConfig actual aplica flechas a todos los arcos en un grafo dirigido.
ARROW_LOGIC_CONFIG = {
    "apply_custom_logic": False, # Cambiar a True para activar (requiere implementación)
    "direction_attribute": INPUT_COLUMNS["edge_direction_indicator_column"], # ej. "LOG_DIRECTION"
    "directed_value": 1, # Valor en la columna que indica un arco dirigido con flecha
    # "undirected_arrow_kwargs": None, # o {} para no dibujar flecha, o arrow_kwargs específicos
    # "default_arrow_kwargs": {"arrowstyle": "-|>", "mutation_scale": 20} # Para los dirigidos
}

# --- VALORES POR DEFECTO PARA ATRIBUTOS NO ENCONTRADOS (Recomendación 7) ---
LABEL_DEFAULTS = {
    "text_if_attribute_missing": "N/D", # Texto a mostrar si un attribute_source no se encuentra
}

# --- CONFIGURACIÓN DEL SISTEMA DE COORDENADAS DE ENTRADA ---
# Aunque graph_crs se pasa al constructor, tenerlo aquí puede ser útil para referencia
# o si se necesita en otras partes de la configuración.
INPUT_CRS = "EPSG:3006"