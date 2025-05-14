# map_visualization.py: fusión de folium_viz.py, networkx_viz.py y utils.py

import plotly.graph_objects as go
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Callable
import osmnx as ox
import contextily as cx
from shapely.geometry import Point
import pandas as pd
import geopandas as gpd
import logging
import numpy as np # Necesario para manipular geometrías y calcular puntos intermedios
import matplotlib
import osmnx as ox
import inspect
import matplotlib.pyplot as plt
from shapely.geometry import LineString # Necesario para trabajar con geometrías
import networkx as nx
from configs.viz_styles import (
    FOLIUM_MAP_ZOOM_START, folium_style,
    NODE_SIZE, ARROW_STYLE, ARROW_SIZE,
    EDGE_COLOR, NODE_COLOR_DEFAULT
)

# Importar configuraciones. Se hará dentro de la clase o métodos donde se necesite
# para evitar problemas de importación circular a nivel de módulo si settings importa
# elementos de este archivo.
# import settings # Movido a donde se usa para mayor seguridad con ciclos

# --- Dataclasses de Configuración (Definidas aquí para que settings.py pueda importarlas) ---
@dataclass
class LabelProperties:
    """Properties for text labels."""
    font_size: int = 8
    font_color: str = "black"
    font_family: str = "sans-serif"
    x_offset: float = 0.0  # Offset en unidades de datos/mapa
    y_offset: float = 0.0
    horizontal_alignment: str = "center"
    vertical_alignment: str = "center"
    bbox: Optional[Dict[str, Any]] = None  # e.g. {'facecolor':'white', 'alpha':0.7, 'pad':0.1}
    rotation: float = 0.0

@dataclass
class NodeStyle:
    """Styling for a specific node type."""
    color: str = "gray"
    size: int = 30  # Corresponde a 's' en matplotlib.pyplot.scatter
    marker: str = "o" # Marcador de Matplotlib
    alpha: float = 1.0
    zorder: int = 3

@dataclass
class NodeLabelConfig:
    """Configuration for a single node label."""
    label_id: str # Identificador único para esta configuración de etiqueta
    attribute_source: Optional[str] = None # Atributo del nodo para obtener el texto
    node_types: Optional[List[str]] = None  # Si es None, aplica a todos los tipos de nodo
    prefix: str = ""
    suffix: str = ""
    formatting_function: Optional[Callable[[Dict[str, Any]], Optional[str]]] = None
    properties: LabelProperties = field(default_factory=LabelProperties)

@dataclass
class NodePlotConfig:
    """Overall configuration for plotting nodes."""
    style_mapping: Dict[str, NodeStyle] = field(default_factory=dict)
    default_style: NodeStyle = field(default_factory=lambda: NodeStyle(color="lightgray", size=20))
    warning_style: NodeStyle = field(default_factory=lambda: NodeStyle(color="#FFFF00", size=25, zorder=4)) # Amarillo brillante
    label_configs: List[NodeLabelConfig] = field(default_factory=list)

@dataclass
class EdgeStyle:
    """Styling for a specific edge type."""
    color: str = "gray"
    linewidth: float = 1.0
    alpha: float = 0.8
    linestyle: str = '-' # Estilo de línea de Matplotlib
    zorder: int = 2

@dataclass
class EdgeGlobalArrowConfig:
    """Global arrow configuration for ox.plot_graph, usando arrow_kwargs."""
    arrowstyle: str = '-|>' # Estilo de flecha de Matplotlib, e.g. '-|>', 'simple', 'fancy'
    mutation_scale: int = 20 # Afecta el tamaño de la cabeza de la flecha. Relacionado con linewidth.

@dataclass
class EdgeLabelConfig:
    """Configuration for a single edge label."""
    label_id: str # Identificador único
    attribute_source: Optional[str] = None # Atributo del arco
    edge_types: Optional[List[str]] = None  # Si es None, aplica a todos los tipos de arco
    prefix: str = ""
    suffix: str = ""
    formatting_function: Optional[Callable[[Dict[str, Any], Optional[int], Optional[Dict[str,Any]]], Optional[str]]] = None
    properties: LabelProperties = field(default_factory=LabelProperties)

@dataclass
class EdgePlotConfig:
    """Overall configuration for plotting edges."""
    style_mapping: Dict[str, EdgeStyle] = field(default_factory=dict)
    default_style: EdgeStyle = field(default_factory=lambda: EdgeStyle(color="lightgray", linewidth=0.8))
    warning_style: EdgeStyle = field(default_factory=lambda: EdgeStyle(color="#FFFF00", linewidth=1.5, zorder=2)) # Amarillo brillante
    label_configs: List[EdgeLabelConfig] = field(default_factory=list)
    arrow_config: EdgeGlobalArrowConfig = field(default_factory=EdgeGlobalArrowConfig)

# --- Funciones de Formateo de Etiquetas ---
# Estas funciones NO deben importar 'settings' directamente a nivel de módulo
# si 'settings' las importa a ellas, para evitar ciclos.
# El valor por defecto para atributos no encontrados se manejará en _draw_labels.

def format_node_taz_salidas_entradas(data: Dict[str, Any]) -> Optional[str]:
    salidas = data.get('salidas') # Obtener el valor, puede ser None
    entradas = data.get('entradas') # Obtener el valor, puede ser None
    if salidas is None and entradas is None: # Si ambos son None, no mostrar nada o un placeholder
        return None # Opcionalmente, podrías devolver "S/E: (N/A)" aquí si siempre quieres la etiqueta
    
    s_str = str(salidas) if salidas is not None else "N/A"
    e_str = str(entradas) if entradas is not None else "N/A"
    return f"S/E: ({s_str}/{e_str})"

def format_link_attributes_list(data: Dict[str, Any], selected_year: Optional[int]=None, aadt_config: Optional[Dict[str, Any]]=None) -> str:
    lanes = data.get('emme_LANES', "N/A") # Usar el default de .get()
    speed = data.get('emme_@hast', "N/A") 
    vtype = data.get('emme_@vtyp', "N/A") 
    return f"Lanes: {lanes}\nSpeed: {speed} km/h\nType: {vtype}"

def format_aadt_label(data: Dict[str, Any], selected_year: Optional[int], aadt_config: Optional[Dict[str, Any]]) -> Optional[str]:
    if not selected_year or not aadt_config: return None
    labels = []
    year_str = str(selected_year)
    
    show_total = aadt_config.get('type') in ['total', 'both']
    show_heavy = aadt_config.get('type') in ['heavy', 'both']

    # El valor por defecto si la columna no existe se manejará con .get()
    # El reemplazo final por settings.LABEL_DEFAULTS["text_if_attribute_missing"]
    # se hará en _draw_labels si el valor final es None o un placeholder específico.
    default_val_placeholder = "_MISSING_" # Un placeholder interno

    if show_total:
        col_name = f"{aadt_config.get('total_col_prefix', 'Adt_samtliga_fordon_')}{year_str}"
        value = data.get(col_name, default_val_placeholder)
        prefix = aadt_config.get('total_label_prefix', 'Total AADT: ')
        labels.append(f"{prefix}{value}")
    
    if show_heavy:
        col_name = f"{aadt_config.get('heavy_col_prefix', 'Adt_tunga_fordon_')}{year_str}"
        value = data.get(col_name, default_val_placeholder)
        prefix = aadt_config.get('heavy_label_prefix', 'Heavy AADT: ')
        labels.append(f"{prefix}{value}")
        
    processed_labels = "\n".join(labels)
    # Si todos los valores eran placeholder, la etiqueta podría ser solo placeholders.
    # _draw_labels se encargará de reemplazar default_val_placeholder con el texto de settings.
    return processed_labels if labels else None


# --- Clase Principal de Visualización ---
class NetworkVisualizerOSMnx:
    def __init__(self,
                 nodes_gdf: gpd.GeoDataFrame,
                 links_gdf: gpd.GeoDataFrame,
                 node_plot_config: NodePlotConfig,
                 edge_plot_config: EdgePlotConfig,
                 graph_crs: Optional[str] = None):
        
        # Importar settings aquí, dentro del __init__ o de los métodos donde se use.
        # Esto asegura que settings.py se cargue completamente antes de acceder a sus atributos.
        import configs.settings as app_settings # Usar un alias para evitar confusión con un módulo 'settings' global

        self.logger = logging.getLogger(__name__)
        self.raw_nodes_gdf = nodes_gdf.copy()
        self.raw_links_gdf = links_gdf.copy()
        self.node_plot_config = node_plot_config
        self.edge_plot_config = edge_plot_config
        self.graph_crs = graph_crs if graph_crs is not None else app_settings.INPUT_CRS
        self.app_settings = app_settings # Guardar referencia a las configuraciones

        if not hasattr(self.app_settings, 'INPUT_COLUMNS'):
            self.logger.error("app_settings.INPUT_COLUMNS no está definido.")
            raise AttributeError("app_settings.INPUT_COLUMNS no está definido.")
        self.input_cols = self.app_settings.INPUT_COLUMNS

        self.graph = self._create_graph()

    def _create_graph(self) -> nx.MultiDiGraph:
        # (El contenido de _create_graph permanece igual que en la versión anterior,
        #  usando self.input_cols para los nombres de las columnas)
        self.logger.info("Creating graph from GeoDataFrames using column names from settings.INPUT_COLUMNS...")
        nodes_for_graph = self.raw_nodes_gdf.copy()
        links_for_graph = self.raw_links_gdf.copy()

        self.logger.debug(f"Initial raw_nodes_gdf index type: {type(nodes_for_graph.index)}, name: {nodes_for_graph.index.name}")
        if isinstance(nodes_for_graph.index, pd.MultiIndex):
            self.logger.error("The raw_nodes_gdf has a MultiIndex. This is not supported.")
            raise ValueError("Nodes GeoDataFrame (raw_nodes_gdf) has a MultiIndex.")
        if isinstance(nodes_for_graph.columns, pd.MultiIndex):
            self.logger.warning("Nodes GeoDataFrame (raw_nodes_gdf) has MultiIndex columns. Attempting to flatten.")
            try:
                nodes_for_graph.columns = ['_'.join(map(str, col)).strip('_') if isinstance(col, tuple) else str(col) for col in nodes_for_graph.columns.to_flat_index()]
            except Exception as e:
                self.logger.error(f"Failed to flatten node columns: {e}")
                raise

        current_node_id_type = int 
        self.logger.info(f"Targeting node ID type: {current_node_id_type}")

        node_id_col_internal = self.input_cols.get("node_id_internal", "osmid")
        node_original_id_col = self.input_cols.get("node_original_id_column")

        if node_id_col_internal not in nodes_for_graph.columns:
            if node_original_id_col and node_original_id_col in nodes_for_graph.columns:
                self.logger.info(f"Creating '{node_id_col_internal}' from column '{node_original_id_col}'.")
                source_ids = nodes_for_graph[node_original_id_col]
            else:
                self.logger.info(f"Creating '{node_id_col_internal}' from nodes_for_graph.index.")
                source_ids = nodes_for_graph.index
            
            try:
                nodes_for_graph[node_id_col_internal] = source_ids.astype(current_node_id_type)
            except ValueError as e:
                self.logger.warning(f"Failed to cast source for '{node_id_col_internal}' to {current_node_id_type}. Error: {e}. Falling back to string.")
                nodes_for_graph[node_id_col_internal] = source_ids.astype(str)
                current_node_id_type = str 
        else: 
            self.logger.info(f"Using existing column '{node_id_col_internal}'. Original dtype: {nodes_for_graph[node_id_col_internal].dtype}")
            try:
                nodes_for_graph[node_id_col_internal] = nodes_for_graph[node_id_col_internal].astype(current_node_id_type)
            except ValueError as e:
                self.logger.warning(f"Failed to cast existing '{node_id_col_internal}' to {current_node_id_type}. Error: {e}. Falling back to string.")
                nodes_for_graph[node_id_col_internal] = nodes_for_graph[node_id_col_internal].astype(str)
                current_node_id_type = str
        
        nodes_for_graph = nodes_for_graph.set_index(node_id_col_internal, drop=False)
        self.logger.debug(f"Nodes GDF prepared. Index name: {nodes_for_graph.index.name}, Index dtype: {nodes_for_graph.index.dtype}")

        if not nodes_for_graph.index.is_unique:
            self.logger.warning(f"Node index ('{node_id_col_internal}') is not unique. Duplicates: {nodes_for_graph.index[nodes_for_graph.index.duplicated()].unique().tolist()}")

        node_x_col = self.input_cols.get("node_x_coord_column", "x")
        node_y_col = self.input_cols.get("node_y_coord_column", "y")
        node_geom_col = self.input_cols.get("node_geometry_column", "geometry")

        if 'x' not in nodes_for_graph.columns or 'y' not in nodes_for_graph.columns:
            if node_x_col in nodes_for_graph.columns and node_y_col in nodes_for_graph.columns:
                nodes_for_graph['x'] = nodes_for_graph[node_x_col]
                nodes_for_graph['y'] = nodes_for_graph[node_y_col]
                self.logger.info(f"Using columns '{node_x_col}' and '{node_y_col}' for node coordinates.")
            elif node_geom_col in nodes_for_graph.columns and hasattr(nodes_for_graph[node_geom_col], 'geom_type'):
                self.logger.info(f"Attempting to derive x,y coordinates from node geometry column '{node_geom_col}'.")
                is_point = nodes_for_graph[node_geom_col].geom_type == 'Point'
                is_valid_geom = nodes_for_graph[node_geom_col].is_valid & (~nodes_for_graph[node_geom_col].is_empty)
                valid_points_mask = is_point & is_valid_geom
                if not valid_points_mask.all():
                    self.logger.warning(f"{(~valid_points_mask).sum()} node geometries in '{node_geom_col}' are not valid points. Coordinates might be missing.")
                nodes_for_graph['x'] = nodes_for_graph[node_geom_col].x
                nodes_for_graph['y'] = nodes_for_graph[node_geom_col].y
                if nodes_for_graph.loc[valid_points_mask, ['x', 'y']].isnull().any().any():
                     self.logger.error("NaN values found in derived 'x' or 'y' coordinates from valid node geometries.")
            else:
                self.logger.error(f"Node coordinates ('x', 'y') could not be determined from configured columns: "
                                   f"X='{node_x_col}', Y='{node_y_col}', Geometry='{node_geom_col}'.")
                raise ValueError("Node coordinates not found or invalid based on settings.INPUT_COLUMNS.")
        
        edge_from_col = self.input_cols.get("edge_from_node_column", "INODE")
        edge_to_col = self.input_cols.get("edge_to_node_column", "JNODE")
        edge_geom_col = self.input_cols.get("edge_geometry_column", "geometry")
        edge_key_col = self.input_cols.get("edge_key_column", "key") 

        if not ({edge_from_col, edge_to_col}.issubset(links_for_graph.columns)):
            self.logger.error(f"Links GeoDataFrame must have '{edge_from_col}' and '{edge_to_col}' columns as per settings.")
            raise ValueError(f"Links GDF missing '{edge_from_col}' or '{edge_to_col}'.")
        if links_for_graph[[edge_from_col, edge_to_col]].isnull().any().any():
            self.logger.error(f"{links_for_graph[[edge_from_col, edge_to_col]].isnull().any(axis=1).sum()} links have NaN in from/to node columns.")
            raise ValueError(f"NaN values in '{edge_from_col}'/'{edge_to_col}' columns of links.")

        links_for_graph = links_for_graph.rename(columns={edge_from_col: 'u', edge_to_col: 'v'})
        self.logger.debug(f"Casting link 'u' and 'v' columns to node ID type: {current_node_id_type}")
        try:
            links_for_graph['u'] = links_for_graph['u'].astype(current_node_id_type)
            links_for_graph['v'] = links_for_graph['v'].astype(current_node_id_type)
        except Exception as e:
            self.logger.error(f"Failed to cast 'u'/'v' in links to {current_node_id_type}. Error: {e}")
            raise

        missing_u = links_for_graph[~links_for_graph['u'].isin(nodes_for_graph.index)]
        missing_v = links_for_graph[~links_for_graph['v'].isin(nodes_for_graph.index)]
        if not missing_u.empty: self.logger.warning(f"{len(missing_u)} 'u' values in links not in node index. Examples: {missing_u['u'].unique()[:3]}")
        if not missing_v.empty: self.logger.warning(f"{len(missing_v)} 'v' values in links not in node index. Examples: {missing_v['v'].unique()[:3]}")
        
        if edge_geom_col not in links_for_graph.columns:
            self.logger.warning(f"Links GeoDataFrame does not have a '{edge_geom_col}' column as per settings. "
                                "OSMnx might create straight-line geometries if nodes have x,y.")
        elif links_for_graph[edge_geom_col].isnull().any():
            self.logger.warning(f"{links_for_graph[edge_geom_col].isnull().sum()} links have null geometry in '{edge_geom_col}'.")
        
        self.logger.info("Calling ox.graph_from_gdfs...")
        try:
            if edge_key_col not in links_for_graph.columns:
                self.logger.info(f"Adding '{edge_key_col}' column to links_for_graph from its index (user fix).")
                links_for_graph[edge_key_col] = links_for_graph.index 
            
            self.logger.info(f"Setting MultiIndex ['u', 'v', '{edge_key_col}'] on links_for_graph (user fix).")
            links_for_graph.set_index(['u', 'v', edge_key_col], inplace=True) 
            
            G = ox.graph_from_gdfs(gdf_nodes=nodes_for_graph, gdf_edges=links_for_graph, graph_attrs={'crs': self.graph_crs})
        except Exception as e:
            self.logger.error(f"Error during ox.graph_from_gdfs: {e}", exc_info=True)
            self.logger.error("--- links_for_graph details before error ---")
            self.logger.error(f"Index: {links_for_graph.index.names if isinstance(links_for_graph.index, pd.MultiIndex) else links_for_graph.index.name}")
            self.logger.error(f"Columns: {links_for_graph.columns.tolist()}")
            self.logger.error(f"Sample head:\n{links_for_graph.head().to_string(max_cols=10)}")
            raise e

        self.logger.info(f"Graph successfully created with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
        return G

    def _get_render_attributes(self, G: nx.MultiDiGraph, element_type: str) -> tuple:
        # (El contenido de _get_render_attributes permanece igual que en la versión anterior,
        #  usando self.input_cols para los nombres de las columnas de tipo)
        node_type_col = self.input_cols.get("node_type_column", "node_type")
        edge_type_col = self.input_cols.get("edge_type_column", "edge_type")

        if element_type == 'nodes':
            elements = list(G.nodes(data=True))
            config = self.node_plot_config
            type_attr_name = node_type_col
            colors, sizes, alphas, zorders = [], [], [], []
        elif element_type == 'edges':
            elements = list(G.edges(data=True, keys=True))
            config = self.edge_plot_config
            type_attr_name = edge_type_col
            colors, linewidths, alphas, zorders, linestyles_list = [], [], [], [], []
        else:
            raise ValueError("element_type must be 'nodes' or 'edges'")

        for item_data_tuple in elements:
            data = item_data_tuple[-1]
            item_type_value = data.get(type_attr_name) 
            style = None
            if item_type_value and item_type_value in config.style_mapping:
                style = config.style_mapping[item_type_value]
            elif item_type_value:
                self.logger.warning(f"No style for {element_type[:-1]}_type '{item_type_value}' (from column '{type_attr_name}'). Using warning. ID: {item_data_tuple[0]}")
                style = config.warning_style
            else:
                self.logger.debug(f"No '{type_attr_name}' found for {element_type[:-1]} or type is None. Using default. ID: {item_data_tuple[0]}")
                style = config.default_style

            colors.append(style.color)
            alphas.append(style.alpha)
            zorders.append(style.zorder)

            if element_type == 'nodes':
                sizes.append(style.size)
            else:
                linewidths.append(style.linewidth)
                linestyles_list.append(style.linestyle)
        
        if element_type == 'nodes':
            return colors, sizes, max(alphas) if alphas else 1.0, max(zorders) if zorders else 1
        else: 
            default_linestyle_for_plot_call = config.default_style.linestyle
            if config.style_mapping:
                # Intenta obtener un linestyle de los estilos mapeados; si no, usa el del default_style.
                # Esto es una heurística; podrías querer una lógica más explícita si es crucial.
                first_style_key = next(iter(config.style_mapping), None)
                if first_style_key:
                    default_linestyle_for_plot_call = config.style_mapping[first_style_key].linestyle
            elif linestyles_list: 
                default_linestyle_for_plot_call = linestyles_list[0]
            return colors, linewidths, max(alphas) if alphas else 1.0, max(zorders) if zorders else 1, default_linestyle_for_plot_call

    def _draw_labels(self, ax: plt.Axes, G: nx.MultiDiGraph, element_type: str,
                     selected_year: Optional[int] = None,
                     aadt_config: Optional[Dict[str,Any]] = None):
        
        # Acceder a LABEL_DEFAULTS desde la instancia de settings guardada
        text_if_missing = self.app_settings.LABEL_DEFAULTS.get("text_if_attribute_missing", "N/A")
        # Placeholder interno usado por format_aadt_label
        internal_missing_placeholder = "_MISSING_" 

        node_type_col = self.input_cols.get("node_type_column", "node_type")
        edge_type_col = self.input_cols.get("edge_type_column", "edge_type")
        edge_geom_col_name = self.input_cols.get("edge_geometry_column", "geometry")

        if element_type == 'nodes':
            elements_iterator = G.nodes(data=True)
            config_labels = self.node_plot_config.label_configs
            get_pos = lambda data: (data.get('x'), data.get('y')) 
            item_type_attr_name = node_type_col
        elif element_type == 'edges':
            elements_iterator = G.edges(data=True, keys=True)
            config_labels = self.edge_plot_config.label_configs
            def get_pos_edge(data_edge, u_node, v_node): 
                if edge_geom_col_name in data_edge and \
                   hasattr(data_edge[edge_geom_col_name], 'centroid') and \
                   data_edge[edge_geom_col_name].is_valid:
                    geom = data_edge[edge_geom_col_name]
                    return geom.centroid.x, geom.centroid.y
                u_data = G.nodes.get(u_node, {}) 
                v_data = G.nodes.get(v_node, {}) 
                return (u_data.get('x',0) + v_data.get('x',0)) / 2, (u_data.get('y',0) + v_data.get('y',0)) / 2
            item_type_attr_name = edge_type_col
        else: return

        for item_key_data_tuple in elements_iterator:
            data = item_key_data_tuple[-1]
            item_type_value = data.get(item_type_attr_name)
            
            pos_x, pos_y = (None, None)
            log_identifier = ""
            if element_type == 'nodes':
                log_identifier = item_key_data_tuple[0] 
                pos_x, pos_y = get_pos(data)
            else: 
                u, v, k = item_key_data_tuple[0], item_key_data_tuple[1], item_key_data_tuple[2]
                log_identifier = (u,v,k)
                pos_x, pos_y = get_pos_edge(data, u, v)

            if pos_x is None or pos_y is None or pd.isna(pos_x) or pd.isna(pos_y):
                self.logger.warning(f"Skipping label for {element_type[:-1]} {log_identifier} due to invalid/missing position ({pos_x}, {pos_y}).")
                continue

            for lbl_config in config_labels:
                if element_type == 'nodes' and lbl_config.node_types and item_type_value not in lbl_config.node_types:
                    continue
                if element_type == 'edges' and lbl_config.edge_types and item_type_value not in lbl_config.edge_types:
                    continue

                label_text_val = None
                if lbl_config.formatting_function:
                    func_kwargs = {'data': data}
                    if element_type == 'edges': 
                        func_kwargs['selected_year'] = selected_year
                        func_kwargs['aadt_config'] = aadt_config
                    
                    sig = inspect.signature(lbl_config.formatting_function)
                    valid_kwargs = {k: v for k, v in func_kwargs.items() if k in sig.parameters}
                    try:
                        label_text_val = lbl_config.formatting_function(**valid_kwargs)
                    except Exception as e_fmt:
                        self.logger.error(f"Error in formatting_function for label '{lbl_config.label_id}' on {log_identifier}: {e_fmt}", exc_info=True)
                        continue
                elif lbl_config.attribute_source:
                    # Usar .get() con un placeholder si el atributo no existe
                    label_text_val = data.get(lbl_config.attribute_source, internal_missing_placeholder)
                
                # Reemplazar el placeholder interno con el texto de settings si es necesario
                if label_text_val == internal_missing_placeholder:
                    label_text_val = text_if_missing
                    if lbl_config.attribute_source: # Solo loguear si se esperaba un atributo
                         self.logger.debug(f"Attribute '{lbl_config.attribute_source}' for label '{lbl_config.label_id}' not found in data for {log_identifier}. Using default: '{text_if_missing}'.")

                if label_text_val is None or str(label_text_val).strip() == "": continue
                
                final_text = f"{lbl_config.prefix}{str(label_text_val)}{lbl_config.suffix}" 
                props = lbl_config.properties
                ax.text(pos_x + props.x_offset, pos_y + props.y_offset, final_text,
                        fontsize=props.font_size, color=props.font_color, family=props.font_family,
                        ha=props.horizontal_alignment, va=props.vertical_alignment, rotation=props.rotation,
                        bbox=props.bbox, zorder=10) 

    def plot(self, ax: Optional[plt.Axes] = None, **kwargs):
        # Importar settings aquí para asegurar que esté cargado
        import configs.settings as app_settings

        plot_params = app_settings.DEFAULT_PLOT_PARAMS.copy()
        plot_params.update(kwargs)

        figsize = plot_params.get("figsize")
        bgcolor = plot_params.get("bgcolor")
        selected_year_for_labels = plot_params.get("selected_year_for_labels")
        aadt_label_config = plot_params.get("aadt_label_config")
        save_path = plot_params.get("save_path")
        dpi = plot_params.get("dpi")
        show_basemap = plot_params.get("show_basemap", True)
        # Usar el default_source de BASEMAP_CONFIG si existe, sino OSM.Mapnik
        basemap_source_cfg = app_settings.BASEMAP_CONFIG.get("default_source") if hasattr(app_settings, 'BASEMAP_CONFIG') else None
        basemap_source = plot_params.get("basemap_source", basemap_source_cfg if basemap_source_cfg else cx.providers.OpenStreetMap.Mapnik)
        basemap_zoom_cfg = app_settings.BASEMAP_CONFIG.get("default_zoom") if hasattr(app_settings, 'BASEMAP_CONFIG') else "auto"
        basemap_zoom = plot_params.get("basemap_zoom", basemap_zoom_cfg)
        
        if ax is None:
            fig, ax_plot = plt.subplots(figsize=figsize, facecolor=bgcolor) 
            ax_plot.set_facecolor(bgcolor)
        else:
            fig = ax.get_figure()
            ax_plot = ax 

        G = self.graph
        if G is None:
            self.logger.error("Graph object is None. Cannot plot.")
            return None, None

        self.logger.info(f"Starting plot. Graph: {G.number_of_nodes()} N, {G.number_of_edges()} E. CRS: {G.graph.get('crs', self.graph_crs)}")

        valid_graph = True
        for node_id, data in G.nodes(data=True):
            if 'x' not in data or 'y' not in data or pd.isna(data['x']) or pd.isna(data['y']):
                self.logger.error(f"Node {node_id} missing/NaN 'x'/'y'. Data: {data}")
                valid_graph = False; break
        if G.number_of_edges() > 0: 
            edge_geom_col_name = self.input_cols.get("edge_geometry_column", "geometry")
            for u, v, key, data in G.edges(data=True, keys=True):
                if edge_geom_col_name in data and (data[edge_geom_col_name] is None or \
                   (hasattr(data[edge_geom_col_name], 'is_empty') and data[edge_geom_col_name].is_empty)):
                    self.logger.warning(f"Edge ({u},{v},{key}) has None or empty '{edge_geom_col_name}'. OSMnx might create straight line. Geom: {data.get(edge_geom_col_name)}")
        
        if not valid_graph:
            self.logger.error("Graph data for nodes (x,y) is invalid. Aborting plot."); return fig, ax_plot

        self.logger.info("Attempting simplified ox.plot_graph call for debugging...")
        try:
            simple_nc = ['blue'] * G.number_of_nodes() if G.number_of_nodes() > 0 else 'blue'
            simple_ns = [10] * G.number_of_nodes() if G.number_of_nodes() > 0 else 10
            simple_ec = ['gray'] * G.number_of_edges() if G.number_of_edges() > 0 else 'gray'
            simple_elw = [1] * G.number_of_edges() if G.number_of_edges() > 0 else 1
            
            if not isinstance(G, (nx.Graph, nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph)):
                 self.logger.error(f"Graph object not a recognized NetworkX type: {type(G)}"); raise TypeError("Invalid graph type.")

            ox.plot_graph(G, ax=ax_plot, show=False, close=False, save=False,
                          node_color=simple_nc, node_size=simple_ns,
                          edge_color=simple_ec, edge_linewidth=simple_elw,
                          ) 
            self.logger.info("Simplified ox.plot_graph call SUCCEEDED.")
        except Exception as e: 
            self.logger.error(f"Error during SIMPLIFIED ox.plot_graph: {e}", exc_info=True)
        
        self.logger.info("Proceeding with detailed plotting attributes...")
        ax_plot.cla() 

        node_colors, node_sizes, node_alpha, node_zorder = self._get_render_attributes(G, 'nodes')
        edge_colors, edge_linewidths, edge_alpha, edge_zorder, edge_linestyle_val = self._get_render_attributes(G, 'edges')
        
        arrow_cfg_obj = self.edge_plot_config.arrow_config # Renombrado para claridad
        current_arrow_kwargs = {
            "arrowstyle": arrow_cfg_obj.arrowstyle,
            "mutation_scale": arrow_cfg_obj.mutation_scale
        }
        current_edge_kwargs = {}
        if edge_linestyle_val: 
             current_edge_kwargs['linestyle'] = edge_linestyle_val
        
        try:
            nc = node_colors if G.number_of_nodes() > 0 else 'blue'
            ns = node_sizes if G.number_of_nodes() > 0 else 10
            ec = edge_colors if G.number_of_edges() > 0 else 'gray'
            elw = edge_linewidths if G.number_of_edges() > 0 else 1

            # TODO aqui elimine el edge_zorder
            # TODO arrow_kwargs=current_arrow_kwargs if G.is_directed() and G.number_of_edges() > 0 else None, 
            # TODO edge_kwargs=current_edge_kwargs if current_edge_kwargs else None
            ox.plot_graph(
                G, ax=ax_plot, show=False, close=False, save=False, bgcolor=bgcolor,
                node_color=nc, node_size=ns, 
                node_alpha=node_alpha, node_zorder=node_zorder,
                edge_color=ec, edge_linewidth=elw, 
                edge_alpha=edge_alpha,
            )
            self.logger.info("Detailed ox.plot_graph call completed.")
        except Exception as e_detailed: 
            self.logger.error(f"Error during DETAILED ox.plot_graph: {e_detailed}", exc_info=True)
            raise e_detailed 

        ax_plot.set_aspect('equal', adjustable='datalim')
        ax_plot.set_axis_off() 

        self.logger.info("Drawing labels...")
        try:
            self._draw_labels(ax_plot, G, 'nodes', selected_year_for_labels, aadt_label_config)
            self._draw_labels(ax_plot, G, 'edges', selected_year_for_labels, aadt_label_config)
            self.logger.info("Labels drawn.")
        except Exception as e_label: 
            self.logger.error(f"Error during _draw_labels: {e_label}", exc_info=True)

        if show_basemap:
            self.logger.info(f"Adding basemap from {basemap_source} with CRS {self.graph_crs}")
            try:
                # Usar configuración de atribución de settings si está disponible
                attr_text = self.app_settings.BASEMAP_CONFIG.get("attribution_text", None) if hasattr(self.app_settings, 'BASEMAP_CONFIG') else None
                attr_size = self.app_settings.BASEMAP_CONFIG.get("attribution_size", 5) if hasattr(self.app_settings, 'BASEMAP_CONFIG') else 5
                
                cx.add_basemap(ax_plot, crs=self.graph_crs, source=basemap_source, zoom=basemap_zoom, 
                               attribution=attr_text, attribution_size=attr_size)
                self.logger.info("Basemap added.")
            except Exception as e_basemap:
                self.logger.error(f"Failed to add basemap: {e_basemap}", exc_info=True)
        
        if hasattr(self.app_settings, 'LEGEND_CONFIG') and self.app_settings.LEGEND_CONFIG.get("show_legend", False):
            self.logger.info("Attempting to draw legend (requires proxy artist implementation).")
            try:
                legend_elements = []
                # Lógica para crear 'proxy artists' de Matplotlib
                # basada en self.app_settings.LEGEND_CONFIG["elements"]
                # Esta parte sigue siendo conceptual y necesita implementación detallada.
                # Ejemplo:
                # for style_info in self.app_settings.LEGEND_CONFIG.get("elements", []):
                #     # style_info podría ser un dict {"marker": "o", "color": "blue", ...} o tu NodeStyle/EdgeStyle
                #     # y el segundo elemento de la tupla es el label_str
                #     # proxy = plt.Line2D([0], [0], **style_info_matplotlib_kwargs, label=label_str)
                #     # legend_elements.append(proxy)
                
                if legend_elements: 
                    ax_plot.legend(handles=legend_elements,
                                 loc=self.app_settings.LEGEND_CONFIG.get("location", "best"),
                                 title=self.app_settings.LEGEND_CONFIG.get("title"),
                                 fontsize=self.app_settings.LEGEND_CONFIG.get("font_size"),
                                 framealpha=self.app_settings.LEGEND_CONFIG.get("frame_alpha"),
                                 ncol=self.app_settings.LEGEND_CONFIG.get("num_columns", 1))
                    self.logger.info("Legend drawn (conceptual).")
                else:
                    self.logger.warning("Legend configured to show, but no elements were prepared (proxy artist logic missing or elements empty).")
            except Exception as e_legend:
                self.logger.error(f"Error drawing legend: {e_legend}", exc_info=True)

        try:
            fig.tight_layout()
        except Exception as e_layout: 
            self.logger.warning(f"Could not apply tight_layout: {e_layout}")

        if save_path and plot_params.get("save_plot", True): # Usa save_plot de plot_params
            self.logger.info(f"Saving plot to {save_path}")
            plt.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor=fig.get_facecolor())
            if plot_params.get("close_plot_after_save", True): # Usa close_plot de plot_params
                plt.close(fig)
        elif not (save_path and plot_params.get("save_plot", True)): # Si no se guarda, mostrar
             plt.show()
        
        return fig, ax_plot


class NetworkXPlotter:
    def __init__(self, node_size=NODE_SIZE, arrowstyle=ARROW_STYLE,
                 arrowsize=ARROW_SIZE, edge_color=EDGE_COLOR, node_color=NODE_COLOR_DEFAULT):
        self.node_size = node_size
        self.arrowstyle = arrowstyle
        self.arrowsize = arrowsize
        self.edge_color = edge_color
        self.node_color = node_color

    def plot(self, G, with_labels=False, title='Grafo Vial'):
        if G.number_of_nodes() == 0:
            logger.warning('El grafo no tiene nodos para visualizar.')
            return

        pos = {n: (d.get('coord_x', 0), d.get('coord_y', 0)) for n, d in G.nodes(data=True)}
        plt.figure(figsize=(12, 8))
        nx.draw(G, pos,
                with_labels=with_labels,
                node_size=self.node_size,
                arrowstyle=self.arrowstyle,
                arrowsize=self.arrowsize,
                node_color=self.node_color,
                edge_color=self.edge_color)
        plt.title(title)
        plt.tight_layout()

        if SAVE_FIGURE_INSTEAD_OF_SHOW:
            plt.savefig(SAVE_FILENAME)
            logger.info(f'Visualización NetworkX guardada en {SAVE_FILENAME}.')
        else:
            plt.show()
            logger.info('Visualización NetworkX mostrada.')
