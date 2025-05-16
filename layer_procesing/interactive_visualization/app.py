# interactive_visualization/app.py
import dash
from dash import Input, Output, State, html, dcc
import plotly.graph_objects as go
import geopandas as gpd
from shapely.geometry import Point, LineString
import pandas as pd
import logging
import copy

# Módulos locales
from .layout import create_main_layout
from .data_manager import prepare_data_for_plotly
from .plot_engine import create_graph_figure
from .configs.app_settings import app_settings
from .configs.viz_config import global_viz_config, NodeVizConfig, LinkVizConfig

# --- Variables globales ---
external_nodes_gdf_store = None
external_links_gdf_store = None
loaded_nodes_gdf = None
loaded_links_gdf = None

# Importaciones desde layer_procesing
try:
    from layer_procesing.project_utils import HandlePickle
    from layer_procesing.data_ingestion import GeoPackageHandler
    from layer_procesing.configs.settings import Paths, Filenames
    IMPORTS_OK = True
except ImportError as e:
    IMPORTS_OK = False
    logging.error(f"Error importando módulos de layer_procesing: {e}. ")

# --- Inicialización de la App Dash ---
app = dash.Dash(__name__, title=app_settings.APP_TITLE)
server = app.server
app.layout = create_main_layout()

# --- Funciones de Carga de Datos ---
def load_sample_data():
    # (Misma función que antes)
    nodes_data = {
        'ID': ['N1', 'N2', 'N3', 'N4', 'N5_lonely'],
        'node_type': ['TAZ', 'Intersection', 'aux', 'Border_Node', 'TAZ'],
        'X': [12.00, 12.05, 12.05, 12.10, 12.15],
        'Y': [57.70, 57.72, 57.70, 57.72, 57.75],
        'some_other_attr': ['DatoA', 'DatoB', 'DatoC', 'DatoD', 'DatoE'],
        'name': ['Nodo Uno', 'Cruce A', 'Auxiliar X', 'Frontera Sur', 'Nodo Aislado']
    }
    nodes_geometry = [Point(x, y) for x, y in zip(nodes_data['X'], nodes_data['Y'])]
    nodes_gdf = gpd.GeoDataFrame(nodes_data, geometry=nodes_geometry, crs=app_settings.PLOTLY_MAP_CRS)
    nodes_gdf['coords_tuple'] = nodes_gdf.apply(lambda row: (row['X'], row['Y']), axis=1)


    links_data = {
        'LINK_ID': ['L1', 'L2', 'L3'],
        'INODE': ['N1', 'N2', 'N3'],
        'JNODE': ['N2', 'N3', 'N4'],
        'edge_type': ['road', 'taz_link', 'aux_link'],
        'flow': [100, 50, 75]
    }
    link_geometries = []
    nodes_dict_for_links = {
        node_id: geom for node_id, geom in zip(nodes_gdf[app_settings.INPUT_COLUMNS['node_id']], nodes_gdf.geometry)
    }
    for i in range(len(links_data['LINK_ID'])):
        start_node_id = links_data['INODE'][i]
        end_node_id = links_data['JNODE'][i]
        start_geom = nodes_dict_for_links.get(start_node_id)
        end_geom = nodes_dict_for_links.get(end_node_id)
        if start_geom and end_geom:
            link_geometries.append(LineString([start_geom, end_geom]))
        else:
            link_geometries.append(None)
    links_gdf = gpd.GeoDataFrame(links_data, geometry=link_geometries, crs=app_settings.PLOTLY_MAP_CRS)
    links_gdf = links_gdf.dropna(subset=['geometry'])
    return nodes_gdf, links_gdf

def load_actual_data_from_files():
    global loaded_nodes_gdf, loaded_links_gdf
    if loaded_nodes_gdf is not None and loaded_links_gdf is not None:
        logging.debug("Usando GDFs cacheados de la carga de archivos.")
        return loaded_nodes_gdf, loaded_links_gdf

    if not IMPORTS_OK:
        logging.error("No se pueden cargar datos reales desde archivos: importaciones de layer_procesing fallaron.")
        return gpd.GeoDataFrame(), gpd.GeoDataFrame()
    
    nodes_gdf, links_gdf = None, None
    try:
        retrieved_variables = HandlePickle().open_pickle(
            route=str(Paths.OUTPUT_DIR), filename=Filenames.FINAL_NETWORK
        )
        if 'variables' in retrieved_variables and len(retrieved_variables['variables']) >= 2:
            links_gdf_raw, nodes_gdf_raw = retrieved_variables['variables'][0], retrieved_variables['variables'][1]
            nodes_gdf = gpd.GeoDataFrame(nodes_gdf_raw) if not isinstance(nodes_gdf_raw, gpd.GeoDataFrame) else nodes_gdf_raw
            links_gdf = gpd.GeoDataFrame(links_gdf_raw) if not isinstance(links_gdf_raw, gpd.GeoDataFrame) else links_gdf_raw
            if nodes_gdf.crs is None: nodes_gdf.set_crs(app_settings.INPUT_DATA_CRS, inplace=True, allow_override=True)
            if links_gdf.crs is None: links_gdf.set_crs(app_settings.INPUT_DATA_CRS, inplace=True, allow_override=True)
            logging.info("Datos cargados desde Pickle.")
        else: raise FileNotFoundError("Formato Pickle inesperado.")
    except FileNotFoundError:
        logging.warning("Pickle no encontrado o formato incorrecto. Intentando GeoPackage...")
        try:
            route = Paths.GEOPACKAGES_DIR / Filenames.FINAL_NETWORK
            nodes_gdf = GeoPackageHandler(str(route)).read_layer(layer_name="nodes")
            links_gdf = GeoPackageHandler(str(route)).read_layer(layer_name="links")
            logging.info("Datos cargados desde GeoPackage.")
        except Exception as e_gpkg:
            logging.error(f"FALLO AL CARGAR DATOS (Pickle y GeoPackage): {e_gpkg}")
            nodes_gdf, links_gdf = gpd.GeoDataFrame(), gpd.GeoDataFrame()
    except Exception as e_main:
        logging.error(f"Error inesperado cargando datos: {e_main}")
        nodes_gdf, links_gdf = gpd.GeoDataFrame(), gpd.GeoDataFrame()

    if nodes_gdf.empty: logging.warning("Nodes GDF está vacío después de la carga.")
    if links_gdf.empty: logging.warning("Links GDF está vacío después de la carga.")
    
    loaded_nodes_gdf, loaded_links_gdf = nodes_gdf, links_gdf
    return nodes_gdf, links_gdf

# --- Callback Principal ---
@app.callback(
    Output("network-graph", "figure"),
    Output("node-label-attribute-dropdown", "options"),
    Output("node-label-attribute-dropdown", "value"),
    Input("node-label-visibility-checklist", "value"),
    Input("node-label-attribute-dropdown", "value"),
    Input("link-tooltip-visibility-checklist", "value"),
    Input("app-title", "id") 
)
def update_graph_and_controls(
    node_label_visibility_val, 
    node_label_selected_attr_val,
    link_tooltip_visibility_val,
    _app_title_id_trigger
):
    global external_nodes_gdf_store, external_links_gdf_store, loaded_nodes_gdf, loaded_links_gdf
    
    ctx = dash.callback_context
    triggered_input_id = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else "initial_load"
    logging.info(f"Callback disparado por: {triggered_input_id}")

    current_nodes_gdf, current_links_gdf = None, None
    node_attr_options = []
    
    # Determinar la fuente de datos
    if external_nodes_gdf_store is not None and external_links_gdf_store is not None:
        current_nodes_gdf = external_nodes_gdf_store
        current_links_gdf = external_links_gdf_store
    elif loaded_nodes_gdf is not None and loaded_links_gdf is not None:
        current_nodes_gdf = loaded_nodes_gdf
        current_links_gdf = loaded_links_gdf
    else:
        USE_ACTUAL_DATA_WHEN_DIRECT_RUN = True # O False para datos de ejemplo
        if USE_ACTUAL_DATA_WHEN_DIRECT_RUN:
            current_nodes_gdf, current_links_gdf = load_actual_data_from_files()
        else:
            current_nodes_gdf, current_links_gdf = load_sample_data()
        loaded_nodes_gdf, loaded_links_gdf = current_nodes_gdf, current_links_gdf # Cachear

    # Configuración de visualización activa para esta actualización
    active_viz_config = copy.deepcopy(global_viz_config)
    
    # Determinar el atributo de etiqueta del nodo
    # Si el dropdown fue el que disparó el callback, usar su valor.
    # Si no, usar el valor por defecto de la config o el primero disponible.
    determined_node_label_attr = active_viz_config.nodes.label_attribute

    if current_nodes_gdf is not None and not current_nodes_gdf.empty:
        excluded_cols = [current_nodes_gdf.geometry.name] if hasattr(current_nodes_gdf, 'geometry') else []
        potential_label_cols = [
            col for col in current_nodes_gdf.columns if col not in excluded_cols and \
            (pd.api.types.is_string_dtype(current_nodes_gdf[col]) or pd.api.types.is_numeric_dtype(current_nodes_gdf[col]))
        ]
        node_attr_options = [{'label': col, 'value': col} for col in potential_label_cols]

        if triggered_input_id == "node-label-attribute-dropdown" and node_label_selected_attr_val in potential_label_cols:
            determined_node_label_attr = node_label_selected_attr_val
        elif node_label_selected_attr_val and node_label_selected_attr_val in potential_label_cols: # Si hay un valor previo válido
             determined_node_label_attr = node_label_selected_attr_val
        elif active_viz_config.nodes.label_attribute in potential_label_cols:
            determined_node_label_attr = active_viz_config.nodes.label_attribute
        elif node_attr_options:
            determined_node_label_attr = node_attr_options[0]['value']
        else:
            determined_node_label_attr = None
    else:
        node_attr_options = [{'label': 'N/A', 'value': 'N/A'}]
        determined_node_label_attr = 'N/A'


    if current_nodes_gdf.empty: # No dibujar si no hay nodos
        logging.warning("Datos de nodos vacíos. Mostrando figura vacía.")
        return go.Figure(), node_attr_options, determined_node_label_attr

    # Actualizar configuración activa
    active_viz_config.nodes.show_labels = bool(node_label_visibility_val and 'show' in node_label_visibility_val)
    active_viz_config.nodes.label_attribute = determined_node_label_attr
    active_viz_config.links.show_tooltips = bool(link_tooltip_visibility_val and 'show' in link_tooltip_visibility_val)
    
    logging.info(f"Configuración para el grafo: show_node_labels={active_viz_config.nodes.show_labels}, node_label_attr='{active_viz_config.nodes.label_attribute}', show_link_tooltips={active_viz_config.links.show_tooltips}")

    prepared_plotly_data = prepare_data_for_plotly(current_nodes_gdf, current_links_gdf, active_viz_config)
    fig = create_graph_figure(prepared_plotly_data, active_viz_config)
    
    return fig, node_attr_options, determined_node_label_attr

# --- Función para lanzar la app desde un script externo ---
def launch_dash_app(nodes_input_gdf, links_input_gdf, port=8050, host='127.0.0.1', debug_mode=True):
    global external_nodes_gdf_store, external_links_gdf_store, loaded_nodes_gdf, loaded_links_gdf
    
    if not isinstance(nodes_input_gdf, gpd.GeoDataFrame) or not isinstance(links_input_gdf, gpd.GeoDataFrame):
        logging.error("Error: nodes_input_gdf y links_input_gdf deben ser GeoDataFrames.")
        return

    external_nodes_gdf_store = nodes_input_gdf.copy()
    external_links_gdf_store = links_input_gdf.copy()
    loaded_nodes_gdf, loaded_links_gdf = None, None # Invalidar caché de archivos
    
    if not logging.getLogger().hasHandlers():
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    logging.info(f"Iniciando Dash app en http://{host}:{port}")
    app.run(debug=debug_mode, port=port, host=host)

# --- Punto de Entrada ---
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(module)s - %(message)s')
    app.run(debug=app_settings.DEBUG_MODE, port=8050, host='127.0.0.1')
