# interactive_visualization/data_manager.py
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point, LineString
import logging # Añadido para logging

# Importar configuraciones
from .configs.app_settings import app_settings
from .configs.viz_config import GlobalVizConfig

def get_node_tooltip_text(node_row, node_id_col, node_type_col, tooltip_attrs: list):
    """Genera texto para el tooltip de un nodo basado en tooltip_attrs."""
    parts = []
    for attr in tooltip_attrs:
        value = node_row.get(attr, "N/A")
        parts.append(f"<b>{attr}:</b> {value}") # Poner en negrita el nombre del atributo
    return "<br>".join(parts) if parts else "Nodo"


def get_link_tooltip_text(link_row, link_id_col, link_type_col, from_node_col, to_node_col, tooltip_attrs: list):
    """Genera texto para el tooltip de un enlace basado en tooltip_attrs."""
    parts = []
    for attr in tooltip_attrs:
        value = link_row.get(attr, "N/A")
        parts.append(f"<b>{attr}:</b> {value}") # Poner en negrita el nombre del atributo
    return "<br>".join(parts) if parts else "Enlace"


def prepare_data_for_plotly(
    nodes_gdf: gpd.GeoDataFrame,
    links_gdf: gpd.GeoDataFrame,
    current_viz_config: GlobalVizConfig
):
    """
    Prepara los datos de nodos y enlaces para ser consumidos por Plotly Scattermapbox.
    """
    prepared_data = {
        "nodes": {"lons": [], "lats": [], "colors": [], "sizes": [], "symbols": [], "texts": [], "tooltips": [], "ids": []},
        "links": {"lons": [], "lats": [], "colors": [], "widths": [], "dash_styles": [], "tooltips": [], "ids": []}
    }

    if nodes_gdf is None or links_gdf is None or nodes_gdf.empty:
        logging.warning("GeoDataFrame de nodos o enlaces está vacío o es None al inicio de prepare_data_for_plotly.")
        # Devolver datos vacíos pero con estructura correcta para evitar errores en plot_engine
        if links_gdf is not None and not links_gdf.empty: # Si solo los nodos están vacíos, procesar links
             pass # La lógica de links se ejecutará
        else: # Si ambos o los nodos están vacíos, no hay mucho que hacer
            prepared_data["links"]["colors"] = [current_viz_config.links.default_style.color]
            prepared_data["links"]["widths"] = [current_viz_config.links.default_style.width]
            prepared_data["links"]["dash_styles"] = [current_viz_config.links.default_style.dash]
            return prepared_data


    # --- Transformación CRS ---
    try:
        # Si el CRS de entrada es None, no se puede transformar. Asumir que ya está en EPSG:4326 o que el usuario lo manejará.
        if nodes_gdf.crs:
            if str(nodes_gdf.crs).upper() != app_settings.PLOTLY_MAP_CRS:
                nodes_gdf_proj = nodes_gdf.to_crs(app_settings.PLOTLY_MAP_CRS)
            else:
                nodes_gdf_proj = nodes_gdf.copy()
        else:
            logging.warning(f"Nodes GDF no tiene CRS definido. Asumiendo que las coordenadas son compatibles con {app_settings.PLOTLY_MAP_CRS}.")
            nodes_gdf_proj = nodes_gdf.copy()

        if links_gdf.crs:
            if str(links_gdf.crs).upper() != app_settings.PLOTLY_MAP_CRS:
                links_gdf_proj = links_gdf.to_crs(app_settings.PLOTLY_MAP_CRS)
            else:
                links_gdf_proj = links_gdf.copy()
        else:
            logging.warning(f"Links GDF no tiene CRS definido. Asumiendo que las coordenadas son compatibles con {app_settings.PLOTLY_MAP_CRS}.")
            links_gdf_proj = links_gdf.copy()
            
    except Exception as e:
        logging.error(f"Error en la transformación CRS en data_manager: {e}")
        # Devolver datos vacíos pero con estructura correcta
        prepared_data["links"]["colors"] = [current_viz_config.links.default_style.color]
        prepared_data["links"]["widths"] = [current_viz_config.links.default_style.width]
        prepared_data["links"]["dash_styles"] = [current_viz_config.links.default_style.dash]
        return prepared_data

    # --- Nombres de Columnas ---
    node_id_col = app_settings.INPUT_COLUMNS["node_id"]
    node_type_col = app_settings.INPUT_COLUMNS["node_type"]
    node_geom_col = app_settings.INPUT_COLUMNS["node_geometry"]

    link_id_col = app_settings.INPUT_COLUMNS["link_id"]
    link_type_col = app_settings.INPUT_COLUMNS["link_type"]
    link_geom_col = app_settings.INPUT_COLUMNS["link_geometry"]
    link_from_node_col = app_settings.INPUT_COLUMNS["link_from_node"]
    link_to_node_col = app_settings.INPUT_COLUMNS["link_to_node"]

    # --- Procesar Nodos ---
    node_cfg = current_viz_config.nodes
    if not nodes_gdf_proj.empty:
        for idx, node_row in nodes_gdf_proj.iterrows():
            geom = node_row.get(node_geom_col) # Usar .get() para seguridad
            if not isinstance(geom, Point) or geom.is_empty:
                logging.debug(f"Nodo en índice {idx} ignorado: geometría no es Point o está vacía.")
                continue

            prepared_data["nodes"]["lons"].append(geom.x)
            prepared_data["nodes"]["lats"].append(geom.y)
            
            node_type_val = node_row.get(node_type_col, "default_node_type_placeholder")
            style = node_cfg.style_mapping.get(node_type_val, node_cfg.default_style)
            
            prepared_data["nodes"]["colors"].append(style.color)
            prepared_data["nodes"]["sizes"].append(style.size)
            prepared_data["nodes"]["symbols"].append(style.symbol)
            
            node_id_val = node_row.get(node_id_col, f"nodo_{idx}")
            prepared_data["nodes"]["ids"].append(str(node_id_val))

            label_text = ""
            if node_cfg.show_labels and node_cfg.label_attribute:
                if node_cfg.label_attribute in node_row:
                    label_text = str(node_row.get(node_cfg.label_attribute, ""))
                else:
                    # No añadir texto si el atributo no existe, plot_engine lo manejará
                    logging.debug(f"Atributo de etiqueta de nodo '{node_cfg.label_attribute}' no encontrado para nodo {node_id_val}.")
                    pass # label_text sigue siendo ""
            prepared_data["nodes"]["texts"].append(label_text)
            
            tooltip_text_node = ""
            if node_cfg.show_tooltips:
                tooltip_text_node = get_node_tooltip_text(node_row, node_id_col, node_type_col, node_cfg.tooltip_attributes)
            prepared_data["nodes"]["tooltips"].append(tooltip_text_node)
    else:
        logging.warning("GeoDataFrame de nodos está vacío después de la proyección CRS.")

    # --- Procesar Enlaces ---
    link_cfg = current_viz_config.links
    # Aplicar un estilo único para todos los enlaces en esta fase
    # (el estilo del primer tipo de enlace encontrado o el default)
    # Esto es una simplificación para la Fase 2. Para estilos por link, necesitaríamos múltiples trazas.
    
    # Determinar el estilo a aplicar a la traza de enlaces
    # Por simplicidad, usaremos el default_style para todos los enlaces en esta iteración.
    # En el futuro, se podrían crear múltiples trazas si se necesitan estilos variados por enlace.
    effective_link_style = link_cfg.default_style
    
    # Si hay mapeos y queremos usar el primero como representativo (esto es una heurística)
    # if link_cfg.style_mapping:
    #     first_style_key = next(iter(link_cfg.style_mapping), None)
    #     if first_style_key:
    #         effective_link_style = link_cfg.style_mapping[first_style_key]
            
    prepared_data["links"]["colors"] = [effective_link_style.color]
    prepared_data["links"]["widths"] = [effective_link_style.width]
    prepared_data["links"]["dash_styles"] = [effective_link_style.dash] # Nota: 'dash' se aplica a nivel de traza en Plotly

    if not links_gdf_proj.empty:
        for idx, link_row in links_gdf_proj.iterrows():
            geom = link_row.get(link_geom_col) # Usar .get()
            if not isinstance(geom, LineString) or geom.is_empty:
                logging.debug(f"Enlace en índice {idx} ignorado: geometría no es LineString o está vacía.")
                continue

            link_lons, link_lats = list(geom.xy[0]), list(geom.xy[1])
            prepared_data["links"]["lons"].extend(link_lons + [None]) # None para separar paths
            prepared_data["links"]["lats"].extend(link_lats + [None])
            
            link_id_val = link_row.get(link_id_col, f"link_{idx}")
            prepared_data["links"]["ids"].append(str(link_id_val))

            tooltip_text_link = ""
            if link_cfg.show_tooltips:
                tooltip_text_link = get_link_tooltip_text(link_row, link_id_col, link_type_col, link_from_node_col, link_to_node_col, link_cfg.tooltip_attributes)
            prepared_data["links"]["tooltips"].append(tooltip_text_link)
    else:
        logging.warning("GeoDataFrame de enlaces está vacío después de la proyección CRS.")
        # Asegurar que los arrays de estilo tengan al menos un valor por defecto si no hay links
        if not prepared_data["links"]["colors"]:
            prepared_data["links"]["colors"] = [link_cfg.default_style.color]
            prepared_data["links"]["widths"] = [link_cfg.default_style.width]
            prepared_data["links"]["dash_styles"] = [link_cfg.default_style.dash]


    # Logging final de datos preparados
    logging.debug(f"Nodos preparados: {len(prepared_data['nodes']['lons'])} coords")
    logging.debug(f"Enlaces preparados: {prepared_data['links']['lons'].count(None)} segmentos")
    if not prepared_data["nodes"]["lons"]:
        logging.warning("No se prepararon datos de nodos para Plotly.")
        
    return prepared_data
