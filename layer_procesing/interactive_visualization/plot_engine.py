# interactive_visualization/plot_engine.py
import plotly.graph_objects as go
import logging # Añadido

from .configs.app_settings import app_settings
from .configs.viz_config import GlobalVizConfig

def create_graph_figure(
    prepared_data: dict,
    current_viz_config: GlobalVizConfig
):
    """
    Crea una figura de Plotly Scattermapbox con nodos y enlaces.
    """
    fig = go.Figure()
    
    node_cfg = current_viz_config.nodes
    link_cfg = current_viz_config.links

    # --- Trazado de Enlaces ---
    if prepared_data["links"]["lons"] and prepared_data["links"]["lons"].count(None) > 0: # Asegurar que haya segmentos
        # Usar el primer (y único en esta fase) estilo para los links
        line_color = prepared_data["links"]["colors"][0]
        line_width = prepared_data["links"]["widths"][0]
        # El atributo 'dash' para Scattermapbox se define de forma diferente, usualmente no por segmento.
        # Plotly podría no soportar 'dash' directamente en go.scattermapbox.Line.
        # Para líneas discontinuas, a menudo se necesitan trazas separadas o se omite.
        # Por ahora, lo omitiremos de la línea directa.

        fig.add_trace(go.Scattermapbox(
            mode="lines",
            lon=prepared_data["links"]["lons"],
            lat=prepared_data["links"]["lats"],
            line=go.scattermapbox.Line(
                width=line_width,
                color=line_color
            ),
            hoverinfo="text" if link_cfg.show_tooltips and any(prepared_data["links"]["tooltips"]) else "none",
            hovertext=prepared_data["links"]["tooltips"] if link_cfg.show_tooltips else None,
            # customdata=prepared_data["links"]["ids"], # Para futuros callbacks de click en links
            name="Enlaces"
        ))
        logging.info(f"Añadida traza de enlaces. Color: {line_color}, Ancho: {line_width}")
    else:
        logging.warning("No hay datos de enlaces para dibujar o no hay segmentos (falta None).")
    
    # --- Trazado de Nodos ---
    if prepared_data["nodes"]["lons"]:
        node_mode = "markers"
        node_text_content = prepared_data["nodes"]["texts"]
        
        # Solo añadir "+text" si show_labels es True Y hay algún texto no vacío para mostrar
        if node_cfg.show_labels and any(t and t.strip() for t in node_text_content):
            node_mode += "+text"
        else: # Si no se muestran etiquetas, asegurarse de que node_text_content sea None para Plotly
            node_text_content = None 
            if node_cfg.show_labels: # Si se querían mostrar pero no había contenido
                 logging.info("Visibilidad de etiquetas de nodo activada, pero no hay contenido de texto para mostrar.")


        fig.add_trace(go.Scattermapbox(
            mode=node_mode,
            lon=prepared_data["nodes"]["lons"],
            lat=prepared_data["nodes"]["lats"],
            marker=go.scattermapbox.Marker(
                size=prepared_data["nodes"]["sizes"],
                color=prepared_data["nodes"]["colors"],
                symbol=prepared_data["nodes"]["symbols"],
                opacity=node_cfg.default_style.opacity # Usar el default global por ahora
            ),
            text=node_text_content, # Será None si no se deben mostrar etiquetas o no hay texto
            textfont=dict(
                family=node_cfg.label_properties.font_family,
                size=node_cfg.label_properties.font_size,
                color=node_cfg.label_properties.font_color
            ),
            textposition="top right",
            hoverinfo="text" if node_cfg.show_tooltips and any(prepared_data["nodes"]["tooltips"]) else "none",
            hovertext=prepared_data["nodes"]["tooltips"] if node_cfg.show_tooltips else None,
            customdata=prepared_data["nodes"]["ids"],
            name="Nodos"
        ))
        logging.info(f"Añadida traza de nodos. Modo: {node_mode}, {len(prepared_data['nodes']['lons'])} nodos.")
    else:
        logging.warning("No hay datos de nodos para dibujar.")

    fig.update_layout(
        title_text=app_settings.APP_TITLE, # Usar title_text para el título
        title_x=0.5, # Centrar el título
        showlegend=False,
        mapbox_style=app_settings.DEFAULT_MAP_STYLE,
        mapbox_zoom=app_settings.DEFAULT_MAP_ZOOM,
        mapbox_center=app_settings.DEFAULT_MAP_CENTER,
        margin={"r":5,"t":45,"l":5,"b":5}, # Ajustar márgenes
        hovermode='closest'
    )
    return fig
