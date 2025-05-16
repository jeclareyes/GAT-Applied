# interactive_visualization/config/app_settings.py

# Configuraciones generales de la aplicación Dash
class AppSettings:
    APP_TITLE = "Visualizador Interactivo de Red Vial"
    DEBUG_MODE = True  # Poner en False para producción

    # Configuraciones del mapa base de Plotly (Scattermapbox)
    DEFAULT_MAP_STYLE = "open-street-map"  # Estilo de mapa por defecto
    DEFAULT_MAP_ZOOM = 10
    DEFAULT_MAP_CENTER = {"lat": 58.41, "lon": 15.62}  # Centro aproximado (Linköping)

    # CRS esperado para las coordenadas en los GeoDataFrames de entrada
    # Si tus datos están en otro CRS, se transformarán a EPSG:4326 para Plotly
    INPUT_DATA_CRS = "EPSG:3006" # SWEREF99 TM, común en Suecia
    PLOTLY_MAP_CRS = "EPSG:4326"  # WGS84, el que usa Plotly para mapas

    # Nombres de columnas esperados en los GDF de entrada
    # Esto ayuda a que el data_manager sea más flexible
    # (Adaptado de tu settings.py original)
    INPUT_COLUMNS = {
        "node_id": "ID",  # Columna de ID original en el GDF de nodos
        "node_type": "node_type",
        "node_geometry": "geometry", # Se asumirá que es Point
        "node_x_coord": "X", # Coordenada X original (si no se deriva de la geometría)
        "node_y_coord": "Y", # Coordenada Y original (si no se deriva de la geometría)

        "link_id": "LINK_ID", # Un ID único para el link si lo tienes, sino se puede generar
        "link_from_node": "INODE",
        "link_to_node": "JNODE",
        "link_type": "edge_type",
        "link_geometry": "geometry", # Se asumirá que es LineString
    }

# Instancia para ser importada fácilmente
app_settings = AppSettings()
