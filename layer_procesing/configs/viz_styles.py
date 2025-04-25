# config/viz_styles.py
"""
Estilos y constantes de visualización para matplotlib y Folium.
"""

# Matplotlib
NODE_SIZE = 50
ARROW_STYLE = '->'
ARROW_SIZE = 10
EDGE_COLOR = 'gray'
NODE_COLOR_DEFAULT = 'blue'

# Colores para nodos según tipo
NODE_COLOR_INTERSECTION = 'red'
NODE_COLOR_TERMINAL = 'blue'

# Gradiente de colores para segmentos (inicio, medio, fin)
SEGMENT_COLOR_START = '#3333FF'
SEGMENT_COLOR_MIDDLE = '#33FFFF'
SEGMENT_COLOR_END = '#FF3333'

# Folium
FOLIUM_MAP_ZOOM_START = 12

def folium_style(feature):
    """
    Función de estilo para GeoJson en Folium según propiedades del feature.
    """
    props = feature.get('properties', {})
    tipo = props.get('tipo')
    if tipo == 'Intersección':
        color = NODE_COLOR_INTERSECTION
    elif tipo == 'Nodo_final':
        color = NODE_COLOR_TERMINAL
    else:
        color = SEGMENT_COLOR_START

    return {
        'color': color,
        'weight': 2,
        'fillOpacity': 0.7
    }
