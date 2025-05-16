# interactive_visualization/config/viz_config.py
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List

# --- Estilos para Plotly ---
@dataclass
class PlotlyNodeStyle:
    """Estilo para nodos en Plotly."""
    color: str = "blue"
    size: int = 10
    opacity: float = 0.8
    symbol: str = "circle"

@dataclass
class PlotlyLinkStyle:
    """Estilo para enlaces en Plotly."""
    color: str = "grey"
    width: int = 2
    opacity: float = 0.7
    dash: str = "solid"

@dataclass
class PlotlyLabelProperties:
    """Propiedades para etiquetas de texto en Plotly (usando anotaciones o modo texto)."""
    font_size: int = 10
    font_color: str = "black"
    font_family: str = "Arial, sans-serif"
    bgcolor: str = "rgba(255, 255, 255, 0.7)"
    bordercolor: str = "rgba(0, 0, 0, 0.5)"
    borderwidth: int = 1

# --- Configuraciones de Visualización ---
@dataclass
class NodeVizConfig:
    """Configuración general para la visualización de nodos."""
    style_mapping: Dict[str, PlotlyNodeStyle] = field(default_factory=dict)
    default_style: PlotlyNodeStyle = field(default_factory=PlotlyNodeStyle)
    show_labels: bool = True # Cambiado a False por defecto
    label_attribute: Optional[str] = "ID" # Atributo por defecto para la etiqueta del nodo
    label_properties: PlotlyLabelProperties = field(default_factory=PlotlyLabelProperties)
    show_tooltips: bool = True
    tooltip_attributes: List[str] = field(default_factory=lambda: ["ID", "node_type"]) # Atributos por defecto para tooltips

@dataclass
class LinkVizConfig:
    """Configuración general para la visualización de enlaces."""
    style_mapping: Dict[str, PlotlyLinkStyle] = field(default_factory=dict)
    default_style: PlotlyLinkStyle = field(default_factory=PlotlyLinkStyle)
    # Las etiquetas directas en enlaces son complejas en Scattermapbox; usaremos tooltips.
    show_tooltips: bool = True
    tooltip_attributes: List[str] = field(default_factory=lambda: ["LINK_ID", "edge_type", "INODE", "JNODE"]) # Atributos por defecto para tooltips

# --- Instancias de Configuración por Defecto (Ejemplos) ---
default_node_config = NodeVizConfig(
    style_mapping={
        "TAZ": PlotlyNodeStyle(color="orange", size=15, symbol="square"),
        "Intersection": PlotlyNodeStyle(color="purple", size=8),
        "aux": PlotlyNodeStyle(color="green", size=10, symbol="diamond"),
        "Border_Node": PlotlyNodeStyle(color="red", size=12, symbol="star"),
    },
    default_style=PlotlyNodeStyle(color="lightgrey", size=1000),
    label_attribute="ID",
    show_labels=True # Etiquetas de nodo apagadas por defecto
)

default_link_config = LinkVizConfig(
    style_mapping={
        "road": PlotlyLinkStyle(color="black", width=3),
        "taz_link": PlotlyLinkStyle(color="darkorange", width=1.5, dash="dot"),
        "aux_link": PlotlyLinkStyle(color="mediumseagreen", width=1, dash="dash"),
    },
    default_style=PlotlyLinkStyle(color="red", width=3),
    show_tooltips=True
)

@dataclass
class GlobalVizConfig:
    nodes: NodeVizConfig = field(default_factory=lambda: default_node_config)
    links: LinkVizConfig = field(default_factory=lambda: default_link_config)

global_viz_config = GlobalVizConfig()
