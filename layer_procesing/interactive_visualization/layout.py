# interactive_visualization/layout.py
from dash import dcc, html

def create_main_layout():
    """
    Crea el layout principal de la aplicación Dash.
    """
    return html.Div(id="main-container", children=[
        html.H1(
            id="app-title",
            children="Visualizador Interactivo de Red",
            style={'textAlign': 'center', 'marginBottom': '10px'}
        ),
        
        # Panel de Control
        html.Div(id="control-panel-container", children=[
            html.H3("Controles de Visualización", style={'marginTop': '0px', 'marginBottom': '10px'}),
            
            # Controles de Nodos
            html.Div([
                html.Label("Etiquetas de Nodos:", style={'fontWeight': 'bold'}),
                dcc.Checklist(
                    id='node-label-visibility-checklist',
                    options=[{'label': ' Mostrar etiquetas', 'value': 'show'}],
                    value=[], # Apagadas por defecto
                    style={'marginBottom': '5px'}
                ),
                html.Label("Atributo para etiquetas de nodos:"),
                dcc.Dropdown(
                    id='node-label-attribute-dropdown',
                    options=[], # Se llenará dinámicamente
                    value=None, # O el atributo por defecto de viz_config
                    clearable=False,
                    style={'width': '200px', 'marginBottom': '10px'}
                ),
            ], style={'border': '1px solid #ddd', 'padding': '10px', 'marginBottom': '10px', 'borderRadius': '5px'}),

            # Controles de Enlaces
            html.Div([
                html.Label("Tooltips de Enlaces:", style={'fontWeight': 'bold'}),
                dcc.Checklist(
                    id='link-tooltip-visibility-checklist',
                    options=[{'label': ' Mostrar tooltips', 'value': 'show'}],
                    value=['show'], # Encendidos por defecto
                    style={'marginBottom': '5px'}
                ),
                # Aquí podríamos añadir un Dropdown para seleccionar atributos para tooltips de enlaces
            ], style={'border': '1px solid #ddd', 'padding': '10px', 'borderRadius': '5px'}),

        ], style={
            'width': '250px', 
            'float': 'left', 
            'padding': '15px',
            'marginRight': '15px',
            'border': '1px solid #ccc',
            'borderRadius': '5px',
            'backgroundColor': '#f9f9f9'
        }),
        
        # Contenedor para el grafo
        html.Div(id="graph-container", children=[
            dcc.Graph(
                id="network-graph",
                figure={},
                style={'height': '85vh'} # Ajusta según necesidad
            )
        ], style={'marginLeft': '280px'}), # Ajustar el margen para que no se solape con el panel
        
        html.Div(id="info-display-container", style={'clear': 'both', 'padding': '20px'})
    ])
