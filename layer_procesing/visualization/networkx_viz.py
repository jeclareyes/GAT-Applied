import logging
# Import matplotlib.pyplot AFTER setting the backend
import matplotlib

# --- Configuration for Matplotlib Backend ---
# Set this variable to True to save the figure instead of showing it interactively
# This uses the 'Agg' backend, which does not require a GUI.
SAVE_FIGURE_INSTEAD_OF_SHOW = True
SAVE_FILENAME = "grafo_vial.png" # Filename if saving

if SAVE_FIGURE_INSTEAD_OF_SHOW:
    try:
        matplotlib.use('Agg')
        logging.info("Using 'Agg' backend for saving figure.")
    except ImportError:
        logging.warning("Could not set matplotlib backend to 'Agg'. Using default.")
else:
    # Try setting a different interactive backend explicitly
    # You might need to install PyQt5 or Tkinter if you don't have them
    # pip install PyQt5 or pip install tk
    try:
        matplotlib.use('Qt5Agg')
        logging.info("Using 'Qt5Agg' backend.")
    except ImportError:
        try:
            matplotlib.use('TkAgg')
            logging.info("Using 'TkAgg' backend.")
        except ImportError:
            logging.warning("Could not set matplotlib backend to Qt5Agg or TkAgg. Using default.")
            pass # Fallback to default if preferred backends are not available

import matplotlib.pyplot as plt
import networkx as nx
from configs.viz_styles import NODE_SIZE, ARROW_STYLE, ARROW_SIZE, EDGE_COLOR, NODE_COLOR_DEFAULT

logger = logging.getLogger(__name__)

class NetworkXPlotter:
    """
    Dibuja un grafo con NetworkX y Matplotlib.
    """
    def __init__(self, node_size=NODE_SIZE, arrowstyle=ARROW_STYLE,
                 arrowsize=ARROW_SIZE, edge_color=EDGE_COLOR,
                 node_color=NODE_COLOR_DEFAULT):
        self.node_size = node_size
        self.arrowstyle = arrowstyle
        self.arrowsize = arrowsize
        self.edge_color = edge_color
        self.node_color = node_color
        self.logger = logger

    def plot(self, G, with_labels=False, title='Grafo Vial'):
        """
        Genera y muestra/guarda un gráfico del grafo.
        """
        if G.number_of_nodes() == 0:
            self.logger.warning('El grafo no tiene nodos para visualizar.')
            return

        pos = {}
        for n, data in G.nodes(data=True):
            x = data.get('coord_x', 0)
            y = data.get('coord_y', 0)
            pos[n] = (x, y)

        plt.figure(figsize=(12, 8))
        nx.draw(G, pos,
                with_labels=with_labels,
                node_size=self.node_size,
                arrowstyle=self.arrowstyle,
                arrowsize=self.arrowsize,
                node_color=self.node_color,
                edge_color=self.edge_color)
        plt.title(title)
        # The UserWarning about tight_layout might persist depending on the plot structure,
        # but the main error should be resolved by changing the backend.
        plt.tight_layout()

        if SAVE_FIGURE_INSTEAD_OF_SHOW:
            try:
                plt.savefig(SAVE_FILENAME)
                self.logger.info(f'Visualización NetworkX guardada en {SAVE_FILENAME}.')
            except Exception as e:
                self.logger.error(f'Error al guardar la figura: {e}')
        else:
            try:
                plt.show()
                self.logger.info('Visualización NetworkX mostrada.')
            except Exception as e:
                 self.logger.error(f'Error al mostrar la figura: {e}')
                 self.logger.error('Considera cambiar SAVE_FIGURE_INSTEAD_OF_SHOW a True para guardar en lugar de mostrar.')
