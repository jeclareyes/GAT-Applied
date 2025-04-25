# graph/analyzer.py
import logging
import networkx as nx

logger = logging.getLogger(__name__)

class GraphAnalyzer:
    """
    Analiza topología de un grafo dirigido (conexiones, grados, ciclos).
    """
    def __init__(self):
        self.logger = logger

    def analyze(self, G):
        report = {}
        # Conectividad débil
        wcc = list(nx.weakly_connected_components(G))
        report['n_componentes'] = len(wcc)
        report['tamaños_componentes'] = [len(c) for c in wcc]

        # Nodos aislados
        aislados = [n for n in G.nodes() if G.in_degree(n) == 0 and G.out_degree(n) == 0]
        report['n_aislados'] = len(aislados)

        # Grados
        in_deg = dict(G.in_degree())
        out_deg = dict(G.out_degree())
        report['grado_max_entrada'] = max(in_deg.values()) if in_deg else 0
        report['grado_max_salida'] = max(out_deg.values()) if out_deg else 0

        # Ciclos
        try:
            ciclos = list(nx.simple_cycles(G))
            report['n_ciclos'] = len(ciclos)
        except Exception as e:
            self.logger.error(f"Error detectando ciclos: {e}")
            report['n_ciclos'] = None

        self.logger.info(f"Análisis completado: {report}")
        return report