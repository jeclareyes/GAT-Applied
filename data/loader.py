# data/loader.py
import pickle
import torch
from torch_geometric.data import Data
import os
import logging

# Basic logging setup if not already configured elsewhere
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def load_traffic_data_custom(pickle_name):
    """
    Carga los datos de tráfico desde la estructura dict 'datos' (tal como se muestra en la pregunta)
    y construye un objeto Data de PyTorch Geometric con información de nodos y aristas.

    Retorna:
        data (torch_geometric.data.Data): objeto con atributos:
            - x: tensor de características de cada nodo [is_zat, is_intersection, gen, attr]
            - edge_index: topología (bidireccional o no, según se configure)
            # --- ATRIBUTOS DE FLUJO OBSERVADO ---
            - observed_flow_mask: tensor booleano indicando qué aristas tienen flujo observado
            - observed_flow_indices: índices (columnas) dentro de edge_index con flujo observado
            - observed_flow_values: valores numéricos de flujo observado
            # --- INFO DE CONECTIVIDAD POR NODO ---
            - in_edges_idx_to_node: listas de tensores con índices de aristas entrantes por nodo
            - out_edges_idx_to_node: listas de tensores con índices de aristas salientes por nodo
            # --- INFO ADICIONAL ---
            - node_coordinates: coordenadas de cada nodo para visualización (antes 'pos')
            - node_types: lista que conserva el tipo de cada nodo ("zat" o "intersection")
            - zat_demands: dict (opcional) que asocia cada nodo (Zx) con [generación, atracción]
            - node_id_map_rev: mapa inverso de índice->nombre de nodo, útil para debugging
            - num_nodes: número total de nodos
            - link_types: lista con el tipo ('road' o 'logical') de cada arista en edge_index <--- NUEVO
            - config: diccionario de configuración original del generador
            - node_id_to_index: mapa de nombre de nodo -> índice
    """

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Usando dispositivo: {device}")

    # -----------------------------
    # 0) Leyendo archivo Pickle
    # -----------------------------
    # Determina la ruta absoluta al directorio del script actual
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Construye la ruta al archivo pickle relativa al script
    pickle_route = os.path.join(script_dir, pickle_name)
    # Alternativamente, si pickle_name es relativo a la raíz del proyecto:
    # project_root = os.path.dirname(script_dir) # Sube un nivel desde 'data'
    # pickle_route = os.path.join(project_root, pickle_name) # Asume que pickle_name es como 'data/traffic_data_10.pkl'
    # O simplemente usa la ruta tal cual si se pasa como relativa a donde se ejecuta main.py
    # pickle_route = pickle_name # Si main.py se ejecuta desde la raíz y pickle_name es 'data/...'

    logging.info(f"Intentando cargar pickle desde: {os.path.abspath(pickle_route)}")

    if not os.path.exists(pickle_route):
        # Intenta buscar en el directorio padre si no se encuentra en data/
        parent_dir = os.path.dirname(script_dir)
        pickle_route_alt = os.path.join(parent_dir, pickle_name)
        logging.warning(f"Archivo no encontrado en {pickle_route}. Intentando en {pickle_route_alt}")
        if os.path.exists(pickle_route_alt):
            pickle_route = pickle_route_alt
        else:
            # Intenta buscar en el directorio actual si no se encuentra
            pickle_route_cwd = os.path.join(os.getcwd(), pickle_name)
            logging.warning(f"Archivo no encontrado en {pickle_route_alt}. Intentando en {pickle_route_cwd}")
            if os.path.exists(pickle_route_cwd):
                pickle_route = pickle_route_cwd
            else:
                logging.error(
                    f"Archivo pickle no encontrado en ninguna de las rutas intentadas: {pickle_route}, {pickle_route_alt}, {pickle_route_cwd}")
                raise FileNotFoundError(f"Archivo pickle no encontrado: {pickle_name}")

    with open(pickle_route, 'rb') as f:
        datos = pickle.load(f)

    # -----------------------------
    # 1) Extraemos info de nodos
    # -----------------------------
    node_ids = datos['nodes']['node.ids']  # p.ej. ['Z0', 'Z1', 'I0', 'I1', ...]
    node_types_list = datos['nodes']['node.type']  # p.ej. ['zat','zat','intersection','intersection',...]
    node_coords_list = datos['nodes']['node.coordinates']  # p.ej. [(x1,y1), (x2,y2), ...]
    node_demands_list = datos['nodes']['node.demand']  # p.ej. [(7,9), (9,7), (0,0), ...]

    num_nodes = len(node_ids)
    node_id_to_index = {nid: i for i, nid in enumerate(node_ids)}
    node_id_map_rev = {i: nid for nid, i in node_id_to_index.items()}  # Mapa inverso

    # Para cada nodo, creamos un vector de 4 features, al estilo [is_zat, is_int, gen, attr]
    # Si el nodo es un ZAT, usaremos su (gen, attr) de node_demands_list;
    # si es intersection, lo dejamos en 0,0.
    x_list = []
    node_types = []  # para guardar directamente 'zat' o 'intersection'
    zat_demands = {}  # para recuperar la tupla de (gen,attr) solo en nodos ZAT
    node_positions = []

    for i, (nid, ntype, coords, demand_tuple) in enumerate(zip(node_ids,
                                                               node_types_list,
                                                               node_coords_list,
                                                               node_demands_list)):
        # is_zat = 1 si ntype=='zat', si no 0
        # is_int = 1 si ntype=='intersection', si no 0
        if ntype == 'zat':
            is_zat = 1.0
            is_int = 0.0
            gen, attr = demand_tuple
            # Almacenar demanda neta para el nodo ZAT (gen - attr) si es necesario
            # o simplemente [gen, attr]
            zat_demands[nid] = [float(gen), float(attr)]  # Guardar como float
            # Características: [es_zat, es_interseccion, gen, attr]
            x_list.append([is_zat, is_int, float(gen), float(attr)])
        elif ntype == 'intersection':
            is_zat = 0.0
            is_int = 1.0
            gen, attr = 0.0, 0.0
            # Características: [es_zat, es_interseccion, 0, 0]
            x_list.append([is_zat, is_int, gen, attr])
        else:
            logging.warning(f"Tipo de nodo desconocido '{ntype}' para nodo {nid}. Tratando como intersección.")
            is_zat = 0.0
            is_int = 1.0
            gen, attr = 0.0, 0.0
            x_list.append([is_zat, is_int, gen, attr])

        node_types.append(ntype)
        node_positions.append(coords)

    # Convertimos a tensores
    x = torch.tensor(x_list, dtype=torch.float)
    pos = torch.tensor(node_positions, dtype=torch.float)

    # --------------------------------
    # 2) Extraemos info de las aristas
    # --------------------------------
    link_ids_list = datos['links']['link.ids']  # p.ej. ['E0','E1','E2',...]
    link_type_list_orig = datos['links']['link.type']  # p.ej. ['road', 'road', 'logical', ...] <--- IMPORTANTE
    link_ij_list = datos['links']['link.ij']  # p.ej. [('I0','I2'), ('I3','I2'), ...]
    link_obs_flow_list = datos['links']['link.observed_flow']  # p.ej. [None, None, 6, 4, ...]

    # link_assigned_flow = datos['links']['link.assigned_flows'] # No usado directamente por GAT

    edge_list_indices = []
    valid_link_types = []  # Almacenará los tipos solo de los enlaces válidos
    original_indices_map = {}  # Mapea el índice en edge_list_indices al índice original en datos['links']

    for orig_idx, (u_id, v_id) in enumerate(link_ij_list):
        if u_id in node_id_to_index and v_id in node_id_to_index:
            u_idx = node_id_to_index[u_id]
            v_idx = node_id_to_index[v_id]
            current_edge_idx = len(edge_list_indices)  # Índice que tendrá esta arista en la lista final
            edge_list_indices.append((u_idx, v_idx))
            valid_link_types.append(link_type_list_orig[orig_idx])  # Guarda el tipo del enlace válido
            original_indices_map[current_edge_idx] = orig_idx  # Mapea nuevo índice a índice original
        else:
            logging.warning(f"Arista ({u_id}->{v_id}) ignorada. ID de nodo no encontrado.")

    if not edge_list_indices:
        logging.warning("No se encontraron aristas válidas. Creando tensores vacíos.")
        edge_index = torch.empty((2, 0), dtype=torch.long)
        num_edges = 0
        link_types = []  # Lista vacía de tipos
    else:
        edge_index = torch.tensor(edge_list_indices, dtype=torch.long).t().contiguous()
        num_edges = edge_index.size(1)
        link_types = valid_link_types  # Usa la lista de tipos válidos

    # ---------------------------------------------------------
    # 3) Identificamos qué aristas tienen flujos observados
    # ---------------------------------------------------------
    observed_mask_list = [False] * num_edges
    observed_values_dict = {}  # Mapea índice de edge_index a valor observado

    # Iteramos sobre los índices de las aristas válidas (0 a num_edges-1)
    for edge_idx in range(num_edges):
        # Obtenemos el índice original correspondiente en la lista de datos['links']
        orig_link_idx = original_indices_map[edge_idx]
        # Verificamos si había un flujo observado en la lista original
        flow_val = link_obs_flow_list[orig_link_idx]
        if flow_val is not None:
            observed_mask_list[edge_idx] = True
            observed_values_dict[edge_idx] = float(flow_val)

    observed_flow_mask = torch.tensor(observed_mask_list, dtype=torch.bool)  # [num_edges]

    if num_edges > 0 and observed_values_dict:
        # Índices dentro de edge_index que tienen observación
        observed_flow_indices = torch.arange(num_edges, dtype=torch.long)[observed_flow_mask]
        # Valores observados correspondientes a esos índices
        observed_flow_values = torch.tensor([observed_values_dict[idx] for idx in observed_flow_indices.tolist()],
                                            dtype=torch.float)
    else:
        observed_flow_indices = torch.tensor([], dtype=torch.long)
        observed_flow_values = torch.tensor([], dtype=torch.float)

    # ---------------------------------------------------
    # 4) Construimos in_edges_idx y out_edges_idx por nodo
    # ---------------------------------------------------
    in_edges = [[] for _ in range(num_nodes)]
    out_edges = [[] for _ in range(num_nodes)]

    # Iteramos sobre las aristas válidas (índices 0 a num_edges-1)
    for e_col_idx, (u_idx, v_idx) in enumerate(edge_list_indices):
        out_edges[u_idx].append(e_col_idx)
        in_edges[v_idx].append(e_col_idx)

    in_edges_idx_to_node = [torch.tensor(indices, dtype=torch.long) for indices in in_edges]
    out_edges_idx_to_node = [torch.tensor(indices, dtype=torch.long) for indices in out_edges]

    # -------------------------------------
    # 5) Construimos el objeto Data de PyG
    # -------------------------------------
    data = Data(
        x=x,  # [num_nodes, 4] = [is_zat, is_int, gen, attr]
        edge_index=edge_index,  # [2, num_aristas]: Topología de la red
        # Flujo Observado (disperso)
        observed_flow_mask=observed_flow_mask,  # [num_aristas]: Máscara booleana
        observed_flow_indices=observed_flow_indices,  # [num_observed]: Índices de aristas observadas
        observed_flow_values=observed_flow_values,  # [num_observed]: Valores de flujo observados
        # Conectividad por Nodo
        in_edges_idx_tonode=in_edges_idx_to_node,  # Lista [num_nodes] de tensores de índices entrantes
        out_edges_idx_tonode=out_edges_idx_to_node,  # Lista [num_nodes] de tensores de índices salientes
        # Info Adicional de Nodos
        node_coordinates=pos,  # [num_nodes, 2]: Coordenadas (x, y)
        node_types=node_types,  # Lista [num_nodes] de strings ('zat', 'intersection')
        zat_demands=zat_demands,  # Dict {'Z0': [gen, attr], ...}
        num_nodes=num_nodes,
        # Info Adicional de Enlaces
        link_types=link_types,  # Lista [num_aristas] de strings ('road', 'logical') <-- AÑADIDO
        # Mapeos y Configuración Original
        node_id_map_rev=node_id_map_rev,  # Dict {index: node_id}
        node_id_to_index=node_id_to_index,  # Dict {node_id: index}
        config=datos.get('config', {})  # Configuración del generador original
    )

    # Podrías añadir opcionalmente los IDs de los enlaces si son necesarios más adelante
    # data.link_ids = [link_ids_list[original_indices_map[i]] for i in range(num_edges)]

    logging.info(
        f"Datos cargados. Nodos: {num_nodes}, Aristas: {num_edges}, Aristas Observadas: {observed_flow_indices.numel()}")
    logging.info(f"Ejemplo node_types: {data.node_types[:5]}")
    logging.info(f"Ejemplo link_types: {data.link_types[:5]}")  # Log para verificar
    logging.info(f"Ejemplo zat_demands: {list(data.zat_demands.items())[:3]}")

    return data.to(device)


# Bloque if __name__ == "__main__": para pruebas (no modificado, asume estructura anterior del dummy)
# Debería actualizarse si se quiere probar exhaustivamente el nuevo loader
if __name__ == "__main__":
    # Nombre del archivo pickle (ajusta según tu estructura)
    # Asume que se ejecuta desde la raíz del proyecto y el pickle está en 'data/'
    pickle_filename = f'data/traffic_data_{10}.pkl'
    # O si se ejecuta desde dentro de 'data/':
    # pickle_filename = f'traffic_data_{10}.pkl'

    # --- Creación de Dummy Pickle File (Actualizado a la nueva estructura) ---
    dummy_pickle_file = f'dummy_traffic_data_{10}.pkl'
    if not os.path.exists(dummy_pickle_file):
        print(f"Creating dummy pickle file: {dummy_pickle_file}")
        dummy_data_dict = {
            'nodes': {
                'node.ids': ['Z0', 'Z1', 'I0', 'I1'],
                'node.type': ['zat', 'zat', 'intersection', 'intersection'],
                'node.coordinates': [(10, 10), (90, 90), (50, 10), (50, 90)],
                'node.demand': [(10, 5), (5, 10), (0, 0), (0, 0)]  # (gen, attr)
            },
            'links': {
                'link.ids': ['E0', 'E1', 'E2', 'E3', 'E4', 'E5', 'E6'],
                'link.type': ['logical', 'logical', 'road', 'road', 'logical', 'logical', 'road'],
                'link.ij': [('Z0', 'I0'), ('I0', 'Z0'), ('I0', 'I1'), ('I1', 'I0'), ('Z1', 'I1'), ('I1', 'Z1'),
                            ('I0', 'Z1')],  # Añadido I0->Z1 como road
                'link.observed_flow': [None, None, 6.0, 4.0, None, None, 2.0],  # Observados en E2, E3, E6
                'link.assigned_flows': [10.0, 5.0, 6.0, 4.0, 5.0, 10.0, 2.0]  # Flujos asignados por LP (ejemplo)
            },
            'zat_od_matrix': [('Z0', 'Z1', 10), ('Z1', 'Z0', 5)],  # Ejemplo
            'config': {'NUM_ZATS': 2, 'NUM_INTERSECTIONS': 2},  # Ejemplo config
            'assignment_results': None  # Placeholder para el DataFrame
        }
        with open(dummy_pickle_file, 'wb') as f:
            pickle.dump(dummy_data_dict, f)
        pickle_filename_to_load = dummy_pickle_file  # Usa el dummy para probar
    else:
        pickle_filename_to_load = pickle_filename  # Usa el real si existe
        print(f"Dummy file {dummy_pickle_file} already exists. Attempting to load: {pickle_filename_to_load}")

    try:
        # Carga los datos usando la función actualizada
        print(f"Attempting to load: {os.path.abspath(pickle_filename_to_load)}")
        pyg_data = load_traffic_data_custom(pickle_filename_to_load)

        # Imprime información del objeto Data cargado
        print("\n--- Información del Objeto Data Cargado ---")
        print(pyg_data)
        print(f"\nClaves disponibles: {pyg_data.keys}")
        print(f"Número de nodos: {pyg_data.num_nodes}")
        print(f"Shape de características de nodo (x): {pyg_data.x.shape}")
        print(f"Shape de coordenadas de nodo: {pyg_data.node_coordinates.shape}")
        print(f"Shape de edge_index: {pyg_data.edge_index.shape}")
        print(f"Número de aristas: {pyg_data.num_edges}")
        print(f"Tipos de nodo (muestra): {pyg_data.node_types[:10]}")
        # Asegúrate de que link_types existe antes de imprimir
        if hasattr(pyg_data, 'link_types'):
            print(f"Tipos de enlace (muestra): {pyg_data.link_types[:10]}")
        else:
            print("Tipos de enlace (link_types) no encontrados en el objeto Data.")
        print(f"Demandas ZAT (neto): {pyg_data.zat_demands}")
        print(f"Máscara de flujo observado (suma): {pyg_data.observed_flow_mask.sum()}")
        print(f"Índices de flujo observado: {pyg_data.observed_flow_indices}")
        print(f"Valores de flujo observado: {pyg_data.observed_flow_values}")
        print(f"Mapa inverso de ID de nodo (ejemplo): {list(pyg_data.node_id_map_rev.items())[:5]}")
        print(f"Dispositivo de los datos: {pyg_data.x.device}")
        print(f"Configuración original: {pyg_data.config}")

    except FileNotFoundError as e:
        print(f"\nError: Archivo no encontrado.")
        print(e)
        print(f"Directorio de trabajo actual: {os.getcwd()}")
    except Exception as e:
        print(f"\nOcurrió un error inesperado durante la carga o impresión:")
        import traceback

        print(traceback.format_exc())
