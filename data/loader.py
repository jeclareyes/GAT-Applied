#!/usr/bin/env python3
"""
loader.py

Módulo para la carga y procesamiento de datos de tráfico.
Se encarga de leer un archivo pickle con la información de nodos y enlaces,
procesarla para extraer:
  - Características de nodos.
  - Conexión de enlaces (aristas) y tipos de enlace.
  - Información adicional como flujos observados, coordenadas de nodos, demandas,
    y mapeo de aristas a nodos (in_edges_idx_tonode y out_edges_idx_tonode).

El resultado es un objeto Data de PyTorch Geometric que incluye:
  - x: tensor de características de nodos.
  - edge_index: tensor de aristas.
  - Atributos adicionales para entrenamiento y evaluación.

Se incluye manejo de rutas para buscar el archivo pickle en rutas alternas si es necesario.
"""

import os
import pickle
import torch
import numpy as np
import logging
import fiona
import pandas as pd
from torch_geometric.data import Data
import geopandas as gpd


def load_real_traffic_data(network_path, odm_path):

    # --- Se lee la network a Geopandas ---

    # Listar las capas del GeoPackage
    layers = fiona.listlayers(network_path)

    # Leer las capas y asignarlas por tipo geométrico
    gdfs = {layer: gpd.read_file(network_path, layer=layer) for layer in layers}

    # Clasificar por tipo de geometría
    network_nodes = next(gdf for gdf in gdfs.values() if gdf.geometry.geom_type.iloc[0] == 'Point')
    network_links = next(gdf for gdf in gdfs.values() if gdf.geometry.geom_type.iloc[0] in ['LineString', 'MultiLineString'])

    # --- Se lee matriz ODM

    odm = pd.read_csv(odm_path, encoding='utf-8', sep=';')
    x = 1

    # --- Comienza la estructuracion del formato

    # reindexado
    # 1. Normalizar tipos como string para evitar errores de coincidencia
    network_nodes['ID'] = network_nodes['ID'].astype(str)
    network_links['INODE'] = network_links['INODE'].astype(str)
    network_links['JNODE'] = network_links['JNODE'].astype(str)

    # 2. Crear diccionario de mapeo
    id_mapping = {old_id: new_id for new_id, old_id in enumerate(network_nodes['ID'].unique())}

    # 3. Verificar IDs no encontrados
    inodes_unmapped = set(network_links['INODE'].unique()) - set(id_mapping.keys())
    jnodes_unmapped = set(network_links['JNODE'].unique()) - set(id_mapping.keys())

    if inodes_unmapped or jnodes_unmapped:
        print("⚠️ Advertencia: Hay IDs en los enlaces que no existen en los nodos.")
        if inodes_unmapped:
            print(f"  INODE IDs no encontrados: {sorted(inodes_unmapped)}")
        if jnodes_unmapped:
            print(f"  JNODE IDs no encontrados: {sorted(jnodes_unmapped)}")

    # 4. Aplicar el mapeo
    network_nodes['new_id'] = network_nodes['ID'].map(id_mapping)
    network_links['INODE'] = network_links['INODE'].map(id_mapping)
    network_links['JNODE'] = network_links['JNODE'].map(id_mapping)


    # Asignacion de nuevo indexado

    network_nodes.set_index('ID')
    network_links.set_index(['INODE', 'JNODE'])
    node_ids = network_nodes.index
    link_ids = network_links.index

    # Tipos
    node_types_list = network_nodes['node_type']
    link_types_list = network_links['edge_type']
    node_coords_list = network_nodes[['X', 'Y']]
    node_demands_list = list(network_nodes[['salidas', 'entradas']].fillna(0).astype(int).itertuples(index=False, name=None))

    # IDs informativos
    num_nodes = len(node_ids)
    node_id_to_index = {nid: i for i, nid in enumerate(node_ids)}
    node_id_map_rev = {i: nid for nid, i in node_id_to_index.items()}
    
    # Demand
    zat_demands = network_nodes[network_nodes['node_type'] == 'TAZ'][['salidas', 'entradas']] \
                .apply(lambda row: [row['salidas'], row['entradas']], axis=1) \
                .to_dict()

    # Coordinates
    node_positions = network_nodes[['X', 'Y']] \
        .apply(lambda row: [row['X'], row['Y']], axis=1) \
        .to_dict()
    
    # Anadir flujos observados. 
    years = 2022
    # TODO implementar la posibilidad de elegir todo, latt, medel, tunga.
    observed_flows = network_links[f'Adt_samtliga_fordon_{years}'].dropna()
    # observed_flows = network_links[[f'Adt_samtliga_fordon_{years}',f'Adt_tunga_fordon_{years}']].dropna()
    observed_flows_indices = torch.tensor(observed_flows.index, dtype=torch.long)
    observed_flows_values = torch.tensor(observed_flows.values, dtype=torch.float)
    
    # --- Construir el objeto Data ---
    data = Data()
    # Atributos de nodos y enlaces
    data.x = torch.tensor(node_demands_list, dtype=torch.float)
    data.edge_index = torch.tensor(network_links[['INODE', 'JNODE']].values.T, dtype=torch.int32)
    data.pos = torch.tensor(node_coords_list.values, dtype=torch.float)
    
    data.node_types = node_types_list
    data.node_coordinates = node_positions
    data.zat_demands = zat_demands
    data.node_id_map_rev = node_id_map_rev
    data.node_id_to_index = node_id_to_index
    data.num_nodes = num_nodes
    data.link_types = link_types_list
    data.observed_flow_indices = observed_flows_indices
    data.observed_flow_values = observed_flows_values

    def verificar_tensor(nombre, tensor, debe_ser_entero=False, permitir_float_enteros=False, permitir_cero=True):
        #print(f"\n🔍 Verificando tensor '{nombre}'")

        if not isinstance(tensor, torch.Tensor):
            print("❌ No es un tensor.")
            return

        # 1. Verificar NaNs
        if tensor.is_floating_point() and torch.isnan(tensor).any():
            print("❌ Contiene NaN.")
            return

        # 2. Verificar tipo de dato
        if debe_ser_entero:
            if tensor.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64):
                print("❌ No es de tipo entero.")
                return
        elif permitir_float_enteros and tensor.is_floating_point():
            if not torch.all(torch.frac(tensor) == 0):
                print("❌ Contiene valores flotantes no enteros.")
                return

        # 3. Verificar valores negativos
        if (tensor < 0).any():
            print("❌ Contiene valores negativos.")
            return

        # 4. Verificar valores positivos
        if not permitir_cero and (tensor <= 0).any():
            print("❌ Contiene ceros o valores negativos (se requieren positivos estrictos).")
            return
        elif permitir_cero and (tensor < 0).any():
            print("❌ Contiene valores negativos (cero permitido).")
            return

        print("✅ OK - Sin NaN, tipo correcto, sin negativos, y valores positivos.")


    # Aplicar a tus tensores (ajustando si cero es aceptable o no)
    verificar_tensor("data.x", data.x, debe_ser_entero=False, permitir_float_enteros=True, permitir_cero=True)
    verificar_tensor("data.edge_index", data.edge_index, debe_ser_entero=True, permitir_cero=True)
    verificar_tensor("data.pos", data.pos, debe_ser_entero=False, permitir_float_enteros=False, permitir_cero=False)

    data = _build_edge_mappings(data)

    return data

def load_traffic_data_pickle(pickle_name: str) -> Data:
    """
    Carga datos de tráfico desde un archivo pickle y construye un objeto Data de PyTorch Geometric.

    El archivo pickle debe contener un diccionario con claves 'nodes' y 'links' que incluyen:
      - nodes: 'node.ids', 'node.type', 'node.coordinates', 'node.demand'
      - links: 'link.ids', 'link.type', 'link.ij', 'link.observed_flow'

    Se buscan el archivo en múltiples rutas si es necesario.

    Args:
        pickle_name (str): Ruta relativa del archivo pickle (p.ej., "data/traffic_data_10.pkl").

    Returns:
        Data: Objeto PyTorch Geometric con atributos:
            - x: tensor de características [num_nodes, 4] (ej. [is_zat, is_intersection, gen, attr]).
            - edge_index: tensor de aristas [2, num_edges].
            - observed_flow_indices, observed_flow_values: tensores para flujos observados.
            - node_coordinates: lista de coordenadas para visualización.
            - node_types: lista de tipos de nodo.
            - zat_demands: diccionario con demandas de nodos ZAT.
            - node_id_map_rev: mapeo índice -> ID de nodo.
            - node_id_to_index: mapeo ID -> índice.
            - link_types: lista de tipos de enlace.
            - in_edges_idx_tonode, out_edges_idx_tonode: listas con índices de aristas entrantes/salientes por nodo.
    Raises:
        FileNotFoundError: Si no se encuentra el archivo pickle.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Búsqueda del archivo pickle en rutas alternativas
    script_dir = os.path.dirname(os.path.abspath(__file__))
    pickle_route = os.path.join(script_dir, pickle_name)
    if not os.path.exists(pickle_route):
        parent_dir = os.path.dirname(script_dir)
        pickle_route_alt = os.path.join(parent_dir, pickle_name)
        logging.warning(f"Archivo no encontrado en {pickle_route}. Intentando en {pickle_route_alt}")
        if os.path.exists(pickle_route_alt):
            pickle_route = pickle_route_alt
        else:
            pickle_route_cwd = os.path.join(os.getcwd(), pickle_name)
            logging.warning(f"Archivo no encontrado en {pickle_route_alt}. Intentando en {pickle_route_cwd}")
            if os.path.exists(pickle_route_cwd):
                pickle_route = pickle_route_cwd
            else:
                logging.error("Archivo pickle no encontrado en ninguna ruta.")
                raise FileNotFoundError(f"Archivo pickle no encontrado: {pickle_name}")

    # Cargar el archivo pickle
    with open(pickle_route, 'rb') as f:
        datos = pickle.load(f)

    # --- Procesamiento de nodos ---
    node_ids = datos['nodes']['node.ids']
    node_types_list = datos['nodes']['node.type']
    node_coords_list = datos['nodes']['node.coordinates']
    node_demands_list = datos['nodes']['node.demand']

    num_nodes = len(node_ids)
    node_id_to_index = {nid: i for i, nid in enumerate(node_ids)}
    node_id_map_rev = {i: nid for nid, i in node_id_to_index.items()}

    x_list = []
    node_types = []
    zat_demands = {}
    node_positions = []

    for i, (nid, ntype, coords, demand_tuple) in enumerate(
            zip(node_ids, node_types_list, node_coords_list, node_demands_list)):
        if ntype == 'zat':
            is_zat, is_intersection = 1.0, 0.0
            gen, attr = float(demand_tuple[0]), float(demand_tuple[1])
            zat_demands[nid] = [gen, attr]
            x_list.append([is_zat, is_intersection, gen, attr])
        elif ntype == 'intersection':
            is_zat, is_intersection = 0.0, 1.0
            x_list.append([is_zat, is_intersection, 0.0, 0.0])
        else:
            logging.warning(f"Tipo desconocido '{ntype}' para el nodo {nid}. Tratándolo como intersección.")
            is_zat, is_intersection = 0.0, 1.0
            x_list.append([is_zat, is_intersection, 0.0, 0.0])
        node_types.append(ntype)
        node_positions.append(coords)

    x = torch.tensor(x_list, dtype=torch.float)
    pos = torch.tensor(node_positions, dtype=torch.float)

    # --- Procesamiento de enlaces ---
    link_ids_list = datos['links']['link.ids']
    link_type_list_orig = datos['links']['link.type']
    link_ij_list = datos['links']['link.ij']
    link_obs_flow_list = datos['links']['link.observed_flow']

    edge_list_indices = []
    valid_link_types = []
    original_indices_map = {}

    for orig_idx, (u_id, v_id) in enumerate(link_ij_list):
        if u_id in node_id_to_index and v_id in node_id_to_index:
            u_idx = node_id_to_index[u_id]
            v_idx = node_id_to_index[v_id]
            current_edge_idx = len(edge_list_indices)
            edge_list_indices.append((u_idx, v_idx))
            valid_link_types.append(link_type_list_orig[orig_idx])
            original_indices_map[current_edge_idx] = orig_idx
        else:
            logging.warning(f"Arista ({u_id} -> {v_id}) ignorada. ID de nodo no encontrado.")

    if edge_list_indices:
        edge_index = torch.tensor(edge_list_indices, dtype=torch.long).t().contiguous()
    else:
        logging.warning("No se encontraron aristas válidas. Se crea un tensor vacío.")
        edge_index = torch.empty((2, 0), dtype=torch.long)

    # --- Flujos observados ---
    num_edges = edge_index.size(1)
    observed_indices = []
    observed_values = []
    for new_idx in range(num_edges):
        orig_idx = original_indices_map[new_idx]
        obs_flow = link_obs_flow_list[orig_idx]
        if obs_flow is not None:
            observed_indices.append(new_idx)
            observed_values.append(float(obs_flow))

    if observed_indices:
        observed_flow_indices = torch.tensor(observed_indices, dtype=torch.long)
        observed_flow_values = torch.tensor(observed_values, dtype=torch.float)
    else:
        observed_flow_indices = torch.empty((0,), dtype=torch.long)
        observed_flow_values = torch.empty((0,), dtype=torch.float)

    # --- Construir el objeto Data ---
    data = Data(
        x=x,
        edge_index=edge_index,
        pos=pos
    )
    # Atributos de nodos y enlaces
    data.node_types = node_types
    data.node_coordinates = node_positions
    data.zat_demands = zat_demands
    data.node_id_map_rev = node_id_map_rev
    data.node_id_to_index = node_id_to_index
    data.num_nodes = num_nodes
    data.link_types = valid_link_types
    data.observed_flow_indices = observed_flow_indices
    data.observed_flow_values = observed_flow_values

    # --- Construir mapeo de aristas por nodo: in_edges_idx_tonode y out_edges_idx_tonode ---
    data = _build_edge_mappings(data)

    logging.info(f"Datos procesados: {num_nodes} nodos y {num_edges} aristas.")
    return data


def _build_edge_mappings(data: Data) -> Data:
    """
    Construye listas de índices de aristas entrantes y salientes para cada nodo.

    Agrega al objeto Data dos atributos:
      - in_edges_idx_tonode: lista de tensores; para cada nodo, contiene los índices de aristas donde el nodo es destino.
      - out_edges_idx_tonode: lista de tensores; para cada nodo, contiene los índices de aristas donde el nodo es fuente.

    Args:
        data (Data): Objeto PyTorch Geometric con atributos 'edge_index' y 'num_nodes'.

    Returns:
        Data: El objeto Data con los nuevos atributos agregados.
    """
    num_nodes = data.num_nodes
    num_edges = data.edge_index.size(1)
    in_edges = [[] for _ in range(num_nodes)]
    out_edges = [[] for _ in range(num_nodes)]

    edge_index = data.edge_index.cpu().numpy()
    for e in range(num_edges):
        src = int(edge_index[0, e])
        dst = int(edge_index[1, e])
        out_edges[src].append(e)
        in_edges[dst].append(e)

    # Convertir las listas a tensores
    in_edges_tensors = [torch.tensor(idx_list, dtype=torch.long) if idx_list else torch.empty((0,), dtype=torch.long)
                        for idx_list in in_edges]
    out_edges_tensors = [torch.tensor(idx_list, dtype=torch.long) if idx_list else torch.empty((0,), dtype=torch.long)
                         for idx_list in out_edges]

    data.in_edges_idx_tonode = in_edges_tensors
    data.out_edges_idx_tonode = out_edges_tensors
    return data

def add_virtual_links(data: Data) -> Data:
    """
    Crea enlaces virtuales entre todos los nodos tipo TAZ o AUX (insensible a mayúsculas).
    Se crean enlaces en ambas direcciones entre cada par de estos nodos.

    Si ya existe el atributo virtual_edge_index, se retorna el objeto Data sin cambios.

    Args:
        data (Data): Objeto PyTorch Geometric que debe incluir:
            - node_types: lista con el tipo de cada nodo (e.g., 'zat', 'AUX', etc.)

    Returns:
        Data: Objeto Data actualizado con el atributo virtual_edge_index (tensor [2, num_virtual_edges]).
    """
    if hasattr(data, 'virtual_edge_index') and data.virtual_edge_index is not None and not data.virtual_edge_index.numel() == 0:
        return data

    taz_aux_indices = [
        idx for idx, ntype in enumerate(data.node_types)
        if ntype.lower() in ['zat', 'taz', 'aux']
    ]

    virtual_edges = []
    for i in range(len(taz_aux_indices)):
        for j in range(len(taz_aux_indices)):
            if i != j:
                src = taz_aux_indices[i]
                dst = taz_aux_indices[j]
                virtual_edges.append((src, dst))

    if virtual_edges:
        data.virtual_edge_index = torch.tensor(virtual_edges, dtype=torch.long).t().contiguous()
    else:
        data.virtual_edge_index = torch.empty((2, 0), dtype=torch.long)

    return data

def add_virtual_and_logical_links(data: Data) -> Data:
    """
    Agrega enlaces virtuales al objeto Data para ser usados en modelos que requieren información
    de conectividad adicional, como HetGATPyG. Este ejemplo conecta cada nodo ZAT con la intersección
    más cercana (calculada mediante distancia euclidiana) y crea enlaces virtuales en ambos sentidos.

    Si ya existe el atributo virtual_edge_index, se retorna el objeto Data sin cambios.

    Args:
        data (Data): Objeto PyTorch Geometric que debe incluir:
            - node_types: lista con el tipo de cada nodo ('zat' o 'intersection')
            - node_coordinates: lista con las coordenadas de cada nodo.

    Returns:
        Data: Objeto Data actualizado con el atributo virtual_edge_index (tensor [2, num_virtual_edges]).
    """
    if hasattr(data, 'virtual_edge_index') and data.virtual_edge_index is not None:
        return data

    virtual_edges = []
    for idx, ntype in enumerate(data.node_types):
        if ntype == 'zat':
            zat_coord = data.node_coordinates[idx]
            min_dist = float('inf')
            closest_intersection_idx = None
            for j, other_type in enumerate(data.node_types):
                if other_type == 'intersection':
                    inter_coord = data.node_coordinates[j]
                    dist = ((zat_coord[0] - inter_coord[0]) ** 2 + (zat_coord[1] - inter_coord[1]) ** 2) ** 0.5
                    if dist < min_dist:
                        min_dist = dist
                        closest_intersection_idx = j
            if closest_intersection_idx is not None:
                virtual_edges.append((idx, closest_intersection_idx))
                virtual_edges.append((closest_intersection_idx, idx))

    if virtual_edges:
        data.virtual_edge_index = torch.tensor(virtual_edges, dtype=torch.long).t().contiguous()
    else:
        data.virtual_edge_index = torch.empty((2, 0), dtype=torch.long)
    return data

def load_traffic_data_dict():
    # Definición del diccionario con todos los datos
    data_dict = {
        'x': [
            # is_zat, is_int, gen, attr
            [15, 10],
            [10, 15],
            [0.0, 0.0]
        ],

        'edge_index': [
            [0, 2, 1, 2],
            [2, 0, 2, 1]
        ],
        'pos': [
            [74.1551, 24.4892],
            [13.9538, 10.2495],
            [74.0668, 54.5367]
        ],
        'node_types': ['zat', 'zat', 'intersection'],

        'node_coordinates': [(74.15504997598329, 24.48918538034762),
                             (13.95379285251439, 10.24951761715075),
                             (74.06677446676758, 54.53665337483498)],

        'zat_demands': {'Z0': [15, 10],
                        'Z1': [10, 15]}
        ,
        # Estas listas son no rectangulares, se dejarán como listas de Python
        # [t.tolist() for t in data.in_edges_idx_tonode]
        'in_edges_idx_tonode': [
            [1],
            [3],
            [0, 2]
        ],

        'out_edges_idx_tonode': [
            [0],
            [2],
            [1, 3]
        ],
        'virtual_edge_index': [
            [0, 1],
            [1, 0]
        ],

        #'observed_flow_indices': torch.tensor([True, True, True, True], dtype=torch.int64)
        'observed_flow_indices': torch.tensor([], dtype=torch.int64)
        # [] {ndarray: (0,)}
        ,

        'observed_flow_values': torch.tensor([], dtype=torch.int64)
        #'observed_flow_values': [153.0, 138.0, 138.0, 153.0]
        ,
        'node_id_map_rev': {0: 'Z0',
                            1: 'Z1',
                            2: 'I0'},

        'link_types': ['logical', 'logical', 'logical', 'logical']
    }

    def dict_to_data(data_dict):
        data = Data()

        for key, value in data_dict.items():
            if key == 'zat_demands':
                # Procesar como diccionario de tensores
                zat_dict = {k: torch.tensor(v, dtype=torch.float) for k, v in value.items()}
                data.zat_demands = zat_dict
            elif key in ['edge_index', 'virtual_edge_index',
                         'observed_flow_indices', 'observed_flow_values']:
                data_attr = torch.tensor(value, dtype=torch.long if 'index' in key else torch.float)
                setattr(data, key, data_attr)
            elif key in ['x', 'pos']:
                setattr(data, key, torch.tensor(value, dtype=torch.float))
            elif key in ['in_edges_idx_tonode', 'out_edges_idx_tonode']:
                # Convertir listas internas a tensores individualmente
                tensor_list = [torch.tensor(v, dtype=torch.long) for v in value]
                setattr(data, key, tensor_list)
            elif key == 'node_id_map_rev':
                # Guardar directamente como diccionario
                setattr(data, key, value)
            else:
                # Si no es tensorizable, se deja como está
                try:
                    setattr(data, key, torch.tensor(value))
                except Exception:
                    setattr(data, key, value)

        return data
    return dict_to_data(data_dict)
