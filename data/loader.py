#!/usr/bin/env python3
"""
loader.py

Módulo para la carga y procesamiento de datos de tráfico.
"""
from dataclasses import dataclass, field
import warnings
import os
import pickle
import torch
import numpy as np
import logging
import fiona
import pandas as pd
import networkx as nx
from torch_geometric.data import Data
from itertools import islice
import geopandas as gpd
import matplotlib.pyplot as plt
from config import Various, Vdf, Odm_params, TrainingConfig, Directories # Importar Various para los tipos de nodos

# Configuración de logging (asumiendo que setup_logger se llama en el script principal)
logger = logging.getLogger(__name__)

def load_real_traffic_data(network_path, odm_path, config_various: Various, initial_aux_ga_value: float = 0.0):
    """
    Carga datos de la red real y la matriz ODM, preparando el objeto Data para PyG.
    Modificado para manejar nodos AUX con G/A aprendibles.

    Args:
        network_path (str): Ruta al GeoPackage de la red.
        odm_path (str): Ruta al CSV de la matriz ODM.
        config_various (Various): Configuración con los tipos de nodos.
        initial_aux_ga_value (float): Valor inicial para G/A de nodos AUX en data.x.
                                     Estos serán sobrescritos por parámetros aprendibles en el modelo.

    Returns:
        Data: Objeto PyTorch Geometric.
    """
    logger.info(f"Cargando red desde: {network_path}")
    logger.info(f"Cargando ODM desde: {odm_path}")
    eps_val = 1e-8 # Para comparaciones de flotantes y evitar divisiones por cero

    try:
        layers = fiona.listlayers(network_path)
        gdfs = {layer: gpd.read_file(network_path, layer=layer) for layer in layers}
        
        network_nodes_gdf = next(gdf for gdf in gdfs.values() if gdf.geometry.geom_type.iloc[0] == 'Point')
        network_links_gdf = next(gdf for gdf in gdfs.values() if gdf.geometry.geom_type.iloc[0] in ['LineString', 'MultiLineString'])
        
        # Asegurar que las columnas de G/A existan, si no, crearlas con 0
        if 'salidas' not in network_nodes_gdf.columns:
            network_nodes_gdf['salidas'] = 0
        if 'entradas' not in network_nodes_gdf.columns:
            network_nodes_gdf['entradas'] = 0
            
        network_nodes_gdf['salidas'] = pd.to_numeric(network_nodes_gdf['salidas'], errors='coerce').fillna(0.0)
        network_nodes_gdf['entradas'] = pd.to_numeric(network_nodes_gdf['entradas'], errors='coerce').fillna(0.0)


        odm_df = pd.read_csv(odm_path, encoding='utf-8', sep=';')
        logger.info("Archivos de red y ODM cargados.")
    except Exception as e:
        logger.error(f"Error cargando archivos de datos: {e}", exc_info=True)
        raise

    # --- Procesamiento de Nodos ---
    # Asegurar que 'ID' y 'node_type' existan
    if 'ID' not in network_nodes_gdf.columns:
        logger.error("La capa de nodos no tiene la columna 'ID'.")
        raise ValueError("Columna 'ID' faltante en nodos.")
    if 'node_type' not in network_nodes_gdf.columns:
        logger.warning("La capa de nodos no tiene 'node_type'. Se intentará inferir o asignar por defecto.")
        # Aquí podrías añadir lógica para asignar un tipo por defecto si es necesario
        network_nodes_gdf['node_type'] = 'unknown'

    if 'NAMN' not in network_nodes_gdf.columns:
        logger.error('La capa de nodos no tiene la columna NAMN, necesaria para mapear TAZs desde el archivo ODM')
        raise ValueError('Columna NAMN faltante en nodos')


    network_nodes_gdf['original_id_str'] = network_nodes_gdf['ID'].astype(str)
    unique_original_ids = network_nodes_gdf['original_id_str'].unique()
    node_id_to_index = {original_id: i for i, original_id in enumerate(unique_original_ids)}
    node_id_map_rev = {i: original_id for original_id, i in node_id_to_index.items()}
    
    network_nodes_gdf['idx'] = network_nodes_gdf['original_id_str'].map(node_id_to_index)
    network_nodes_gdf = network_nodes_gdf.sort_values(by='idx').set_index('idx', drop=False)

    num_nodes = len(unique_original_ids)
    logger.info(f"Número total de nodos únicos: {num_nodes}")

    # Características de nodo (x)
    # Columnas: [es_taz, es_aux, es_interseccion, gen_taz, attr_taz, gen_aux_placeholder, attr_aux_placeholder]
    # Los placeholders para gen_aux y attr_aux se inicializan aquí pero serán manejados por parámetros aprendibles.
    # La idea es que el modelo use las columnas 3 y 4 para TAZ, y 5 y 6 para AUX.
    node_features_list = []
    aux_node_indices_list = [] # Para identificar nodos AUX en el modelo

    for i in range(num_nodes):
        node_data = network_nodes_gdf.loc[i]
        node_type_lower = str(node_data['node_type']).lower() # Convertir a minúsculas para comparación

        is_taz = 1.0 if node_type_lower in [t.lower() for t in config_various.taz_node_types] else 0.0
        is_aux = 1.0 if node_type_lower in [a.lower() for a in config_various.aux_node_types] else 0.0
        is_intersection = 1.0 if node_type_lower in [i_type.lower() for i_type in config_various.intersection_node_types] else 0.0
        
        # Si no es ninguno de los anteriores, por defecto es intersección (o manejar error)
        if not (is_taz or is_aux or is_intersection):
            # logger.warning(f"Nodo {node_data['original_id_str']} (índice {i}) con tipo '{node_data['node_type']}' no reconocido, asignado como intersección.")
            is_intersection = 1.0

        gen_taz, attr_taz = 0.0, 0.0
        gen_aux_placeholder, attr_aux_placeholder = 0.0, 0.0

        if is_taz:
            gen_taz = float(node_data.get('salidas', 0.0))
            attr_taz = float(node_data.get('entradas', 0.0))
        elif is_aux:
            aux_node_indices_list.append(i) # Guardar índice global del nodo AUX
            # Estos valores son placeholders; los reales serán aprendibles
            gen_aux_placeholder = initial_aux_ga_value 
            attr_aux_placeholder = initial_aux_ga_value

        node_features_list.append([
            is_taz, is_aux, is_intersection, 
            gen_taz, attr_taz,
            gen_aux_placeholder, attr_aux_placeholder 
        ])

    data = Data()
    data.x = torch.tensor(node_features_list, dtype=torch.float)
    data.aux_node_indices = torch.tensor(aux_node_indices_list, dtype=torch.long) if aux_node_indices_list else torch.empty((0,), dtype=torch.long)
    
    # Mapeo de ID original de nodo AUX a su índice *dentro de la lista de nodos AUX*
    # Esto será útil en el modelo para asignar los parámetros aprendibles.
    data.aux_node_original_id_to_aux_idx = {
    node_id_map_rev[global_idx]: aux_list_idx 
    for aux_list_idx, global_idx in enumerate(aux_node_indices_list)
    }


    logger.info(f"Características de nodos (data.x) creadas. Shape: {data.x.shape}")
    logger.info(f"Índices de nodos AUX (data.aux_node_indices): {data.aux_node_indices.tolist()}")


    # Coordenadas y otros atributos de nodo
    data.node_types = network_nodes_gdf['node_type'].tolist()
    data.node_coordinates = network_nodes_gdf[['X', 'Y']].values.tolist()
    data.node_id_map_rev = node_id_map_rev
    data.node_id_to_index = node_id_to_index # Mapeo de ID original a índice global
    data.num_nodes = num_nodes
    data.pos = torch.tensor(network_nodes_gdf[['X', 'Y']].values, dtype=torch.float)

    # Demandas TAZ (para la función de pérdida, usando el ID original)
    data.zat_demands = {
        row['original_id_str']: [float(row.get('salidas', 0.0)), float(row.get('entradas', 0.0))]
        for _, row in network_nodes_gdf.iterrows() if str(row['node_type']).lower() in [t.lower() for t in config_various.taz_node_types]
    }
    
    # --- Procesamiento de Enlaces ---
    network_links_gdf['INODE_idx'] = network_links_gdf['INODE'].astype(str).map(node_id_to_index)
    network_links_gdf['JNODE_idx'] = network_links_gdf['JNODE'].astype(str).map(node_id_to_index)
    
    # Filtrar enlaces donde algún nodo no fue mapeado (NaN en _idx)
    valid_links_mask = network_links_gdf['INODE_idx'].notna() & network_links_gdf['JNODE_idx'].notna()
    if not valid_links_mask.all():
        logger.warning(f"Se descartaron {len(network_links_gdf) - valid_links_mask.sum()} enlaces debido a nodos no encontrados.")
    network_links_gdf = network_links_gdf[valid_links_mask]

    if network_links_gdf.empty:
        logger.error("No hay enlaces válidos después del mapeo de nodos.")
        data.edge_index = torch.empty((2,0), dtype=torch.long)
        data.link_types = []
        data.observed_flow_indices = torch.empty((0,), dtype=torch.long)
        data.observed_flow_values = torch.empty((0,), dtype=torch.float)
        data.vdf_tensor = torch.empty((0,5), dtype=torch.float)
        data.od_pairs = []
    else:
        data.edge_index = torch.tensor(network_links_gdf[['INODE_idx', 'JNODE_idx']].values.T, dtype=torch.long)
        data.link_types = network_links_gdf['edge_type'].tolist()

        # Parametros para VDF
        vp = VDFPreprocessor(network_links_gdf)
        vp.group_curves(by=Vdf.grouped_by)
        vp.set_manual_params(Vdf.vdf_dictionary[Vdf.grouped_by])
        vp.set_capacity_by_group(Vdf.capacity)
        # Se entrega tensor con [capacity, FFS, lanes, length, alpha, beta]
        vdf_params_df = vp.get_vdf_tensor()
        data.vdf_tensor = torch.tensor(vdf_params_df.values, dtype=torch.float)

        # Flujos Observados (ejemplo con año 2022, adaptar según necesidad)
        year_to_observe = 2022 
        observed_col_name = f'Adt_samtliga_fordon_{year_to_observe}'
        if observed_col_name in network_links_gdf.columns:
            observed_flows_series = network_links_gdf[observed_col_name].dropna()
            data.observed_flow_indices = torch.tensor(observed_flows_series.index.values, dtype=torch.long)
            data.observed_flow_values = torch.tensor(observed_flows_series.values, dtype=torch.float)
            
            logger.info(f"Cargados {len(data.observed_flow_indices)} flujos observados de la columna '{observed_col_name}'.")
        else:
            logger.warning(f"Columna de flujo observado '{observed_col_name}' no encontrada. No se cargarán flujos observados.")
            data.observed_flow_indices = torch.empty((0,), dtype=torch.long)
            data.observed_flow_values = torch.empty((0,), dtype=torch.float)

        # --- Procesamiento de Matriz O-D ---
        # Asumimos que odm_df tiene columnas 'origin_zone', 'destination_zone', 'people'
        # y que 'origin_zone', 'destination_zone' son NOMBRES originales de nodos TAZ.

        # 1. Identificar nodos TAZ en network_nodes_gdf
        Network_nodes_ZAT = network_nodes_gdf[
            network_nodes_gdf['node_type'].astype(str).str.lower().isin([t.lower() for t in config_various.taz_node_types])
        ].copy() # Usar .copy() para evitar SettingWithCopyWarning

        # Inicializar estructuras para K-rutas fuera del bloque principal del builder
        data.k_shortest_paths_by_length_link_indices = {}
        # Este mapeo es necesario para convertir las rutas de nodos a rutas de índices de enlace
        node_pair_to_edge_idx = { (u,v): i for i, (u,v) in enumerate(data.edge_index.t().cpu().tolist()) }

        if Network_nodes_ZAT.empty or odm_df.empty: # Añadir la verificación de odm_df aquí también
            if Network_nodes_ZAT.empty:
                logger.error("No se encontraron nodos de tipo TAZ.")
            if odm_df.empty:
                logger.warning("DataFrame ODM está vacío. No se cargarán pares O-D ni k-rutas.")

            # Inicializar estructuras de datos como vacías si no hay TAZs o ODM está vacío
            data.od_tensor = np.empty((0, 0), dtype=float)
            # Asumimos que build_od_list_df retornaría un DataFrame vacío si odm_df está vacío
            data.od_pairs = pd.DataFrame(columns=['origin_node_idx', 'destination_node_idx', 'demand'])
            data.taz_contiguous_idx_to_original_id = {}
            data.taz_original_id_to_contiguous_idx = {}
            data.taz_name_to_original_id = {}
            data.taz_original_id_to_universal_idx = {}
            # Las estructuras de K-rutas ya se inicializaron como vacías arriba

        else: # Si hay TAZs y el DataFrame ODM no está vacío
            # --- Lógica del OdTensorBuilder (basada en tu segundo fragmento) ---
            # 2. Crear el mapeo de ID original de TAZ a índice contiguo
            taz_original_ids = Network_nodes_ZAT['original_id_str'].tolist()
            num_taz = len(taz_original_ids)
            taz_original_id_to_contiguous_idx = {id: i for i, id in enumerate(taz_original_ids)}
            contiguous_idx_to_taz_original_id = {i: id for id, i in taz_original_id_to_contiguous_idx.items()}

            # 3. Crear el mapeo de NOMBRE de TAZ a ID original de TAZ
            taz_name_to_original_id = {} # Inicializar por defecto
            if 'NAMN' in Network_nodes_ZAT.columns and 'original_id_str' in Network_nodes_ZAT.columns:
                    taz_name_to_original_id = dict(zip(Network_nodes_ZAT['NAMN'].astype(str), Network_nodes_ZAT['original_id_str'].astype(str)))
                    logger.info(f"Creado mapeo de nombre TAZ a ID original para {len(taz_name_to_original_id)} TAZs.")
            else:
                    logger.warning("Columnas 'NAMN' o 'original_id_str' faltantes en Network_nodes_ZAT para crear el mapeo de nombre a ID.")


            # 4. Crear el mapeo de ID original de TAZ su indice UNIVERSAL
            taz_original_id_to_universal_idx = {
                original_id: node_id_to_index[original_id]
                for original_id in taz_original_ids if original_id in node_id_to_index
            }
            if len(taz_original_id_to_universal_idx) != num_taz:
                logger.warning('No todos los IDs originales de TAZs se encontraron en el mapeo universal de nodos')

            logger.info(f"Identificadas {num_taz} TAZs. Creando mapeo a índices contiguos.")

            # 5. Construir el tensor O-D y la lista de pares O-D usando el builder
            builder = OdTensorBuilder(
                taz_original_id_to_contiguous_idx=taz_original_id_to_contiguous_idx,
                contiguous_idx_to_taz_original_id=contiguous_idx_to_taz_original_id,
                taz_name_to_original_id=taz_name_to_original_id,
                taz_original_id_to_universal_idx=taz_original_id_to_universal_idx,
                num_taz=num_taz,
                params=Odm_params
            )
            data.od_tensor = builder.build(odm_df) # Construir el tensor (si es necesario)
            # Esta llamada poblará data.od_pairs con los pares O-D procesados (universal_origin_idx, universal_dest_idx, demand)
            # Asumiendo que build_od_list_df retorna un DataFrame con columnas 'origin_node_idx', 'destination_node_idx', 'demand'
            data.od_pairs_df = builder.build_od_list_df(odm_df) # Usar un nombre distinto para evitar confusión si data.od_pairs se usa para otra cosa

            # Guardar los mapeos generados por el builder en el objeto data
            data.taz_contiguous_idx_to_original_id = contiguous_idx_to_taz_original_id
            # Este es un diccionario que mapea los índices contiguos (los números enteros de 0 a 68 que 
            # representan las filas y columnas de tu tensor O-D de 69x69) a los IDs originales de los 
            # nodos TAZ en tu network_nodes_gdf. Utilidad: Es tu "traductor" principal del tensor de vuelta a la 
            # red original. Si tienes un valor en la posición [i, j] del tensor O-D (donde i y j son índices 
            # contiguos), puedes usar data.taz_contiguous_idx_to_original_id[i] para obtener el ID original 
            # del nodo TAZ de origen y data.taz_contiguous_idx_to_original_id[j] para obtener el ID original del 
            # nodo TAZ de destino. Esto es crucial para visualizar resultados, analizar pares O-D específicos o 
            # vincular la demanda del tensor con otras propiedades de los nodos TAZ originales.
            
            data.taz_original_id_to_contiguous_idx = taz_original_id_to_contiguous_idx
            # Este es el inverso del anterior. Mapea los IDs originales de los nodos TAZ (los IDs que vienen
            #  en tu network_nodes_gdf) a los índices contiguos de 0 a 68.Utilidad: Es útil cuando tienes 
            # un ID original de TAZ y quieres saber qué fila o columna le corresponde en el tensor O-D. 
            # Por ejemplo, si quieres encontrar la demanda total de origen de una TAZ específica con un 
            # original_id conocido, primero encuentras su índice contiguo con 
            # data.taz_original_id_to_contiguous_idx[original_id] y luego accedes a la fila correspondiente en el 
            # tensor.

            data.taz_name_to_original_id = taz_name_to_original_id
            # Este diccionario mapea los nombres de las TAZs que aparecen en las columnas 'origin_zone' y 
            # 'destination_zone' de tu odm_df a los IDs originales de los nodos TAZ en tu network_nodes_gdf.
            # Utilidad: Este mapeo se utiliza durante la carga y construcción del tensor O-D para poder 
            # identificar qué nombres de zona en el odm_df corresponden a TAZs válidas y obtener sus IDs 
            # originales. Aunque se usa principalmente internamente en el OdTensorBuilder, tenerlo en el 
            # objeto data podría ser útil si en algún momento necesitas, por ejemplo, buscar el ID original
            #  de una TAZ basándote en su nombre tal como aparece en el archivo ODM.

            # Mapea el ID original de una TAZ a su indice universal (0 a num-nodes-1). Es util para el nuevo
            # DataFrame data.od_list_df. Permite saber que indice universal (que puedes usar para referenciar
            # data.x, data.pos, etc.) corresponde a una TAZ con un ID original dado.
            data.taz_original_id_to_universal_idx = taz_original_id_to_universal_idx

            logger.info(f"Procesados {len(data.od_pairs_df)} pares O-D con demanda desde el archivo ODM usando OdTensorBuilder.")


            # --- Cálculo de K-rutas por longitud para los pares O-D procesados ---

            """try:
                G = nx.read_graphml(Directories.graph_to_load)
                logging.info(f'se leyo grafo desde {Directories.graph_to_load} - Nodos: {len(G.nodes())} Links: {len(G.edges())}')
            except Warning as w:
                logging.warning(f'Construyendo grafo a partir de Geopackage y no de Graphml {w}')"""
            try:
                # Grafo para rutas por longitud
                G = nx.DiGraph()
                G.add_nodes_from(range(data.num_nodes))
                link_lengths_all = data.vdf_tensor[:, 4].cpu().tolist() # Columna 4 de vdf_tensor es 'length'

                edges_with_lengths = []
                for i, (u,v) in enumerate(data.edge_index.t().cpu().tolist()):
                    length = link_lengths_all[i]
                    # Usar un peso muy alto si la longitud es NaN, inf, o <=0 para que no sea elegida
                    weight = float(length) if pd.notna(length) and length > eps_val else float('inf')
                    edges_with_lengths.append((u,v, {'length_weight': weight, 'edge_idx': i}))
                G.add_edges_from(edges_with_lengths)
                # {(u, v): attr for u, v, attr in graph_for_length_paths.edges(data=True)}'
            except Exception as e:
                logging.error(f'No se pudo cargar el grafo para el calculo de las rutas {e}')

            # Comprobacion y Previsualizacion del grafo
            # Suponiendo que G es un DiGraph
            if nx.is_strongly_connected(G):
                print("El grafo dirigido es fuertemente conexo.")
            elif nx.is_weakly_connected(G):
                print("El grafo dirigido es débilmente conexo.")
            else:
                print("El grafo dirigido NO es conexo.")
            # pos = {i: (data.pos[i, 0].item(), data.pos[i, 1].item()) for i in range(data.pos.shape[0])}
            # nx.draw(graph_for_length_paths, pos, with_labels=True, node_color='lightblue', edge_color='gray', node_size=2, font_size=12)
            # plt.show()

            # Iterar sobre los pares O-D obtenidos del builder (data.od_pairs_df)
            # Asegúrate de que data.od_pairs_df tiene las columnas esperadas: 'origin_node_idx', 'destination_node_idx', 'demand'
            if not data.od_pairs_df.empty:
                if not all(col in data.od_pairs_df.columns for col in ['origin_node_idx', 'destination_node_idx', 'demand']):
                    logger.error("El DataFrame data.od_pairs_df generado por OdTensorBuilder no tiene las columnas esperadas para calcular K-rutas.")
                else:
                    logging.info(f'Inciando algoritmo de búsqueda de rutas mínimas')
                    # data.od_pairs_df
                    for _, row in data.od_pairs_df.iterrows():
                        orig_idx_universal = int(row['origin_node_idx']) # Asegurar tipo int
                        dest_idx_universal = int(row['destination_node_idx']) # Asegurar tipo int
                        demand_val = float(row['demand']) # Asegurar tipo float

                        if demand_val <= eps_val: continue

                        # Calcular K-shortest paths por longitud para este par O-D
                        try:
                            # Usar los índices universales directamente del DataFrame del builder
                            # logging.info(f"{orig_idx_universal} - {dest_idx_universal}")
                            paths_node_sequences = list(islice(nx.shortest_simple_paths(G,
                                                                                        source=orig_idx_universal,
                                                                                        target=dest_idx_universal,
                                                                                       weight='length_weight'), TrainingConfig.k_routes_for_ue_loss))
                            paths_link_indices_for_od = []
                            # logging.info(f"Para {orig_idx_universal} - {dest_idx_universal}: la secuencia es: {paths_node_sequences}")
                            # print(paths_node_sequences)
                            for node_path in paths_node_sequences:
                                current_path_link_indices = []
                                valid_path = True
                                for i_node in range(len(node_path) - 1):
                                    u, v = node_path[i_node], node_path[i_node+1]
                                    # Usar el mapeo global de par de nodos a índice de enlace
                                    edge_idx = node_pair_to_edge_idx.get((u,v))
                                    if edge_idx is not None:
                                        current_path_link_indices.append(edge_idx)
                                    else:
                                        # Esto no debería pasar si el grafo de NetworkX se construyó correctamente,
                                        # pero es una buena verificación.
                                        logger.warning(f"Arco ({u},{v}) de ruta precalculada para OD {orig_idx_universal}-{dest_idx_universal} no encontrado en edge_index. Ruta descartada.")
                                        valid_path = False; break
                                if valid_path and current_path_link_indices:
                                    paths_link_indices_for_od.append(current_path_link_indices)

                            if paths_link_indices_for_od:
                                data.k_shortest_paths_by_length_link_indices[(orig_idx_universal, dest_idx_universal)] = paths_link_indices_for_od
                            else:
                                # Si no se encontraron rutas válidas (ej. por arcos faltantes o rutas de 0 enlaces)
                                # TODO se implementa la logica de enlances que apuntan al mismo nodo?
                                data.k_shortest_paths_by_length_link_indices[(orig_idx_universal, dest_idx_universal)] = []

                        except nx.NetworkXNoPath:
                            # logger.debug(f"No length-based paths for OD {orig_idx_universal}-{dest_idx_universal}")
                            data.k_shortest_paths_by_length_link_indices[(orig_idx_universal, dest_idx_universal)] = []
                        except Exception as e_path:
                            logger.error(f"Error pre-calculando rutas para OD {orig_idx_universal}-{dest_idx_universal}: {e_path}")
                            data.k_shortest_paths_by_length_link_indices[(orig_idx_universal, dest_idx_universal)] = []

                logger.info(f"Calculadas K-rutas por longitud para {len(data.k_shortest_paths_by_length_link_indices)} pares O-D con demanda.")
            else:
                logger.warning("DataFrame data.od_pairs_df está vacío después del procesamiento con OdTensorBuilder. No se calcularán k-rutas.")
                data.k_shortest_paths_by_length_link_indices = {} # Asegurar que esté vacío
                
        data = _build_edge_mappings(data)
        data = add_virtual_links(data, config_various) # Asegurar que se pasan los tipos de nodo

        logger.info(f"Carga de datos reales completada. Nodos: {data.num_nodes}, Enlaces: {data.edge_index.shape[1]}")
        return data


def load_traffic_data_pickle(pickle_name: str, config_various: Various, initial_aux_ga_value: float = 0.0) -> Data:
    """
    Carga datos de tráfico desde un archivo pickle.
    Modificado para inicializar placeholders para G/A de nodos AUX en data.x.
    """
    # ... (código de búsqueda de archivo como en tu original) ...
    script_dir = os.path.dirname(os.path.abspath(__file__))
    pickle_route = os.path.join(script_dir, pickle_name)
    # ... (resto de la lógica de búsqueda) ...
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

    with open(pickle_route, 'rb') as f:
        datos = pickle.load(f)

    node_ids = datos['nodes']['node.ids']
    node_types_list_original = datos['nodes']['node.type'] # 'zat', 'intersection', potentially 'aux'
    node_coords_list = datos['nodes']['node.coordinates']
    node_demands_list = datos['nodes']['node.demand'] # Tuplas (gen, attr)

    num_nodes = len(node_ids)
    node_id_to_index = {nid: i for i, nid in enumerate(node_ids)}
    node_id_map_rev = {i: nid for nid, i in node_id_to_index.items()}

    x_list = []
    processed_node_types = [] # Para guardar el tipo de nodo procesado
    aux_node_indices_list = []
    zat_demands_dict = {} # Para data.zat_demands (solo TAZ)

    for i, node_id_original in enumerate(node_ids):
        ntype_original = node_types_list_original[i].lower()
        demand_tuple = node_demands_list[i]

        is_taz = 1.0 if ntype_original in [t.lower() for t in config_various.taz_node_types] else 0.0
        is_aux = 1.0 if ntype_original in [a.lower() for a in config_various.aux_node_types] else 0.0
        is_intersection = 1.0 if ntype_original in [it.lower() for it in config_various.intersection_node_types] else 0.0

        if not (is_taz or is_aux or is_intersection):
            # logger.warning(f"Nodo {node_id_original} (índice {i}) con tipo '{ntype_original}' no reconocido en pickle, asignado como intersección.")
            is_intersection = 1.0
            processed_node_types.append('intersection') # Guardar tipo procesado
        else:
            processed_node_types.append(ntype_original)


        gen_taz, attr_taz = 0.0, 0.0
        gen_aux_placeholder, attr_aux_placeholder = 0.0, 0.0
        
        if is_taz:
            gen_taz = float(demand_tuple[0])
            attr_taz = float(demand_tuple[1])
            zat_demands_dict[node_id_original] = [gen_taz, attr_taz]
        elif is_aux:
            aux_node_indices_list.append(i)
            gen_aux_placeholder = initial_aux_ga_value
            attr_aux_placeholder = initial_aux_ga_value
        
        # Estructura de x: [es_taz, es_aux, es_interseccion, gen_taz, attr_taz, gen_aux_placeholder, attr_aux_placeholder]
        x_list.append([
            is_taz, is_aux, is_intersection,
            gen_taz, attr_taz,
            gen_aux_placeholder, attr_aux_placeholder
        ])

    data = Data()
    data.x = torch.tensor(x_list, dtype=torch.float)
    data.pos = torch.tensor(node_coords_list, dtype=torch.float)
    data.aux_node_indices = torch.tensor(aux_node_indices_list, dtype=torch.long) if aux_node_indices_list else torch.empty((0,), dtype=torch.long)
    
    data.aux_node_original_id_to_aux_idx = {
        node_id_map_rev[global_idx]: aux_list_idx 
        for aux_list_idx, global_idx in enumerate(aux_node_indices_list)
    }

    # --- Enlaces y Flujos Observados (como en tu original, adaptado) ---
    link_ids_list = datos['links']['link.ids']
    link_type_list_orig = datos['links']['link.type']
    link_ij_list = datos['links']['link.ij']
    link_obs_flow_list = datos['links'].get('link.observed_flow', [None] * len(link_ij_list)) # Manejar si no existe

    edge_list_indices_tuples = []
    valid_link_types = []
    original_indices_map = {} # Mapea nuevo índice de arista a su índice original en el pickle

    for orig_idx, (u_id, v_id) in enumerate(link_ij_list):
        if u_id in node_id_to_index and v_id in node_id_to_index:
            u_idx = node_id_to_index[u_id]
            v_idx = node_id_to_index[v_id]
            
            current_edge_idx_in_new_list = len(edge_list_indices_tuples)
            edge_list_indices_tuples.append((u_idx, v_idx))
            valid_link_types.append(link_type_list_orig[orig_idx])
            original_indices_map[current_edge_idx_in_new_list] = orig_idx
        else:
            logger.warning(f"Arista ({u_id} -> {v_id}) en pickle ignorada. ID de nodo no encontrado.")

    if edge_list_indices_tuples:
        data.edge_index = torch.tensor(edge_list_indices_tuples, dtype=torch.long).t().contiguous()
    else:
        data.edge_index = torch.empty((2, 0), dtype=torch.long)

    # Flujos Observados
    observed_indices = []
    observed_values = []
    if link_obs_flow_list:
        for new_idx in range(data.edge_index.shape[1]): # Iterar sobre las aristas válidas añadidas
            orig_idx = original_indices_map[new_idx]
            if orig_idx < len(link_obs_flow_list): # Asegurar que el índice original es válido
                obs_flow = link_obs_flow_list[orig_idx]
                if obs_flow is not None:
                    observed_indices.append(new_idx) # Usar el nuevo índice de la arista
                    observed_values.append(float(obs_flow))
            else:
                logger.warning(f"Índice original {orig_idx} fuera de rango para link_obs_flow_list.")


    data.observed_flow_indices = torch.tensor(observed_indices, dtype=torch.long) if observed_indices else torch.empty((0,), dtype=torch.long)
    data.observed_flow_values = torch.tensor(observed_values, dtype=torch.float) if observed_values else torch.empty((0,), dtype=torch.float)

    # Atributos restantes
    data.node_types = processed_node_types # Usar los tipos procesados
    data.node_coordinates = node_coords_list
    data.zat_demands = zat_demands_dict # Solo TAZ
    data.node_id_map_rev = node_id_map_rev
    data.node_id_to_index = node_id_to_index
    data.num_nodes = num_nodes
    data.link_types = valid_link_types
    
    # Flujos asignados por LP (si existen en el pickle)
    if 'assigned_flow_values' in datos['links']: # Asumiendo que se guardan así
        lp_flows_list = datos['links']['assigned_flow_values']
        # Necesitamos mapear estos flujos a los nuevos índices de arista
        mapped_lp_flows = np.full(data.edge_index.shape[1], np.nan)
        for new_idx, orig_idx in original_indices_map.items():
            if orig_idx < len(lp_flows_list) and lp_flows_list[orig_idx] is not None:
                mapped_lp_flows[new_idx] = float(lp_flows_list[orig_idx])
        data.lp_assigned_flows = torch.tensor(mapped_lp_flows, dtype=torch.float)
    elif 'results_df' in datos and isinstance(datos['results_df'], pd.DataFrame):
        # Intentar obtener de results_df si 'assigned_flow_values' no está
        # Esto requiere que 'results_df' tenga 'origin', 'dest' y 'estimated_flow'
        # y que podamos mapear (origin, dest) a los nuevos índices de arista.
        # Esta parte es más compleja y depende de cómo esté estructurado 'results_df'.
        # Por simplicidad, la omito aquí, pero sería necesario un mapeo robusto.
        logger.info("Intentando obtener 'lp_assigned_flows' de 'results_df', requiere implementación de mapeo.")
        # Crear un tensor de NaNs por ahora si no se puede mapear fácilmente
        data.lp_assigned_flows = torch.full((data.edge_index.shape[1],), float('nan'), dtype=torch.float)


    data = _build_edge_mappings(data)
    data = add_virtual_links(data, config_various)

    logger.info(f"Datos de pickle procesados: {data.num_nodes} nodos y {data.edge_index.shape[1]} aristas.")
    logger.info(f"Índices de nodos AUX (data.aux_node_indices): {data.aux_node_indices.tolist()}")
    return data


# Definición de VDFPreprocessor (tal como la proporcionaste)
@dataclass
class VDFPreprocessor:
    """
    Preprocesador para generar grupos de curvas VDF y asignar parámetros alpha/beta y capacidad.
    """
    df: pd.DataFrame # Se espera que este df sea network_links_gdf
    default_alpha: float = 0.15
    default_beta: float = 4.0
    default_capacity: float = 2000.0 # Capacidad por carril por hora (ejemplo)

    # Columnas esperadas en el df de entrada (network_links_gdf)
    # 'edge_type', 'emme_LANES', 'emme_VDF', 'emme_@hast' (velocidad flujo libre), 
    # 'LENGTH' (longitud del enlace), 'emme_@vtyp' (tipo de vía EMME)

    road_df: pd.DataFrame = field(init=False)
    nonroad_df: pd.DataFrame = field(init=False)
    vdf_groups: dict = field(init=False, default_factory=dict)
    params: dict = field(init=False, default_factory=dict) # Para alpha, beta
    capacities_group: dict = field(init=False, default_factory=dict) # Para capacidad por grupo VDF
    
    # Columnas que se usarán para el tensor final
    # El orden es importante: [capacity, FFS, lanes, length, alpha, beta]
    # 'lanes' se usa para calcular capacidad pero no va directo al tensor BPR usual.
    # 'ff_speed' se usa para calcular free_flow_time.
    _output_cols_bpr: list = field(default_factory=lambda: ['capacity', 'FFS', 'lanes', 'length', 'alpha', 'beta'])


    def __post_init__(self):
        self.df = self.df.copy()
        if 'edge_type' not in self.df.columns:
            logger.warning("'edge_type' no encontrado en el DataFrame de enlaces. Asumiendo todos los enlaces como 'unknown'.")
            self.df['edge_type'] = 'unknown'
        self.split_by_edge_type()
        self._ensure_road_df_columns() # Asegurar que las columnas necesarias existan en road_df

    def _ensure_road_df_columns(self):
        """Asegura que las columnas necesarias para VDF existan en road_df, rellenando con defaults si es necesario."""
        # Columnas EMME que podrían o no estar presentes. Usaremos nombres genéricos después.
        # El usuario mencionó que su VDFPreprocessor usa:
        # 'emme_LANES', 'emme_VDF', 'emme_@hast', 'LENGTH', 'emme_@vtyp'
        required_cols_map = {
            'lanes': 'emme_LANES',
            'vdf_emme': 'emme_VDF', # Identificador de curva VDF de EMME
            'free_flow_speed_kmh': 'emme_@hast',
            'length_km': 'LENGTH', # Asumimos que LENGTH está en KM. Si no, ajustar.
            'road_type_emme': 'emme_@vtyp'
        }
        
        default_values = {
            'lanes': 1,
            'vdf_emme': 0, # Un VDF por defecto
            'free_flow_speed_kmh': 50,
            'length_km': 0.1, # Longitud por defecto pequeña
            'road_type_emme': 0 # Un tipo de vía por defecto
        }

        if not self.road_df.empty:
            for generic_name, emme_name in required_cols_map.items():
                if emme_name not in self.road_df.columns:
                    logger.warning(f"Columna '{emme_name}' (para '{generic_name}') no encontrada en road_df. Usando valor por defecto: {default_values[generic_name]}")
                    self.road_df[emme_name] = default_values[generic_name]
                # Convertir a numérico, errores a NaN y luego rellenar NaN
                self.road_df[emme_name] = pd.to_numeric(self.road_df[emme_name], errors='coerce').fillna(default_values[generic_name])
        # Renombrar a nombres genéricos para uso interno si se desea, o usar los emme_names directamente.
        # Por ahora, VDFPreprocessor usa los nombres emme_ directamente.

    def split_by_edge_type(self):
        mask = self.df['edge_type'].str.lower() == 'road' # Insensible a mayúsculas
        self.road_df = self.df[mask].copy()
        self.nonroad_df = self.df[~mask].copy()
        logger.info(f"Enlaces 'road': {len(self.road_df)}, Enlaces no 'road': {len(self.nonroad_df)}")

    def group_curves(self, by: str = 'composite'): # 'composite' como default si Vdf.grouped_by no está definido
        """
        Agrupa por: 'emme_VDF', 'emme_@vtyp', o 'composite' (vtyp + hast).
        """
        # ... (lógica de group_curves como la proporcionaste, adaptada para usar los nombres de columna correctos)
        valid_by = ('emme_VDF', 'emme_@vtyp', 'composite')
        if by not in valid_by:
            logger.error(f"Método de agrupación '{by}' no válido. Debe ser uno de {valid_by}. Usando 'composite'.")
            by = 'composite'

        self.vdf_groups.clear()
        self.params.clear()
        self.capacities_group.clear()

        if self.road_df.empty:
            logger.warning("road_df está vacío. No se pueden crear grupos VDF.")
            return

        if by == 'composite':
            # Agrupar por 'emme_@vtyp' y 'emme_@hast'
            # Asegurar que estas columnas existan y no tengan NaNs para la agrupación
            group_cols = ['emme_@vtyp', 'emme_@hast']
            if not all(col in self.road_df.columns for col in group_cols):
                logger.error(f"Columnas para agrupación composite {group_cols} no encontradas en road_df.")
                return
            
            # Iterar sobre combinaciones únicas de vtyp y hast
            for (vtyp_val, speed_val), subdf in self.road_df.groupby(group_cols):
                name = f"VTYP_{vtyp_val}_HAST_{speed_val}"
                self.vdf_groups[name] = subdf.copy()
        elif by == 'emme_VDF':
            if 'emme_VDF' not in self.road_df.columns: logger.error("'emme_VDF' no encontrado."); return
            for vdf_val, subdf in self.road_df.groupby('emme_VDF'):
                name = f"VDF_{vdf_val}"
                self.vdf_groups[name] = subdf.copy()
        elif by == 'emme_@vtyp':
            if 'emme_@vtyp' not in self.road_df.columns: logger.error("'emme_@vtyp' no encontrado."); return
            for vtyp_val, subdf in self.road_df.groupby('emme_@vtyp'):
                name = f"VTYP_{vtyp_val}"
                self.vdf_groups[name] = subdf.copy()
        
        for name in self.vdf_groups:
            self.params[name] = {'alpha': self.default_alpha, 'beta': self.default_beta}
            # La capacidad por defecto aquí es por carril. Se multiplicará por carriles luego.
            self.capacities_group[name] = self.default_capacity 
        logger.info(f"Curvas VDF agrupadas por '{by}'. {len(self.vdf_groups)} grupos creados.")

    def set_manual_params(self, manual_params: dict): # Para alpha, beta
        if not self.vdf_groups: logger.warning("No hay grupos VDF definidos. Llama a group_curves primero."); return
        for name, p_values in manual_params.items():
            if name in self.params:
                self.params[name]['alpha'] = p_values.get('alpha', self.params[name]['alpha'])
                self.params[name]['beta'] = p_values.get('beta', self.params[name]['beta'])
            else:
                logger.warning(f"Grupo VDF '{name}' no encontrado en set_manual_params. Ignorando.")

    def set_capacity_by_group(self, cap_params_group: dict): # Para capacidad por grupo VDF
        if not self.vdf_groups: logger.warning("No hay grupos VDF definidos. Llama a group_curves primero."); return
        for name, cap_val in cap_params_group.items():
            if name in self.capacities_group:
                self.capacities_group[name] = cap_val
            else:
                logger.warning(f"Grupo VDF '{name}' no encontrado en set_capacity_by_group. Ignorando.")

    def get_vdf_tensor(self) -> pd.DataFrame:
        """
        Prepara el DataFrame final con parámetros BPR para todos los enlaces.
        Columnas: [capacity, free_flow_time, alpha, beta, length]
        Los valores son NaN para enlaces no-road.
        """
        # DataFrame final con el índice original de self.df (todos los enlaces)
        # y las columnas requeridas para BPR.
        final_params_df = pd.DataFrame(index=self.df.index, columns=self._output_cols_bpr, dtype=float)

        if self.road_df.empty:
            logger.warning("road_df está vacío, no se pueden asignar parámetros BPR.")
            return final_params_df # Retorna DataFrame de NaNs

        for group_name, group_df in self.vdf_groups.items():
            group_alpha = self.params.get(group_name, {}).get('alpha', self.default_alpha)
            group_beta = self.params.get(group_name, {}).get('beta', self.default_beta)
            
            # Capacidad por carril para este grupo
            group_capacity_per_lane = self.capacities_group.get(group_name, self.default_capacity)

            for link_idx, row in group_df.iterrows():
                lanes = float(row.get('emme_LANES', 0)) # Usar .get con default
                link_capacity_per_lane = group_capacity_per_lane
                ff_speed_kmh = float(row.get('emme_@hast', 0))
                length_km = float(row.get('LENGTH', 0.001))
                # ['capacity', 'FFS', 'lanes', 'length', 'alpha', 'beta']
                final_params_df.loc[link_idx] = [link_capacity_per_lane, ff_speed_kmh, lanes, length_km, group_alpha, group_beta]
        
        logger.info(f"Tensor de parámetros VDF generado para {len(self.road_df)} enlaces 'road'.")
        return final_params_df
    
class OdTensorBuilder:
    def __init__(self,
                 taz_original_id_to_contiguous_idx: dict,
                 contiguous_idx_to_taz_original_id: dict,
                 taz_name_to_original_id: dict,
                 taz_original_id_to_universal_idx: dict, # Nuevo: mapeo de ID original TAZ a índice universal
                 num_taz: int,
                 params: Odm_params,
                 logger: logging.Logger = None):
        """
        Inicializa el constructor del tensor O-D y la lista agregada para TAZs.

        :param taz_original_id_to_contiguous_idx: mapeo de IDs originales de TAZ (str) a índices contiguos (int)
        :param contiguous_idx_to_taz_original_id: mapeo inverso de índices contiguos (int) a IDs originales de TAZ (str)
        :param taz_name_to_original_id: mapeo de nombres de TAZ (str) a IDs originales de TAZ (str)
        :param taz_original_id_to_universal_idx: mapeo de IDs originales de TAZ (str) a índices universales (int)
        :param num_taz: Número total de TAZs.
        :param params: parámetros de agregación en un dataclass OdmParams
        :param logger: logger para mensajes de info/warning
        """
        self.taz_original_id_to_contiguous_idx = taz_original_id_to_contiguous_idx
        self.contiguous_idx_to_taz_original_id = contiguous_idx_to_taz_original_id
        self.taz_name_to_original_id = taz_name_to_original_id
        self.taz_original_id_to_universal_idx = taz_original_id_to_universal_idx # Guardar el nuevo mapeo
        self.num_taz = num_taz
        self.taz_original_ids = set(taz_original_id_to_contiguous_idx.keys())
        self.taz_original_ids_list = [contiguous_idx_to_taz_original_id[i] for i in range(num_taz)]
        self.params = params
        self.logger = logger or logging.getLogger(__name__)

    def _preprocess_odm_df(self, odm_df: pd.DataFrame) -> pd.DataFrame:
        """
        Realiza el preprocesamiento común para construir tanto el tensor como el DataFrame.
        Mapea nombres de zona a IDs originales de TAZ y aplica filtros de fecha.
        Retorna un DataFrame listo para agregación.
        """
        df = odm_df.copy()

        # Asegurar que las columnas existan y tengan el tipo correcto
        if 'date' not in df.columns:
             self.logger.warning("Columna 'date' no encontrada en odm_df. Se asume que no se requiere agregación por fecha.")
             df['date'] = pd.to_datetime('today').normalize() # Asignar una fecha dummy si no existe
        else:
            df['date'] = pd.to_datetime(df['date'])

        if 'people' not in df.columns:
            self.logger.error("Columna 'people' (demanda) no encontrada en odm_df.")
            raise ValueError("Columna 'people' faltante en odm_df.")
        df['people'] = pd.to_numeric(df['people'], errors='coerce').fillna(0.0) # Convertir a float, NaN a 0

        if 'origin_zone' not in df.columns or 'destination_zone' not in df.columns:
             self.logger.error("Columnas 'origin_zone' o 'destination_zone' faltantes en odm_df.")
             raise ValueError("Columnas de zona O-D faltantes en odm_df.")

        # Mapear nombres de zona a IDs originales de TAZ
        df['origin_original_id'] = df['origin_zone'].astype(str).map(self.taz_name_to_original_id)
        df['dest_original_id'] = df['destination_zone'].astype(str).map(self.taz_name_to_original_id)

        # Filtrar filas donde el origen o destino no pudo ser mapeado a un ID original de TAZ
        initial_rows = len(df)
        df = df.dropna(subset=['origin_original_id', 'dest_original_id'])
        filtered_rows = len(df)
        if initial_rows > filtered_rows:
            self.logger.info(f"Filtradas {initial_rows - filtered_rows} filas de odm_df donde el nombre de origen o destino no corresponde a una TAZ conocida.")

        # Ahora, filtrar filas donde el ID original mapeado no es una TAZ válida (redundante si taz_name_to_original_id solo incluye TAZs, pero seguro)
        initial_rows = len(df)
        df = df[df['origin_original_id'].isin(self.taz_original_ids) &
                df['dest_original_id'].isin(self.taz_original_ids)]
        filtered_rows = len(df)
        if initial_rows > filtered_rows:
             self.logger.info(f"Filtradas {initial_rows - filtered_rows} filas de odm_df donde el ID original mapeado no es una TAZ válida.")

        if df.empty:
            self.logger.warning("DataFrame ODM vacío después de filtrar por TAZs válidas.")
            return pd.DataFrame(columns=['origin_original_id', 'dest_original_id', 'date', 'people']) # Retornar df vacío con columnas esperadas

        # Aplicar filtro de fecha según el método de agregación
        method = self.params.aggregation_method
        dates_in_df = df['date'].dt.normalize().unique()
        filtered_df = df.copy() # Usar una copia para el filtrado por fecha

        dates_used = []
        if method == 'one_date':
            if not self.params.one_date:
                raise ValueError("Debe especificar 'one_date' para aggregation_method='one_date'.")
            try:
                day = pd.to_datetime(self.params.one_date).normalize()
                filtered_df = filtered_df[filtered_df['date'].dt.normalize() == day]
                dates_used = [day]
            except Exception as e:
                 self.logger.error(f"Error al parsear 'one_date': {self.params.one_date}. {e}")
                 raise ValueError(f"Formato de fecha inválido para 'one_date': {self.params.one_date}")

        elif method == 'weekdays':
            filtered_df = filtered_df[filtered_df['date'].dt.weekday < 5]
            dates_used = filtered_df['date'].dt.normalize().unique()

        elif method == 'weekends':
            filtered_df = filtered_df[filtered_df['date'].dt.weekday >= 5]
            dates_used = filtered_df['date'].dt.normalize().unique()

        elif method == 'dates_range':
            if not self.params.dates_range or len(self.params.dates_range) != 2:
                raise ValueError("Debe especificar 'dates_range' con dos fechas para aggregation_method='dates_range'.")
            try:
                start_dt = pd.to_datetime(self.params.dates_range[0]).normalize()
                end_dt = pd.to_datetime(self.params.dates_range[1]).normalize()
                filtered_df = filtered_df[(filtered_df['date'].dt.normalize() >= start_dt) & (filtered_df['date'].dt.normalize() <= end_dt)]
                dates_used = filtered_df['date'].dt.normalize().unique()
            except Exception as e:
                 self.logger.error(f"Error al parsear 'dates_range': {self.params.dates_range}. {e}")
                 raise ValueError(f"Formato de fecha inválido para 'dates_range': {self.params.dates_range}")
        else:  # 'average' o cualquier otro caso
            dates_used = dates_in_df # Usar todas las fechas si no se especifica un método o es 'average'

        days_count = len(dates_used)
        if days_count == 0 and method != 'average':
             raise ValueError("No hay días en el filtro de fecha especificado.")
        elif days_count == 0 and method == 'average':
             self.logger.warning("No hay días en el filtro de fecha especificado para calcular el promedio.")
             # Retornar un DataFrame vacío con las columnas esperadas si no hay datos
             return pd.DataFrame(columns=['origin_original_id', 'dest_original_id', 'date', 'people'])


        # Retornar el DataFrame filtrado y con IDs originales mapeados
        return filtered_df


    def build(self, odm_df: pd.DataFrame) -> np.ndarray:
        """
        Construye el tensor O-D agregado para las TAZs usando índices contiguos.
        """
        # Usar el preprocesamiento común
        filtered_df = self._preprocess_odm_df(odm_df)

        if filtered_df.empty:
             self.logger.warning("DataFrame preprocesado está vacío. No se puede construir el tensor O-D.")
             return np.zeros((self.num_taz, self.num_taz), dtype=float)

        # Mapear IDs originales de TAZ a índices contiguos de TAZ para el tensor
        filtered_df['origin_idx_contiguous'] = filtered_df['origin_original_id'].map(self.taz_original_id_to_contiguous_idx)
        filtered_df['dest_idx_contiguous'] = filtered_df['dest_original_id'].map(self.taz_original_id_to_contiguous_idx)

        # Filtrar filas donde el mapeo a índice contiguo falló (debería ser 0 si los pasos anteriores fueron correctos)
        filtered_df = filtered_df.dropna(subset=['origin_idx_contiguous', 'dest_idx_contiguous'])

        if filtered_df.empty:
             self.logger.warning("DataFrame está vacío después de mapear a índices contiguos. No se puede construir el tensor O-D.")
             return np.zeros((self.num_taz, self.num_taz), dtype=float)


        # Sumar demanda por origen, destino y fecha (usando índices contiguos)
        daily_aggregated = (
            filtered_df.astype({'origin_idx_contiguous': int, 'dest_idx_contiguous': int})
            .groupby(['origin_idx_contiguous', 'dest_idx_contiguous', pd.Grouper(key='date', freq='D')])['people']
            .sum()
            .reset_index()
        )

        # Inicializar tensor O-D con las dimensiones correctas para TAZs
        od_tensor = np.zeros((self.num_taz, self.num_taz), dtype=float)

        if not daily_aggregated.empty:
            days_count = len(daily_aggregated['date'].unique()) # Contar días reales en los datos agregados
            if days_count == 0: # Esto no debería pasar si filtered_df no estaba vacío, pero por seguridad
                 self.logger.warning("No hay días en los datos agregados diariamente. Tensor O-D será de ceros.")
                 return np.zeros((self.num_taz, self.num_taz), dtype=float)

            if self.params.aggregation_method == 'average':
                # Calcular el promedio diario por par O-D
                avg = (
                    daily_aggregated.groupby(['origin_idx_contiguous', 'dest_idx_contiguous'])['people']
                    .mean()
                    .reset_index()
                )
                # Llenar el tensor usando los índices contiguos
                for _, row in avg.iterrows():
                    orig_idx = int(row['origin_idx_contiguous'])
                    dest_idx = int(row['dest_idx_contiguous'])
                    od_tensor[orig_idx, dest_idx] = row['people']
            else:
                # Sumar la demanda total para los días filtrados
                total = (
                    daily_aggregated.groupby(['origin_idx_contiguous', 'dest_idx_contiguous'])['people']
                    .sum()
                    .reset_index()
                )
                # Llenar el tensor usando los índices contiguos y dividir por el número de días si se requiere promedio
                # (La lógica original dividía siempre por days_count si no era 'average', lo mantengo)
                for _, row in total.iterrows():
                    orig_idx = int(row['origin_idx_contiguous'])
                    dest_idx = int(row['dest_idx_contiguous'])
                    # Dividir por days_count para obtener el promedio diario para el período filtrado
                    od_tensor[orig_idx, dest_idx] = row['people'] / days_count

        self.logger.info(f"Tensor O-D para TAZs construido con dimensiones {od_tensor.shape}.")
        return od_tensor

    def build_od_list_df(self, odm_df: pd.DataFrame) -> pd.DataFrame:
        """
        Construye un DataFrame con los pares O-D agregados para las TAZs,
        incluyendo pares con demanda cero, usando los índices universales de los nodos.
        Columnas: 'origin_node_idx', 'destination_node_idx', 'demand'.
        """
        # Usar el preprocesamiento común
        filtered_df = self._preprocess_odm_df(odm_df)

        # Sumar demanda por origen, destino y fecha (usando IDs originales de TAZ)
        # Agregamos por IDs originales primero para luego unir con el grid completo
        if not filtered_df.empty:
            daily_aggregated = (
                filtered_df.groupby(['origin_original_id', 'dest_original_id', pd.Grouper(key='date', freq='D')])['people']
                .sum()
                .reset_index()
            )
        else:
             daily_aggregated = pd.DataFrame(columns=['origin_original_id', 'dest_original_id', 'date', 'people'])


        if daily_aggregated.empty:
             self.logger.warning("DataFrame agregado diariamente (por ID original) está vacío. DataFrame de lista O-D será vacío.")
             # Si no hay datos agregados, aún debemos retornar el grid completo con demanda 0
             # return pd.DataFrame(columns=['origin_node_idx', 'destination_node_idx', 'demand'])
             aggregated_demand = pd.DataFrame(columns=['origin_original_id', 'dest_original_id', 'demand']) # DataFrame vacío para unir

        else:
            days_count = len(daily_aggregated['date'].unique())
            if days_count == 0:
                 self.logger.warning("No hay días en los datos agregados diariamente (por ID original). DataFrame de lista O-D será vacío.")
                 aggregated_demand = pd.DataFrame(columns=['origin_original_id', 'dest_original_id', 'demand']) # DataFrame vacío para unir
            else:
                if self.params.aggregation_method == 'average':
                    # Calcular el promedio diario por par O-D (usando IDs originales)
                    aggregated_demand = (
                        daily_aggregated.groupby(['origin_original_id', 'dest_original_id'])['people']
                        .mean()
                        .reset_index()
                    )
                else:
                    # Sumar la demanda total para los días filtrados y dividir por días si se requiere promedio (usando IDs originales)
                    aggregated_demand = (
                        daily_aggregated.groupby(['origin_original_id', 'dest_original_id'])['people']
                        .sum()
                        .reset_index()
                    )
                    # Dividir por days_count para obtener el promedio diario para el período filtrado
                    aggregated_demand['people'] = aggregated_demand['people'] / days_count

                # Renombrar la columna de demanda
                aggregated_demand = aggregated_demand.rename(columns={'people': 'demand'})


        # --- Paso clave: Crear un DataFrame con todos los pares O-D posibles (69x69) ---
        # Usamos la lista ordenada de IDs originales de TAZs que creamos en __init__
        all_origins = self.taz_original_ids_list
        all_destinations = self.taz_original_ids_list

        # Crear todas las combinaciones de origen y destino
        full_od_grid = pd.DataFrame({
            'origin_original_id': np.repeat(all_origins, len(all_destinations)),
            'dest_original_id': np.tile(all_destinations, len(all_origins))
        })

        # --- Unir la demanda agregada con el grid completo ---
        # Usamos un merge left para mantener todos los pares del grid completo
        # Los pares que no estaban en aggregated_demand tendrán NaN en la columna 'demand'
        od_list_df = pd.merge(
            full_od_grid,
            aggregated_demand,
            on=['origin_original_id', 'dest_original_id'],
            how='left'
        )

        # Rellenar los valores NaN en 'demand' con 0.0
        od_list_df['demand'] = od_list_df['demand'].fillna(0.0)

        # --- Mapear IDs originales a índices universales ---
        # Ahora que tenemos todos los pares y su demanda (incluyendo ceros),
        # mapeamos los IDs originales a los índices universales
        od_list_df['origin_node_idx'] = od_list_df['origin_original_id'].map(self.taz_original_id_to_universal_idx)
        od_list_df['destination_node_idx'] = od_list_df['dest_original_id'].map(self.taz_original_id_to_universal_idx)

        # Reordenar columnas y seleccionar las finales
        od_list_df = od_list_df[['origin_node_idx', 'destination_node_idx', 'demand']]

        # Asegurarse de que los índices universales son enteros
        od_list_df = od_list_df.astype({'origin_node_idx': int, 'destination_node_idx': int})


        self.logger.info(f"DataFrame de lista O-D para TAZs construido con {len(od_list_df)} pares (incluyendo demanda cero).")
        return od_list_df

def _build_edge_mappings(data: Data) -> Data:
    """
    Construye listas de índices de aristas entrantes y salientes para cada nodo.
    """
    num_nodes = data.num_nodes
    # Asegurar que edge_index no esté vacío antes de acceder a size(1)
    if data.edge_index.numel() == 0:
        logger.warning("edge_index está vacío. No se pueden construir mapeos de aristas.")
        data.in_edges_idx_tonode = [torch.empty((0,), dtype=torch.long) for _ in range(num_nodes)]
        data.out_edges_idx_tonode = [torch.empty((0,), dtype=torch.long) for _ in range(num_nodes)]
        return data

    num_edges = data.edge_index.size(1)
    in_edges = [[] for _ in range(num_nodes)]
    out_edges = [[] for _ in range(num_nodes)]

    edge_index_np = data.edge_index.cpu().numpy() # Mover a CPU para numpy
    for e in range(num_edges):
        src = int(edge_index_np[0, e])
        dst = int(edge_index_np[1, e])
        if 0 <= src < num_nodes and 0 <= dst < num_nodes:
            out_edges[src].append(e)
            in_edges[dst].append(e)
        else:
            logger.warning(f"Índice de nodo fuera de rango en edge_index: src={src}, dst={dst}. Num_nodes={num_nodes}. Arista {e} ignorada para mapeos.")


    data.in_edges_idx_tonode = [torch.tensor(idx_list, dtype=torch.long) if idx_list else torch.empty((0,), dtype=torch.long)
                                for idx_list in in_edges]
    data.out_edges_idx_tonode = [torch.tensor(idx_list, dtype=torch.long) if idx_list else torch.empty((0,), dtype=torch.long)
                                 for idx_list in out_edges]
    return data

def add_virtual_links(data: Data, config_various: Various) -> Data:
    """
    Crea enlaces virtuales entre todos los nodos tipo TAZ o AUX.
    """
    if hasattr(data, 'virtual_edge_index') and data.virtual_edge_index is not None and data.virtual_edge_index.numel() > 0:
        logger.info("virtual_edge_index ya existe y no está vacío. No se agregarán nuevos enlaces virtuales.")
        return data

    # Usar los tipos de nodo definidos en config_various
    taz_types_lower = [t.lower() for t in config_various.taz_node_types]
    aux_types_lower = [a.lower() for a in config_various.aux_node_types]
    
    source_destination_node_indices = [
        idx for idx, ntype_original in enumerate(data.node_types)
        if str(ntype_original).lower() in taz_types_lower + aux_types_lower
    ]
    
    virtual_edges = []
    if len(source_destination_node_indices) > 1:
        for i in range(len(source_destination_node_indices)):
            for j in range(len(source_destination_node_indices)):
                if i != j: # Evitar auto-bucles
                    src_node_global_idx = source_destination_node_indices[i]
                    dst_node_global_idx = source_destination_node_indices[j]
                    virtual_edges.append((src_node_global_idx, dst_node_global_idx))
        logger.info(f"Agregados {len(virtual_edges)} enlaces virtuales entre nodos TAZ/AUX.")
    else:
        logger.info("No hay suficientes nodos TAZ/AUX para crear enlaces virtuales.")

    if virtual_edges:
        data.virtual_edge_index = torch.tensor(virtual_edges, dtype=torch.long).t().contiguous()
    else:
        data.virtual_edge_index = torch.empty((2, 0), dtype=torch.long)
    
    return data


def add_virtual_and_logical_links(data: Data) -> Data:
    """
    Agrega enlaces virtuales (lógicos) entre ZATs y su intersección más cercana.
    Esta función es específica para la lógica original de `add_virtual_and_logical_links`
    y podría necesitar ser adaptada o fusionada con `add_virtual_links` si la lógica es la misma.
    """
    if hasattr(data, 'virtual_edge_index') and data.virtual_edge_index is not None and data.virtual_edge_index.numel() > 0 :
        logger.info("virtual_edge_index ya existe. No se agregarán nuevos enlaces lógicos/virtuales por este método.")
        return data

    virtual_edges = []
    # Asumimos que config_various está disponible o se pasa como argumento si es necesario
    # Para este ejemplo, usaré una definición simple de tipos.
    taz_node_type_str = 'zat' # Ejemplo, ajustar a tus datos
    intersection_node_type_str = 'intersection' # Ejemplo

    zat_indices = [i for i, nt in enumerate(data.node_types) if str(nt).lower() == taz_node_type_str]
    intersection_indices = [i for i, nt in enumerate(data.node_types) if str(nt).lower() == intersection_node_type_str]

    if not zat_indices or not intersection_indices:
        logger.warning("No hay nodos ZAT o de intersección para crear enlaces lógicos/virtuales.")
        data.virtual_edge_index = torch.empty((2,0), dtype=torch.long)
        return data

    for zat_idx in zat_indices:
        zat_coord = data.node_coordinates[zat_idx]
        min_dist = float('inf')
        closest_intersection_global_idx = None
        
        for inter_idx_global in intersection_indices:
            inter_coord = data.node_coordinates[inter_idx_global]
            dist = ((zat_coord[0] - inter_coord[0]) ** 2 + (zat_coord[1] - inter_coord[1]) ** 2) ** 0.5
            if dist < min_dist:
                min_dist = dist
                closest_intersection_global_idx = inter_idx_global
        
        if closest_intersection_global_idx is not None:
            virtual_edges.append((zat_idx, closest_intersection_global_idx))
            virtual_edges.append((closest_intersection_global_idx, zat_idx)) # Enlace bidireccional

    if virtual_edges:
        data.virtual_edge_index = torch.tensor(virtual_edges, dtype=torch.long).t().contiguous()
        logger.info(f"Agregados {len(virtual_edges)} enlaces lógicos (virtuales) ZAT-Intersección.")
    else:
        data.virtual_edge_index = torch.empty((2, 0), dtype=torch.long)
        logger.info("No se crearon enlaces lógicos (virtuales) ZAT-Intersección.")
    return data

# Ejemplo de cómo podrías cargar datos (descomentar y adaptar según sea necesario):
# if __name__ == '__main__':
#     from config import Various as ConfigVarious # Asegúrate que Various esté definido en config.py
#     various_config = ConfigVarious()
#     # Para datos reales:
#     # real_network_gpkg = 'path/to/your/final_network.gpkg'
#     # odm_csv = 'path/to/your/odm.csv'
#     # data_object = load_real_traffic_data(real_network_gpkg, odm_csv, various_config)
#     # print(data_object)
#     # print(f"Nodos AUX: {data_object.aux_node_indices}")

#     # Para datos de pickle:
#     pickle_file_name = 'traffic_data_big.pkl' # o el nombre de tu archivo
#     data_object_pickle = load_traffic_data_pickle(pickle_file_name, various_config)
#     print(data_object_pickle)
#     print(f"Nodos AUX (pickle): {data_object_pickle.aux_node_indices}")
#     print(f"x shape (pickle): {data_object_pickle.x.shape}")
#     print(f"x sample (pickle):\n{data_object_pickle.x[:5]}")
