#!/usr/bin/env python3
"""
Graph Generator and Traffic Assignment for ZAT/Intersection Networks
---------------------------------------------------------------------
This script generates a network of ZATs (zones or traffic analysis zones) and intersections,
creates links (roads and logical connections), generates an OD matrix and observed flows,
solves a traffic assignment problem via linear programming, and finally visualizes and saves
the results.

Main flow:
1. Generate nodes (ZATs and intersections) with coordinates.
2. Create network links:
   - Road links between intersections (with a completeness percentage)
   - Logical links connecting each ZAT to its nearest intersection.
3. Generate a ZAT-to-ZAT origin-destination demand matrix.
4. Calculate node features (generation/attraction for ZATs).
5. Generate synthetic observed flows on a subset of links.
6. Solve the traffic assignment using linear programming.
7. Visualize the network and save the results.
"""

import os
import math
import json
import random
import pickle
import logging
from datetime import datetime
import numpy as np
import pandas as pd
from tqdm import tqdm
import pulp
import math
import logging
from pyvis.network import Network

# -------------------------
# Logging Setup
# -------------------------
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")

# -------------------------
#Some utils
# -------------------------

class FilterableDict:
    def __init__(self, data_dict):
        """
        Initializes the class with a dictionary of arrays or lists.

        Parameters:
        - data_dict: Dictionary where values are lists or arrays (e.g., node info, links, etc.)
        """
        self.data = {k: np.array(v) for k, v in data_dict.items()}

    def filter_by_value(self, field, value):
        """
        Filters the dictionary where data[field] == value.

        Parameters:
        - field: The key on which to apply the filter
        - value: The exact value to match
        """
        mask = self.data[field] == value
        return {k: v[mask] for k, v in self.data.items()}

    def filter_by_values(self, field, values):
        """
        Filters the dictionary where data[field] is in values.

        Parameters:
        - field: The key on which to apply the filter
        - values: A list or set of accepted values
        """
        mask = np.isin(self.data[field], values)
        return {k: v[mask] for k, v in self.data.items()}

    def filter_by_function(self, field, function):
        """
        Filters the dictionary using a boolean function applied to each value in a field.

        Parameters:
        - field: The key on which to apply the filter
        - function: A function that returns True for values to keep
        """
        mask = np.array([function(x) for x in self.data[field]])
        return {k: v[mask] for k, v in self.data.items()}


# -------------------------
# Node and Link Generation Functions
# -------------------------

def generate_nodes(num_zats: int, num_intersections: int):
    """
    Genera nodos para ZATs e intersecciones con coordenadas aleatorias.

    Args:
        num_zats (int): Número de nodos tipo ZAT.
        num_intersections (int): Número de nodos tipo intersección.

    Returns:
        dict: Diccionario con IDs, tipos y coordenadas de los nodos.
    """
    node_ids = []
    node_type = []
    node_coordinates = []

    # Generate ZAT nodes with random coordinates
    for i in range(num_zats):
        zat_id = f'Z{i}'
        node_ids.append(zat_id)
        node_type.append('zat')
        node_coordinates.append((random.uniform(0, 100), random.uniform(0, 100)))

    # Generate Intersection nodes with random coordinates
    for i in range(num_intersections):
        int_id = f'I{i}'
        node_ids.append(int_id)
        node_type.append('intersection')
        node_coordinates.append((random.uniform(0, 100), random.uniform(0, 100)))

    return {'node.ids': node_ids,
             'node.type': node_type,
             'node.coordinates': node_coordinates}


def find_nearest_intersection(nodes, zat_point):
    """
    Encuentra el nodo de tipo intersección más cercano a un punto ZAT.

    Args:
        nodes (dict): Diccionario con información de nodos.
        zat_point (tuple): Coordenadas (x, y) del ZAT.

    Returns:
        str: ID del nodo intersección más cercano.
    """
    filter_nodes = FilterableDict(nodes)
    filtered = filter_nodes.filter_by_value('node.type', 'intersection')

    intersection_ids = filtered['node.ids']
    intersection_coords = filtered['node.coordinates']

    min_dist = float('inf')
    nearest_node_id = None

    for node_id, coord in zip(intersection_ids, intersection_coords):
        dist = math.dist(zat_point, coord)
        if dist < min_dist:
            min_dist = dist
            nearest_node_id = node_id

    return str(nearest_node_id)


def generate_network_links_zat(nodes, completeness_percent, seed=None):
    """
    Genera enlaces de tipo 'road' entre intersecciones y enlaces 'logical' entre cada ZAT
    y su intersección más cercana.

    Args:
        nodes (dict): Diccionario con nodos y sus atributos.
        completeness_percent (float): Porcentaje de enlaces posibles entre intersecciones a generar.
        seed (int, opcional): Semilla para reproducibilidad.

    Returns:
        dict: Diccionario con IDs de enlaces, tipos y pares (i, j).
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    filter_nodes = FilterableDict(nodes)

    # Separate node IDs by type
    zat_nodes = filter_nodes.filter_by_values('node.type', 'zat')
    intersection_nodes = filter_nodes.filter_by_value('node.type', 'intersection')
    intersection_nodes_ids = intersection_nodes['node.ids']

    link_ids = []
    link_types = []
    link_ijs = []

    link_counter = 0

    # Generate road links between intersections
    if len(intersection_nodes_ids) > 1:
        possible_road_links = [
            (str(i), str(j)) for i in intersection_nodes_ids for j in intersection_nodes_ids if i != j
        ]
        num_road_links = int(len(possible_road_links) * completeness_percent / 100)
        selected_road_links = random.sample(possible_road_links, num_road_links)
        for u, v in selected_road_links:
            link_ijs.append((u, v))
            link_types.append('road')
            link_ids.append(f"E{link_counter}")
            link_counter += 1

    # Generate logical links: ZAT → nearest intersection (and back)
    if not len(intersection_nodes_ids):
        logging.warning("No intersection nodes found. ZATs cannot be connected.")
        return {
            'link.ids': link_ids,
            'link.type': link_types,
            'link.ij': link_ijs
        }

    for zat_id in zat_nodes['node.ids']:
        zat_coords = filter_nodes.filter_by_values('node.ids', zat_id)['node.coordinates']
        nearest_int_id = find_nearest_intersection(nodes, zat_coords[0])  # Ensure it's in [[x, y]] format

        if nearest_int_id:
            link_out = (str(zat_id), nearest_int_id)
            link_in = (nearest_int_id, str(zat_id))

            # Check and add link from ZAT to intersection
            if link_out not in link_ijs:
                link_ijs.append(link_out)
                link_types.append('logical')
                link_ids.append(f"E{link_counter}")
                link_counter += 1

            # Check and add link from intersection to ZAT
            if link_in not in link_ijs:
                link_ijs.append(link_in)
                link_types.append('logical')
                link_ids.append(f"E{link_counter}")
                link_counter += 1
        else:
            logging.warning(f"ZAT {zat_id} could not find a nearest intersection.")

    return {
        'link.ids': link_ids,
        'link.type': link_types,
        'link.ij': link_ijs
    }

# -------------------------
# OD Matrix and Flow Generation Functions
# -------------------------

def generate_zat_od_matrix(nodes, demand_range, seed=None):
    """
    Crea una matriz OD sintética entre pares ZAT con demanda aleatoria.

    Args:
        nodes (dict): Diccionario con nodos, incluyendo ZATs.
        demand_range (tuple): Tupla (mín, máx) para valores de demanda entre pares.
        seed (int, opcional): Semilla para aleatoriedad.

    Returns:
        list: Lista de tuplas (origen, destino, demanda).
    """
    if demand_range[0] > demand_range[1]:
        raise ValueError("Minimum demand must be less than or equal to maximum demand.")

    if seed is not None:
        np.random.seed(seed)

    filter_nodes = FilterableDict(nodes)
    zat_nodes = filter_nodes.filter_by_value('node.type', 'zat')

    od_matrix = []

    if len(zat_nodes['node.ids']) < 2:
        logging.warning("Not enough ZATs to generate OD pairs.")
        return []

    for origin in zat_nodes['node.ids']:
        for dest in zat_nodes['node.ids']:
            if origin != dest:
                demand = np.random.randint(demand_range[0], demand_range[1] + 1)
                od_matrix.append((str(origin), str(dest), demand))

    return od_matrix


def calculate_node_features_zat(zat_od_matrix, nodes):
    """
    Calcula generación y atracción de demanda para cada nodo ZAT.

    Args:
        zat_od_matrix (list): Lista de tuplas (origen, destino, demanda).
        nodes (dict): Diccionario de nodos.

    Returns:
        dict: Diccionario de nodos con clave adicional 'demand' tipo (gen, attr).
    """
    all_node_ids = list(nodes['node.ids'])
    all_node_types = list(nodes['node.type'])

    # Crear diccionario de demandas solo para ZATs
    gen_attr_dict = {
        node_id: {'gen': 0, 'attr': 0}
        for node_id, node_type in zip(all_node_ids, all_node_types)
        if node_type == 'zat'
    }

    # Acumular generación y atracción
    for origin, dest, demand in zat_od_matrix:
        if origin in gen_attr_dict:
            gen_attr_dict[origin]['gen'] += demand
        if dest in gen_attr_dict:
            gen_attr_dict[dest]['attr'] += demand

    # Generar la lista de demandas alineada con node.ids
    demand_list = []
    for node_id, node_type in zip(all_node_ids, all_node_types):
        if node_type == 'zat':
            gen = gen_attr_dict[node_id]['gen']
            attr = gen_attr_dict[node_id]['attr']
            demand_list.append((gen, attr))
        else:
            demand_list.append((0, 0))

    # Agregar al diccionario
    nodes['node.demand'] = demand_list
    return nodes


def generate_observed_flows(links, coverage_percent, flow_range, seed=None):
    """
    Asigna flujos observados sintéticos a un subconjunto de enlaces tipo 'road'.

    Args:
        links (dict): Diccionario de enlaces con tipo e ID.
        coverage_percent (float): Porcentaje de enlaces tipo 'road' a observar.
        flow_range (tuple): Tupla (mín, máx) para los flujos observados.
        seed (int, opcional): Semilla para reproducibilidad.

    Returns:
        dict: Diccionario actualizado con clave 'observed_flow'.
    """
    if seed is not None:
        random.seed(seed)

    filter_links = FilterableDict(links)
    all_link_ids = list(links['link.ids'])
    all_link_types = list(links['link.type'])

    # Inicializamos todos los flujos como None
    observed_flow_list = [None] * len(all_link_ids)

    # Filtrar links de tipo 'road'
    road_links = filter_links.filter_by_value('link.type', 'road')
    num_road_links = len(road_links['link.ids'])

    if num_road_links == 0:
        links['link.observed_flow'] = observed_flow_list
        return links

    num_to_observe = int(num_road_links * coverage_percent / 100)
    num_to_observe = min(num_to_observe, num_road_links)

    # Obtener índices reales dentro del diccionario original
    road_indices = [i for i, t in enumerate(all_link_types) if t == 'road']
    sampled_indices = random.sample(road_indices, num_to_observe)

    for i in sampled_indices:
        observed_flow_list[i] = random.randint(flow_range[0], flow_range[1])

    # Guardamos en el diccionario original
    links['link.observed_flow'] = observed_flow_list
    return links


# -------------------------
# Traffic Assignment via Linear Programming
# -------------------------

def traffic_assignment_zat(nodes, links, zat_od_matrix):
    """
    Resuelve la asignación de tráfico ZAT-ZAT con programación lineal.
    Minimiza el flujo total respetando conservación de flujo y flujos observados.

    Args:
        nodes (dict): Diccionario con nodos y sus atributos, incluyendo demanda.
        links (dict): Diccionario con enlaces, tipo y flujos observados.
        zat_od_matrix (list): Lista de demandas entre pares ZAT.

    Returns:
        tuple:
            - pd.DataFrame: Resultados con flujo estimado/observado por enlace.
            - dict: Diccionario de enlaces actualizado con 'assigned_flows'.
    """
    # Creamos una copia del diccionario de enlaces para no modificar el original
    links_dict = {k: list(v) for k, v in links.items()}

    # Convertir node_ids a una lista y crear un diccionario de tipos de nodos
    node_ids = nodes['node.ids']
    node_info = {node_id: {'type': node_type} for node_id, node_type in zip(nodes['node.ids'], nodes['node.type'])}

    # Convertir link.ij a formato de tuplas
    link_tuples = [(link[0], link[1]) for link in links['link.ij']]

    # Crear diccionario de tipos de enlaces
    link_info = {(link[0], link[1]): {'type': link_type}
                 for link, link_type in zip(links['link.ij'], links['link.type'])}

    # Crear diccionario de flujos observados
    observed_flows = []
    for (u, v), obs, link_id in zip(links['link.ij'], links['link.observed_flow'], links['link.ids']):
        if obs is not None:
            observed_flows.append((u, v, obs))

    # Crear diccionario de generación/atracción para ZATs basado en la matriz OD
    gen_attr_dict = {}
    for node_id, node_type in zip(nodes['node.ids'], nodes['node.type']):
        if node_type == 'zat':
            gen_attr_dict[node_id] = {'gen': 0, 'attr': 0}

    # Actualizar generación/atracción para cada ZAT basado en la matriz OD
    for origin, dest, demand in zat_od_matrix:
        if origin in gen_attr_dict:
            gen_attr_dict[origin]['gen'] = gen_attr_dict[origin].get('gen', 0) + demand
        if dest in gen_attr_dict:
            gen_attr_dict[dest]['attr'] = gen_attr_dict[dest].get('attr', 0) + demand

    # Validar que los flujos observados correspondan a enlaces definidos
    link_set = set(link_tuples)
    for u, v, _ in observed_flows:
        if (u, v) not in link_set:
            raise ValueError(f"El enlace observado ({u}, {v}) no está en los enlaces definidos de la red.")

    # Definir comodidades como pares ZAT a ZAT con demanda positiva
    zat_od_dict = {(o, d): demand for (o, d, demand) in zat_od_matrix}
    commodities = [(o, d) for (o, d, demand) in zat_od_matrix if demand > 0]

    if not commodities:
        logging.warning("No hay comodidades ZAT a ZAT con demanda > 0. La asignación es trivial.")
        return pd.DataFrame(
            columns=["link_id", "origin", "dest", "link_type", "estimated_flow", "observed_flow"]), links_dict

    # Crear la instancia del problema LP
    prob = pulp.LpProblem("TrafficAssignmentZAT", pulp.LpMinimize)

    # Definir variables LP: flujo para cada comodidad en cada enlace
    flow = pulp.LpVariable.dicts("flow", (commodities, link_tuples), lowBound=0, cat=pulp.LpContinuous)

    logging.info("Añadiendo restricciones de conservación de flujo para cada comodidad y nodo...")
    # Añadir restricciones de conservación de flujo
    for (o, d) in commodities:
        demand = zat_od_dict[(o, d)]
        for k in node_ids:
            # Sumar flujos que salen y entran al nodo k para esta comodidad
            out_flow = pulp.lpSum(flow[(o, d)][(k, v)] for (i, v) in link_tuples if i == k)
            in_flow = pulp.lpSum(flow[(o, d)][(u, k)] for (u, j) in link_tuples if j == k)

            if node_info[k]['type'] == 'intersection':
                # Para intersecciones, el flujo neto debe ser cero
                prob += out_flow - in_flow == 0, f"flow_conservation_int_{o}_{d}_{k}"
            elif node_info[k]['type'] == 'zat':
                if k == o:
                    prob += out_flow - in_flow == demand, f"flow_conservation_origin_{o}_{d}_{k}"
                elif k == d:
                    prob += out_flow - in_flow == -demand, f"flow_conservation_dest_{o}_{d}_{k}"
                else:
                    prob += out_flow - in_flow == 0, f"flow_conservation_transit_{o}_{d}_{k}"

    # Aplicar restricciones en flujos observados (la suma de flujos en un enlace observado iguala al valor observado)
    observed_links_dict = {(u, v): obs for u, v, obs in observed_flows}
    for (u, v), obs_flow in observed_links_dict.items():
        if (u, v) in link_tuples:
            prob += pulp.lpSum(flow[(o, d)][(u, v)] for (o, d) in commodities) == obs_flow, f"observed_flow_{u}_{v}"
        else:
            logging.warning(f"Omitiendo restricción de flujo observado para enlace no definido ({u}, {v}).")

    # Minimizar el flujo total a través de todas las comodidades y enlaces
    prob += pulp.lpSum(flow[(o, d)][(u, v)] for (o, d) in commodities for (u, v) in link_tuples), "Total_Flow"

    # Resolver el problema LP
    logging.info("Resolviendo el problema LP...")
    solver = pulp.PULP_CBC_CMD(msg=True)
    status = prob.solve(solver)

    if pulp.LpStatus[status] != "Optimal":
        logging.error(f"Estado del Solucionador LP: {pulp.LpStatus[status]}")
        try:
            prob.writeLP("traffic_assignment_infeasible.lp")
            logging.error("Modelo LP guardado como 'traffic_assignment_infeasible.lp' para depuración.")
        except Exception as e:
            logging.error(f"Error al guardar el modelo LP: {e}")
        raise RuntimeError(f"No se encontró solución óptima. Estado: {pulp.LpStatus[status]}.")

    logging.info("LP resuelto óptimamente. Procesando resultados...")
    rows = []
    estimated_flows_dict = {}

    # Crear un diccionario para mapear tuples (i,j) a link_ids
    link_id_mapping = {(i, j): link_id for link_id, (i, j) in zip(links['link.ids'], links['link.ij'])}

    for idx, (u, v) in enumerate(link_tuples):
        est_flow = sum(flow[(o, d)][(u, v)].varValue for (o, d) in commodities)
        estimated_flows_dict[(u, v)] = est_flow

        link_id = link_id_mapping.get((u, v), f"Link_{idx}")

        rows.append({
            "link_id": link_id,
            "origin": u,
            "dest": v,
            "link_type": link_info.get((u, v), {}).get('type', 'unknown'),
            "estimated_flow": est_flow,
            "observed_flow": observed_links_dict.get((u, v), None)
        })

    results_df = pd.DataFrame(rows)

    # Agregar los flujos estimados al diccionario de enlaces bajo la clave 'assigned_flows'
    assigned_flows = [estimated_flows_dict.get((i, j), 0) for i, j in links['link.ij']]
    links_dict['link.assigned_flows'] = assigned_flows

    # Verificar el balance de flujo en los nodos y registrar desequilibrios
    logging.info("Verificando balances de flujo en nodos...")
    for k in node_ids:
        generation = gen_attr_dict.get(k, {}).get('gen', 0) if node_info[k]['type'] == 'zat' else 0
        attraction = gen_attr_dict.get(k, {}).get('attr', 0) if node_info[k]['type'] == 'zat' else 0

        total_out = sum(flow[(o, d)][(u, v)].varValue for (o, d) in commodities
                        for (u, v) in link_tuples if u == k)
        total_in = sum(flow[(o, d)][(u, v)].varValue for (o, d) in commodities
                       for (u, v) in link_tuples if v == k)

        balance = total_out - total_in - (generation - attraction)
        if abs(balance) > 1e-5:
            logging.warning(
                f"Desequilibrio en nodo {k}: {balance:.6f} (Salida: {total_out:.2f}, Entrada: {total_in:.2f})")

    return results_df, links_dict


# -------------------------
# Visualization and Data Saving Functions
# -------------------------

def draw_network_zat(nodes, links):
    """
    Genera una visualización interactiva de la red con PyVis,
    incluyendo atributos como tipo de nodo, flujo asignado y observado.

    Args:
        nodes (dict): Información de nodos (ID, tipo, coords, demanda).
        links (dict): Información de enlaces, incluyendo flujos observados y asignados.

    Returns:
        str: Ruta al archivo HTML generado.
    """
    net = Network(directed=True, height="750px", width="100%", notebook=False,
                  bgcolor="white", font_color="white")

    # =====================================================================
    # SECTION: PARÁMETROS DE VISUALIZACIÓN (FÁCIL DE PERSONALIZAR)
    # =====================================================================

    # Parámetros para nodos
    node_params = {
        'zat': {
            'color': "gold",  # Color dorado para ZATs
            'shape': 'circle',  # Forma de diamante
            'size': 15,  # Tamaño más grande
            'font_weight': 'bold',
            'font_size': 16,  # Tamaño de fuente
            'font_color': 'black',
            'border_width': 3,  # Ancho del borde
            'border_color': "black"  # Color del borde
        },
        'intersection': {
            'color': "gray",  # Color azul cielo para intersecciones
            'shape': 'dot',  # Forma circular
            'size': 15,  # Tamaño medio
            'font_size': 12,  # Tamaño de fuente
            'font_color': 'white',
            'font_weight': 'normal',
            'border_width': 1,  # Ancho del borde
            'border_color': "black"  # Color del borde
        }
    }

    # Parámetros para enlaces
    link_params = {
        'road': {
            'color': "black",  # Color gris para carreteras
            'width_base': 1,  # Ancho base
            'width_factor': 1,  # Factor de escala para el ancho
            'dashes': False,  # Sin líneas discontinuas
            'arrows': {'to': {'enabled': True, 'scaleFactor': 0.5}},  # Flechas
            'font_size': 10,  # Tamaño de fuente
            'font_weight': 'normal',
            'font_color': "black",  # Color de fuente
            'font_align': 'middle',  # Alineación de texto
            'smooth': {'enabled': True, 'type': 'dynamic', 'roundness':0.5}  # Curvas suaves
            # "dynamic" "continuous" "discrete" "diagonalCross" "straightCross"
            # "horizontal" "vertical" "curvedCW" "curvedCCW"
        },
        'logical': {
            'color': "lightgray",  # Color salmón para enlaces lógicos
            'width_base': 0.5,  # Ancho base más delgado
            'width_factor': 0.8,  # Factor de escala para el ancho
            'dashes': [5, 5],  # Patrón de línea discontinua
            'arrows': {'to': {'enabled': True, 'scaleFactor': 0.3}},  # Flechas más pequeñas
            'font_size': 8,  # Tamaño de fuente más pequeño
            'font_weight': 'normal',
            'font_color': "black",  # Color de fuente
            'font_align': 'middle',  # Alineación de texto
            'smooth': {'enabled': True, 'type': 'dynamic', 'roundness':0.5}  # Curvas suaves
        }
    }

    # Prefijos para etiquetas
    label_prefixes = {
        'assigned_flow': "T.A.",  # Prefijo para flujo asignado
        'observed_flow': "G.T."  # Prefijo para flujo observado
    }

    # Colores para destacar flujos
    flow_highlight_colors = {
        'assigned_only': "darkslategray",  # Antes era "#4169E1"
        'observed_only': "red",  # Antes era "#32CD32"
        'both_flows': "royalblue"  # Antes era "#FF4500"
    }

    # Formato para valores numéricos
    number_format = {
        'decimal_places': 1,  # Número de decimales
        'thousands_separator': True  # Usar separador de miles
    }

    # =====================================================================
    # FIN DE LA SECCIÓN DE PARÁMETROS
    # =====================================================================

    # Función de ayuda para formatear números
    def format_number(value):
        if value is None:
            return "N/A"
        if number_format['thousands_separator']:
            return f"{value:,.{number_format['decimal_places']}f}"
        return f"{value:.{number_format['decimal_places']}f}"

    # Crear un mapeo de nodos para acceso rápido
    node_ids = nodes['node.ids']
    node_types = nodes['node.type']
    node_demands = nodes['node.demand']

    # Agregar nodos a la red
    for i, node_id in enumerate(node_ids):
        node_type = node_types[i]
        param = node_params[node_type]

        # Crear etiqueta según el tipo de nodo
        if node_type == 'zat':
            demand = node_demands[i]
            gen, attr = demand
            # Solo ID dentro de la forma, atributos como subtítulo
            label = f"{node_id}"
            title = f"ZAT: {node_id}\nGeneración: {gen}\nAtracción: {attr}"
            # Añadir atributos debajo como sub-label
            sub_label = f"G:{gen}, A:{attr}"
        else:
            label = f"{node_id}"
            title = f"Intersección: {node_id}"
            sub_label = ""

        # Añadir nodo a la red con los parámetros correspondientes
        net.add_node(
            node_id,
            label=label,
            title=title,
            color=param['color'],
            shape=param['shape'],
            size=param['size'],
            font={'size': param['font_size'], 'face': 'arial', 'color': param['font_color'], 'weight': param['font_weight']},
            borderWidth=param['border_width'],
            borderWidthSelected=param['border_width'] * 2
        )

        # Para añadir un atributo "value" para mostrar debajo
        if node_type == 'zat':
            net.nodes[len(net.nodes) - 1]['value'] = sub_label

    # Crear mapeos para los enlaces
    link_ids = links['link.ids']
    link_types = links['link.type']
    link_pairs = links['link.ij']
    observed_flows = links['link.observed_flow']
    assigned_flows = links['link.assigned_flows']

    # Añadir enlaces a la red
    for i, link_id in enumerate(link_ids):
        u, v = link_pairs[i]
        link_type = link_types[i]
        param = link_params[link_type]

        assigned_flow = assigned_flows[i]
        observed_flow = observed_flows[i]

        # Determina el color del enlace según los flujos presentes
        if link_type == 'road':
            if observed_flow is not None and assigned_flow > 0:
                edge_color = flow_highlight_colors['both_flows']
            elif observed_flow is not None:
                edge_color = flow_highlight_colors['observed_only']
            elif assigned_flow > 0:
                edge_color = flow_highlight_colors['assigned_only']
            else:
                edge_color = param['color']
        else:
            edge_color = param['color']

        # Crear etiqueta según el tipo de enlace
        label_parts = []
        if link_type == 'road':
            if assigned_flow > 0:
                label_parts.append(f"{label_prefixes['assigned_flow']} {format_number(assigned_flow)}")
            if observed_flow is not None:
                label_parts.append(f"{label_prefixes['observed_flow']} {format_number(observed_flow)}")
        elif link_type == 'logical':
            if assigned_flow > 0:
                label_parts.append(f"{label_prefixes['assigned_flow']} {format_number(assigned_flow)}")

        label_text = "\n".join(label_parts)

        # Crear título informativo para el enlace
        title_parts = [
            f"ID: {link_id}",
            f"De: {u} -> A: {v}",
            f"Tipo: {link_type}",
            f"Flujo Asignado: {format_number(assigned_flow)}"
        ]

        if observed_flow is not None:
            title_parts.append(f"Flujo Observado: {format_number(observed_flow)}")

        title = "\n".join(title_parts)

        # Determinar ancho del enlace basado en el flujo
        # Utilizamos logaritmo para visualizar mejor diferentes magnitudes de flujo
        width = param['width_base']
        if assigned_flow > 0:
            width = param['width_base'] + math.log1p(assigned_flow) * param['width_factor']
            width = min(width, 10)  # Limitar el ancho máximo

        # Añadir enlace a la red
        net.add_edge(
            u, v,
            title=title,
            label=label_text,
            width=width,
            color=edge_color,
            dashes=param['dashes'],
            arrows=param['arrows'],
            font={
                'size': param['font_size'],
                'color': param['font_color'],
                'align': param['font_align'],
                'face': 'arial',
                'weight': param['font_weight']  # Aplicar el grosor de fuente
            },
            smooth=param['smooth']
        )

    # Configurar opciones de física para mejorar la visualización
    net.toggle_physics(True)
    physics_options = {
        # Puedes comentar con # el solver que no quieras usar y descomentar el que quieras probar
        "solver": "forceAtlas2Based",
        # "solver": "barnesHut", # Bueno para redes grandes, más eficiente computacionalmente
        # "solver": "repulsion", # Enfatiza la repulsión entre nodos, útil para redes pequeñas
        # "solver": "hierarchicalRepulsion", # Organiza los nodos en una jerarquía, bueno para estructuras de árbol

        # Opciones para forceAtlas2Based
        "forceAtlas2Based": {
            "gravitationalConstant": -50,  # Fuerza de atracción global (-) o repulsión (+)
            "centralGravity": 0.01,  # Fuerza que atrae los nodos al centro
            "springLength": 200,  # Longitud ideal de los resortes (enlaces)
            "springConstant": 0.05,  # Rigidez de los resortes
            "damping": 0.4,  # Factor de amortiguación (0 a 1)
            "avoidOverlap": 0.8  # Evita solapamiento de nodos (0 a 1)
        },

        # Opciones para barnesHut
        "barnesHut": {
            "gravitationalConstant": -2000,  # Fuerza de atracción global
            "centralGravity": 0.3,  # Fuerza que atrae los nodos al centro
            "springLength": 95,  # Longitud ideal de los resortes (enlaces)
            "springConstant": 0.04,  # Rigidez de los resortes
            "damping": 0.09,  # Factor de amortiguación
            "avoidOverlap": 0.5  # Evita solapamiento de nodos
        },

        # Opciones para repulsion
        "repulsion": {
            "nodeDistance": 120,  # Distancia mínima entre nodos
            "centralGravity": 0.2,  # Fuerza que atrae los nodos al centro
            "springLength": 200,  # Longitud ideal de los resortes
            "springConstant": 0.05,  # Rigidez de los resortes
            "damping": 0.09  # Factor de amortiguación
        },

        # Opciones para hierarchicalRepulsion
        "hierarchicalRepulsion": {
            "nodeDistance": 120,  # Distancia mínima entre nodos
            "centralGravity": 0.0,  # Fuerza que atrae los nodos al centro
            "springLength": 100,  # Longitud ideal de los resortes
            "springConstant": 0.01,  # Rigidez de los resortes
            "damping": 0.09,  # Factor de amortiguación
            "avoidOverlap": 1.0  # Evita solapamiento de nodos
        },

        # Opciones de estabilización
        "stabilization": {
            "enabled": True,  # Activar/desactivar estabilización inicial
            "iterations": 1000,  # Número de iteraciones
            "updateInterval": 50,  # Frecuencia de actualización visual durante estabilización
            "onlyDynamicEdges": False,  # Solo estabilizar enlaces dinámicos
            "fit": True  # Ajustar la visualización después de la estabilización
        },

        # Opciones para la interacción del usuario
        "minVelocity": 0.75,  # Velocidad mínima para considerar que la red está estabilizada
        "maxVelocity": 30,  # Velocidad máxima de los nodos durante la simulación
        "timestep": 0.5,  # Tamaño del paso de tiempo en la simulación
        "adaptiveTimestep": True  # Ajustar el timestep automáticamente
    }

    net.set_options(json.dumps({"physics": physics_options}))

    # Guardar la visualización como un archivo HTML
    output_file = "outputs/generated_synthetic_network.html"
    net.write_html(output_file)
    logging.info(f"Visualización de la red guardada en {output_file}")

    return output_file

def save_results(config, nodes, links, zat_od_matrix, assignment_results,
                 generated_synthetic_network, pickle_file, log_file):
    """
    Guarda todos los resultados de la simulación en un archivo pickle
    y opcionalmente registra el resumen en un log JSONL.

    Args:
        config (dict): Configuración de parámetros usados.
        nodes (dict): Nodos de la red.
        links (dict): Enlaces de la red.
        zat_od_matrix (list): Matriz OD entre ZATs.
        assignment_results (pd.DataFrame): Resultados del modelo de asignación.
        generated_synthetic_network (str): Ruta al HTML con la red visualizada.
        pickle_file (str): Ruta para guardar el archivo pickle.
        log_file (str): Ruta para guardar el log en formato JSONL.
    """

    data_dict = {
        'config': config,
        'nodes': nodes,
        'links': links,
        'zat_od_matrix': zat_od_matrix,
        'assignment_results': assignment_results,
        'generated_synthetic_network': generated_synthetic_network
    }

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(pickle_file), exist_ok=True)
    with open(pickle_file, 'wb') as f:
        pickle.dump(data_dict, f)
    logging.info(f"Results saved to {pickle_file}")

    # Write run summary to log file (JSONL format)
    """record = {
        "timestamp": datetime.now().isoformat(),
        "status": "Success",
        "config": config,
        "num_nodes": len(nodes['node.ids']),
        "num_links": len(links['link.ids']),
        "num_zat_od_pairs": len(zat_od_matrix),
    }
    with open(log_file, "a", encoding="utf-8") as logfile:
        logfile.write(json.dumps(record, ensure_ascii=False) + "\n")
    logging.info(f"Run summary appended to {log_file}")"""


# -------------------------
# Main Execution Block
# -------------------------

if __name__ == "__main__":
    # Configuration parameters
    config = {
        'NUM_ZATS': 10,
        'NUM_INTERSECTIONS': 20,
        'ZAT_DEMAND_RANGE': (0, 500),
        'INTERSECTION_COMPLETENESS': 100,  # Percentage of links between intersections
        'OBSERVATION_COVERAGE': 30,  # Percentage of road links to observe
        'OBSERVED_FLOW_RANGE': (0, 200),
        'SEED_NETWORK': None,
        'SEED_DEMAND': None,
        'SEED_OBSERVED': None,
        'MAX_TRIES': 1000,
        'NAME_PICKLE_FILE': f'../data/traffic_data_big.pkl',
        'LOG_FILE': "successful_runs_zat_log.jsonl"
    }

    # Set overall seed
    random.seed(42)
    np.random.seed(42)

    for attempt in range(1, config['MAX_TRIES'] + 1):
        # Generate new seeds for variability if needed
        if config['SEED_NETWORK'] is None:
            config['SEED_NETWORK'] = random.randint(0, 99999)
        if config['SEED_DEMAND'] is None:
            config['SEED_DEMAND'] = random.randint(0, 99999)
        if config['SEED_OBSERVED'] is None:
            config['SEED_OBSERVED'] = random.randint(0, 99999)

        logging.info(f"--- Attempt {attempt}/{config['MAX_TRIES']} ---")
        logging.info(
            f"SEED_NETWORK={config['SEED_NETWORK']}, SEED_DEMAND={config['SEED_DEMAND']}, SEED_OBSERVED={config['SEED_OBSERVED']}")

        try:
            # Step 1: Generate nodes (ZATs and intersections)
            nodes = generate_nodes(config['NUM_ZATS'], config['NUM_INTERSECTIONS'])

            # Step 2: Generate links (road links + logical connections)
            links = generate_network_links_zat(nodes, config['INTERSECTION_COMPLETENESS'],
                                                                        config['SEED_NETWORK'])
            if not links:
                raise ValueError("No links were generated. Please check network parameters.")

            # Step 3: Generate ZAT-to-ZAT OD matrix
            zat_od_matrix = generate_zat_od_matrix(nodes, config['ZAT_DEMAND_RANGE'], config['SEED_DEMAND'])
            if not zat_od_matrix and config['NUM_ZATS'] >= 2:
                logging.warning("Empty ZAT OD matrix generated; assignment might be trivial.")

            # Step 4: Calculate node features (Generation and Attraction)
            nodes = calculate_node_features_zat(zat_od_matrix, nodes)

            # Step 5: Generate observed flows (on road links only)
            links = generate_observed_flows(links, config['OBSERVATION_COVERAGE'],
                                                     config['OBSERVED_FLOW_RANGE'], config['SEED_OBSERVED'])

            #logging.info(f"Generated {len(observed_flows)} observed flows on road links.")

            # Step 6: Run traffic assignment using LP
            assignment_results, links = traffic_assignment_zat(nodes, links, zat_od_matrix)

            # Step 7: Create network visualization
            generated_synthetic_network = draw_network_zat(nodes, links)

            # Step 8: Save data and run summary
            save_results(config, nodes, links, zat_od_matrix, assignment_results,
                         generated_synthetic_network,
                         pickle_file=config['NAME_PICKLE_FILE'],
                         log_file=config['LOG_FILE'])

            logging.info(f"✅ Success on attempt {attempt}.")
            break  # Exit loop on success

        except (RuntimeError, ValueError, pulp.PulpError) as e:
            logging.warning(f"❌ Failed attempt {attempt}: {str(e)}")
            failure_record = {
                "timestamp": datetime.now().isoformat(),
                "attempt": attempt,
                "status": "Failure",
                "SEED_NETWORK": config['SEED_NETWORK'],
                "SEED_DEMAND": config['SEED_DEMAND'],
                "SEED_OBSERVED": config['SEED_OBSERVED'],
                "error_message": str(e)
            }
            with open(config['LOG_FILE'], "a", encoding="utf-8") as logfile:
                logfile.write(json.dumps(failure_record, ensure_ascii=False) + "\n")

            if attempt == config['MAX_TRIES']:
                logging.error("🔴 Maximum attempts reached. Could not find a feasible solution.")
            else:
                logging.info("Retrying with new seeds...")
                # Reset seeds for next attempt
                config['SEED_NETWORK'] = None
                config['SEED_DEMAND'] = None
                config['SEED_OBSERVED'] = None
