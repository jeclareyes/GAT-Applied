import pulp
import numpy as np
import pandas as pd
from pyvis.network import Network
import random
from tqdm import tqdm
import pickle
import logging
from utils.logger_config import setup_logger
import json
from datetime import datetime

#%%

# Function to generate OD matrix
def generate_od_matrix(num_nodes, demand_range, seed=None):
    """
    Generates a random origin-destination matrix.
    :param num_nodes: number of nodes in the network
    :param demand_range: tuple (min, max) for demand
    :param seed: seed for reproducibility
    :return: list of tuples (origin, destination, demand)
    """
    if seed is not None:
        np.random.seed(seed)

    od_matrix = []
    nodes = list(range(num_nodes))
    for o in nodes:
        for d in nodes:
            if o != d:
                demand = np.random.randint(demand_range[0], demand_range[1] + 1)
                od_matrix.append((o, d, demand))
    return od_matrix


# Function to generate network links
def generate_network_links(num_nodes, completeness_percent, seed=None):
    """
    Generates random network links.
    :param num_nodes: number of nodes
    :param completeness_percent: percentage of completeness (0-100)
    :param seed: seed for reproducibility
    :return: list of tuples (origin, destination)
    """
    if seed is not None:
        random.seed(seed)

    all_possible_links = [(i, j) for i in range(num_nodes) for j in range(num_nodes) if i != j]
    num_links = int(len(all_possible_links) * completeness_percent / 100)
    return random.sample(all_possible_links, num_links)


# Function to generate observed flows
def generate_observed_flows(links, coverage_percent, flow_range, seed=None):
    """
    Generates synthetic observed flows.
    :param links: list of available links
    :param coverage_percent: percentage of links to observe
    :param flow_range: tuple (min, max) for flows
    :param seed: seed for reproducibility
    :return: list of tuples (origin, destination, flow)
    """
    if seed is not None:
        random.seed(seed)

    num_observed = int(len(links) * coverage_percent / 100)
    observed_links = random.sample(links, num_observed)
    return [(u, v, random.randint(flow_range[0], flow_range[1])) for (u, v) in observed_links]


# Modified main function
def traffic_assignment_visual(nodes, links, od_matrix, observed_flows, node_features):
    """
    Traffic assignment in a network with visualization, with:
      - Objetivo de minimizar la suma total de flujos.
      - Verificación explícita del equilibrio a nivel de nodo.
    """
    # Input validation: verificar que los flujos observados correspondan a enlaces existentes
    link_set = set(links)
    for u, v, _ in observed_flows:
        if (u, v) not in link_set:
            raise ValueError(f"Observed link ({u}, {v}) is not in the network.")

    # Procesamiento de la matriz OD
    commodities = [(o, d) for (o, d, demand) in od_matrix if demand > 0]
    od_dict = {(o, d): demand for (o, d, demand) in od_matrix}

    # Definir el modelo de optimización
    prob = pulp.LpProblem("TrafficAssignment", pulp.LpMinimize)
    flow = pulp.LpVariable.dicts("flow", (commodities, links), lowBound=0, cat=pulp.LpInteger)

    # Restricciones de conservación de flujo (por commodity)
    for (o, d) in tqdm(commodities, desc="Adding flow conservation constraints"):
        demand = od_dict[(o, d)]
        for k in nodes:
            out_flow = pulp.lpSum(flow[(o, d)][(k, v)] for (k2, v) in links if k2 == k)
            in_flow = pulp.lpSum(flow[(o, d)][(u, k)] for (u, k2) in links if k2 == k)

            if k == o:
                prob += out_flow - in_flow == demand, f"flow_conservation_origin_{o}_{d}_{k}"
            elif k == d:
                prob += out_flow - in_flow == -demand, f"flow_conservation_dest_{o}_{d}_{k}"
            else:
                prob += out_flow - in_flow == 0, f"flow_conservation_node_{o}_{d}_{k}"

    # Restricciones para flujos observados: forzar que la suma de flujos asignados en cada enlace observado sea igual al flujo observado.
    for (u, v, obs) in observed_flows:
        prob += pulp.lpSum(flow[(o, d)][(u, v)] for (o, d) in commodities) == obs

    # Función objetivo: minimizar la suma total de flujos en todos los enlaces y commodities
    prob += pulp.lpSum(flow[(o, d)][(u, v)] for (o, d) in commodities for (u, v) in links)

    # Resolver el LP
    status = prob.solve(pulp.PULP_CBC_CMD(msg=False))
    if pulp.LpStatus[status] != "Optimal":
        raise RuntimeError("No feasible solution found. Possible causes:\n"
                           "1. Observed flows are inconsistent with OD demand\n"
                           "2. Disconnected network for required OD pairs")

    # Preparar los resultados
    obs_dict = {(u, v): f for u, v, f in observed_flows}
    rows = []
    flow_dict = {}
    for (u, v) in links:
        est = sum(flow[(o, d)][(u, v)].varValue for (o, d) in commodities)
        flow_dict[(u, v)] = est
        rows.append({
            "origin": u,
            "dest": v,
            "estimated_flow": est,
            "observed_flow": obs_dict.get((u, v), None)
        })
    df = pd.DataFrame(rows)

    # Verificación explícita del equilibrio a nivel de nodo:
    node_balance = {}
    for k in nodes:
        # Generación: suma de demandas salientes (se considera con signo negativo)
        generation = sum(demand for (o, d, demand) in od_matrix if o == k)
        # Atracción: suma de demandas entrantes (con signo positivo)
        attraction = sum(demand for (o, d, demand) in od_matrix if d == k)
        # Flujo saliente e entrante (acumulados sobre todos los commodities)
        outgoing = sum(flow[(o, d)][(u, v)].varValue for (o, d) in commodities for (u, v) in links if u == k)
        incoming = sum(flow[(o, d)][(u, v)].varValue for (o, d) in commodities for (u, v) in links if v == k)
        # Balance: (- generación) + atracción + flujo entrante - flujo saliente
        balance = (-generation) + attraction + incoming - outgoing
        node_balance[k] = balance
        if abs(balance) > 1e-6:
            logging.warning(f"Nodo {k} no está en equilibrio. Balance: {balance}")

    # Se puede imprimir o registrar el balance de cada nodo si se requiere.
    for k, bal in node_balance.items():
        print(f"Nodo {k}: Balance = {bal}")

    # Visualización: se pasa explícitamente node_features
    draw_network(nodes, links, flow_dict, obs_dict, node_features)

    return df


# Visualization function (unchanged)
def draw_network(nodes, links, estimated_flows, observed_flows, node_features):
    net = Network(directed=True, height="600px", width="100%", notebook=False)

    for node in nodes:
        net.add_node(node, label=str(str(node)+"\n"+str(node_features[node])))
    for (u, v) in links:
        est = estimated_flows.get((u, v), 0)
        obs = observed_flows.get((u, v), None)
        label = f"{est:.1f}"
        if obs is not None:
            label += f" / {obs} (obs)"
        width = 1 + est / 10
        color = "#FFA500" if (u, v) in observed_flows else "#999"
        net.add_edge(u, v, title=label, label=label, width=width, color=color)
    net.write_html("traffic_network.html")


def calculate_node_features(od_matrix, num_nodes):
    """Calculates demand generated and attracted per node"""
    generated = np.zeros(num_nodes, dtype=int)
    attracted = np.zeros(num_nodes, dtype=int)

    for o, d, demand in od_matrix:
        generated[o] += demand
        attracted[d] += demand

    return np.column_stack((generated, attracted))


# Example usage with automatic generation
if __name__ == "__main__":
        logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
        setup_logger()

        # Configuration
        NUM_NODES = 3
        DEMAND_RANGE = (1, 5)
        COMPLETENESS = 50
        COVERAGE = 50
        FLOW_RANGE = (1, 10)
        MAX_TRIES = 1000
        NAME_PICKLE_FILE = '../data/traffic_data_5.pkl'

        for attempt in range(1, MAX_TRIES + 1):
            SEED_NETWORK = random.randint(0, 99999)
            SEED_OBSERVED = random.randint(0, 99999)
            logging.info(f"Iteración {attempt} - SEED_NETWORK={SEED_NETWORK}, SEED_OBSERVED={SEED_OBSERVED}")

            try:
                nodes = list(range(NUM_NODES))
                links = generate_network_links(NUM_NODES, COMPLETENESS, SEED_NETWORK)
                od_matrix = generate_od_matrix(NUM_NODES, DEMAND_RANGE, SEED_NETWORK)
                observed_flows = generate_observed_flows(links, COVERAGE, FLOW_RANGE, SEED_OBSERVED)
                node_features = calculate_node_features(od_matrix, NUM_NODES)

                df_result = traffic_assignment_visual(nodes, links, od_matrix, observed_flows, node_features)

                data_dict = {
                    'config': {
                        'num_nodes': NUM_NODES,
                        'demand_range': DEMAND_RANGE,
                        'completeness': COMPLETENESS,
                        'coverage': COVERAGE,
                        'flow_range': FLOW_RANGE,
                        'seed_network': SEED_NETWORK,
                        'seed_observed': SEED_OBSERVED
                    },
                    'nodes_features': node_features,
                    'links_topology': np.array(links, dtype=int),
                    'od_matrix': np.array([(o, d, dem) for o, d, dem in od_matrix], dtype=int),
                    'observed_flows': np.array(observed_flows, dtype=int),
                    'results_df': df_result,
                    'node_ids': np.array(nodes, dtype=int),
                    'link_ids': np.array([(u, v) for u, v in links], dtype=object)
                }

                # Guardar pickle
                with open(NAME_PICKLE_FILE, 'wb') as f:
                    pickle.dump(data_dict, f)
                print(f"Saved in {NAME_PICKLE_FILE}")

                # Registro acumulativo con timestamp
                record = {
                    "timestamp": datetime.now().isoformat(),
                    "seed_network": SEED_NETWORK,
                    "seed_observed": SEED_OBSERVED,
                    "config": data_dict["config"],
                    "num_links": len(data_dict["links_topology"]),
                    "num_od_pairs": len(data_dict["od_matrix"]),
                    "num_observed_links": len(data_dict["observed_flows"]),
                    "estimated_total_flow": float(df_result["estimated_flow"].sum()),
                    "observed_total_flow": float(df_result["observed_flow"].dropna().sum()),
                    "nodes": nodes,
                    "links": links,
                    "observed_flows": observed_flows,
                    "node_features": node_features.tolist(),
                    "od_matrix": od_matrix

                }

                with open("successful_runs_log.jsonl", "a", encoding="utf-8") as logfile:
                    logfile.write(json.dumps(record, ensure_ascii=False) + "\n")


                logging.info("✅ Solución encontrada y guardada.")
                break  # Salimos si tuvo éxito

            except RuntimeError as e:
                logging.warning(f"❌ Fallo en la iteración {attempt}: {str(e)}")

        else:
            logging.error("🔴 No se encontró una solución factible tras todos los intentos.")