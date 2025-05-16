#!/usr/bin/env python3
"""
optimization.py

Módulo para el entrenamiento y optimización del modelo GNN para asignación de tráfico.
Modificado para manejar G/A aprendibles para nodos AUX en custom_loss_original.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from config import Various, TrainingConfig # Para acceder a los tipos de nodo
import networkx as nx

# Configuración de logging
logger = logging.getLogger(__name__)

class UserEquilibriumLoss(nn.Module):
    def __init__(self, config_various: Various, training_config: TrainingConfig, k_routes_for_ue: int = 3):
        super(UserEquilibriumLoss, self).__init__()
        self.config_various = config_various
        self.training_config = training_config
        self.k_routes_for_ue = k_routes_for_ue
        self.eps = 1e-8
        logger.info(f"UserEquilibriumLoss inicializada con pesos: Cons={training_config.w_conservation}, DemSat={training_config.w_demand_satisfaction}, UE_Wardrop={training_config.w_ue_wardrop}")
        # Grafo NetworkX para búsqueda de rutas (se construye una vez si la topología es fija)
        # Se podría pasar `data` aquí para construirlo, o construirlo en el primer forward pass.
        self._graph_nx = None 
        self._graph_num_nodes = 0
        self._graph_edge_index_hash = None


    def _build_nx_graph(self, data, link_costs_for_paths):
        """
        Construye o actualiza un grafo NetworkX a partir de data.edge_index y link_costs.
        
        :param data: objeto tipo torch_geometric.data.Data, con tensores
        :param link_costs_for_paths: tensor 1D con los costos de los enlaces, en el mismo orden que data.edge_index
        """
        import networkx as nx
        import logging
        logger = logging.getLogger(__name__)

        # Convertir edge_index a bytes de forma segura
        edge_index_bytes = data.edge_index.cpu().numpy().tobytes()
        current_edge_index_hash = hash(edge_index_bytes)

        # Reconstruir si el grafo no existe, si cambió el número de nodos, o si cambió la topología
        if (self._graph_nx is None or
            self._graph_num_nodes != data.num_nodes or
            self._graph_edge_index_hash != current_edge_index_hash):

            self._graph_nx = nx.DiGraph()
            self._graph_nx.add_nodes_from(range(data.num_nodes))

            # Convertir enlaces y costos a formato NetworkX
            edges_with_costs = [
                (u, v, {'weight': cost.item()})
                for (u, v), cost in zip(data.edge_index.t().tolist(), link_costs_for_paths)
            ]
            self._graph_nx.add_edges_from(edges_with_costs)

            self._graph_num_nodes = data.num_nodes
            self._graph_edge_index_hash = current_edge_index_hash
            logger.debug("Grafo NetworkX para rutas reconstruido.")
        else:
            # Solo actualizar pesos si la topología no ha cambiado
            for i, (u, v) in enumerate(data.edge_index.t().tolist()):
                if self._graph_nx.has_edge(u, v):
                    self._graph_nx[u][v]['weight'] = link_costs_for_paths[i].item()
                else:
                    # Este caso no debería ocurrir si el grafo es consistente
                    self._graph_nx.add_edge(u, v, weight=link_costs_for_paths[i].item())
            logger.debug("Pesos del grafo NetworkX actualizados.")



    def _calculate_flow_conservation_loss(self, predicted_flows, data):
        # ... (exactamente como en la respuesta anterior) ...
        loss_conservation = torch.tensor(0.0, device=predicted_flows.device)
        count_intersections = 0
        intersection_types_lower = [it.lower() for it in self.config_various.intersection_node_types]
        for node_idx in range(data.num_nodes):
            if str(data.node_types[node_idx]).lower() in intersection_types_lower:
                in_idx = data.in_edges_idx_tonode[node_idx].to(predicted_flows.device)
                out_idx = data.out_edges_idx_tonode[node_idx].to(predicted_flows.device)
                flow_in = predicted_flows[in_idx].sum() if in_idx.numel() > 0 else torch.tensor(0.0, device=predicted_flows.device)
                flow_out = predicted_flows[out_idx].sum() if out_idx.numel() > 0 else torch.tensor(0.0, device=predicted_flows.device)
                imbalance = flow_in - flow_out
                current_loss_term = imbalance ** 2
                if self.training_config.normalize_losses:
                    denominator = (flow_in.abs() + flow_out.abs())**2 + self.eps
                    loss_conservation += current_loss_term / denominator if denominator > self.eps else current_loss_term
                else:
                    loss_conservation += current_loss_term
                count_intersections += 1
        return loss_conservation / (count_intersections + self.eps) if count_intersections > 0 else torch.tensor(0.0, device=predicted_flows.device)

    def _calculate_demand_satisfaction_loss(self, predicted_flows, data):
        # Modificado para NO incluir nodos AUX en esta componente de pérdida.
        loss_demand_taz = torch.tensor(0.0, device=predicted_flows.device)
        count_taz_nodes = 0
        taz_types_lower = [t.lower() for t in self.config_various.taz_node_types]

        for node_idx in range(data.num_nodes):
            node_type_lower = str(data.node_types[node_idx]).lower()
            if node_type_lower in taz_types_lower:
                node_id_original = data.node_id_map_rev.get(node_idx)
                gen_val, attr_val = 0.0, 0.0
                if node_id_original and node_id_original in data.zat_demands:
                    gen_val, attr_val = data.zat_demands[node_id_original]
                else:
                    logger.debug(f"TAZ node {node_id_original} (idx {node_idx}) sin demanda en data.zat_demands. G/A=0.")

                gen_tensor = torch.tensor(gen_val, device=predicted_flows.device, dtype=predicted_flows.dtype)
                attr_tensor = torch.tensor(attr_val, device=predicted_flows.device, dtype=predicted_flows.dtype)

                in_idx = data.in_edges_idx_tonode[node_idx].to(predicted_flows.device)
                out_idx = data.out_edges_idx_tonode[node_idx].to(predicted_flows.device)

                flow_in = predicted_flows[in_idx].sum() if in_idx.numel() > 0 else torch.tensor(0.0, device=predicted_flows.device)
                flow_out = predicted_flows[out_idx].sum() if out_idx.numel() > 0 else torch.tensor(0.0, device=predicted_flows.device)
                
                loss_gen = (flow_out - gen_tensor)**2
                loss_attr = (flow_in - attr_tensor)**2
                current_demand_loss_term = loss_gen + loss_attr

                if self.training_config.normalize_losses:
                    den_gen = gen_tensor**2 + self.eps
                    den_attr = attr_tensor**2 + self.eps
                    norm_loss_gen = loss_gen / den_gen if den_gen > self.eps else loss_gen
                    norm_loss_attr = loss_attr / den_attr if den_attr > self.eps else loss_attr
                    loss_demand_taz += (norm_loss_gen + norm_loss_attr)
                else:
                    loss_demand_taz += current_demand_loss_term
                count_taz_nodes += 1
        
        return loss_demand_taz / (count_taz_nodes + self.eps) if count_taz_nodes > 0 else torch.tensor(0.0, device=predicted_flows.device)

    def _calculate_bpr_times_all_links(self, predicted_flows, data):
        """
        Calcula los tiempos de viaje BPR para todos los enlaces.
        Para enlaces no-road, el tiempo es un valor muy bajo (o su free_flow_time si está definido como tal).
        data.vdf_tensor debe tener [capacity, free_flow_time, alpha, beta, length]
        """
        num_links = predicted_flows.shape[0]
        link_travel_times = torch.zeros(num_links, device=predicted_flows.device)

        # Obtener parámetros BPR. data.vdf_tensor tiene NaNs para no-road links.
        # Columnas: 0:capacity, 1:free_flow_time, 2:alpha, 3:beta, 4:length
        bpr_params = data.vdf_tensor.to(predicted_flows.device)

        # Máscara para enlaces 'road' (donde los parámetros BPR no son NaN)
        # Chequeamos la primera columna (capacidad) como proxy.
        is_road_link_mask = ~torch.isnan(bpr_params[:, 0])
        
        road_indices = is_road_link_mask.nonzero(as_tuple=True)[0]
        non_road_indices = (~is_road_link_mask).nonzero(as_tuple=True)[0]

        if road_indices.numel() > 0:
            flows_road = predicted_flows[road_indices]
            cap_road = bpr_params[road_indices, 0]
            fft_road = bpr_params[road_indices, 1]
            alpha_road = bpr_params[road_indices, 2]
            beta_road = bpr_params[road_indices, 3]
            
            # Asegurar capacidades y fft positivos para evitar NaN/inf
            cap_road_safe = torch.clamp(cap_road, min=self.eps)
            fft_road_safe = torch.clamp(fft_road, min=self.eps) # fft debe ser > 0
            alpha_road_safe = torch.clamp(alpha_road, min=0) # alpha >=0
            beta_road_safe = torch.clamp(beta_road, min=0)   # beta >=0

            link_travel_times[road_indices] = fft_road_safe * \
                (1.0 + alpha_road_safe * (F.relu(flows_road) / cap_road_safe)**beta_road_safe)

        # Para enlaces no-road, asignar un costo muy bajo (ej. free_flow_time si está, o un epsilon)
        # Si vdf_tensor tiene fft para no-road (ej. representando tiempo de acceso), usarlo.
        # Si no, un epsilon pequeño.
        if non_road_indices.numel() > 0:
            # Si los no-road tienen un free_flow_time definido (col 1 no NaN)
            fft_non_road_defined_mask = ~torch.isnan(bpr_params[non_road_indices, 1])
            indices_with_fft = non_road_indices[fft_non_road_defined_mask]
            indices_without_fft = non_road_indices[~fft_non_road_defined_mask]

            if indices_with_fft.numel() > 0:
                 link_travel_times[indices_with_fft] = torch.clamp(bpr_params[indices_with_fft, 1], min=self.eps)
            if indices_without_fft.numel() > 0:
                 link_travel_times[indices_without_fft] = self.eps # Costo simbólico muy bajo
        
        # Asegurar que no haya NaNs o Infs resultantes (ej. si fft fue 0)
        link_travel_times = torch.where(torch.isnan(link_travel_times) | torch.isinf(link_travel_times), 
                                        torch.tensor(1e5, device=link_travel_times.device), # Tiempo muy alto para errores
                                        link_travel_times)
        return link_travel_times

    def _calculate_ue_loss_k_fixed_routes(self, predicted_flows, data, link_travel_times):
            """
            Calcula la pérdida UE basada en K rutas predefinidas (por longitud) para cada par O-D.
            Penaliza la varianza de los costos BPR de estas K rutas.
            Requiere: data.k_shortest_paths_by_length_link_indices = {(o,d): [[link_indices_ruta1], ...]}
                    data.od_pairs = [(o, d, demand_value), ...]
            """
            loss_ue_variance = torch.tensor(0.0, device=predicted_flows.device)
            num_od_pairs_processed = 0

            if not hasattr(data, 'od_pairs') or data.od_pairs is None:
                logger.debug("No data.od_pairs found. Skipping UE loss (k-fixed-routes).")
                return loss_ue_variance

            for orig_node_idx, dest_node_idx, od_demand_value in data.od_pairs:
                if od_demand_value <= self.eps:
                    continue

                # Recuperar las K rutas predefinidas (listas de índices de arcos)
                fixed_routes_for_od = data.k_shortest_paths_by_length_link_indices.get((orig_node_idx, dest_node_idx))

                if not fixed_routes_for_od or not fixed_routes_for_od[0]: # Si no hay rutas o la primera está vacía
                    # logger.debug(f"No hay rutas predefinidas para O-D: {orig_node_idx}->{dest_node_idx}")
                    # Podría penalizarse si se espera que siempre haya rutas
                    loss_ue_variance += od_demand_value * 1e3 # Penalización moderada
                    num_od_pairs_processed +=1
                    continue

                route_costs_bpr = []
                for route_link_indices in fixed_routes_for_od:
                    if not route_link_indices: # Si una ruta específica está vacía
                        route_costs_bpr.append(torch.tensor(1e5, device=predicted_flows.device)) # Costo alto para ruta inválida/no encontrada
                        continue
                    
                    # Asegurar que los índices de los arcos sean tensores para indexación avanzada
                    current_route_link_indices_tensor = torch.tensor(route_link_indices, dtype=torch.long, device=link_travel_times.device)
                    
                    # Sumar los costos BPR de los arcos en esta ruta
                    # Es crucial que link_travel_times ya esté calculado con los flujos predichos actuales
                    cost_of_current_route = link_travel_times[current_route_link_indices_tensor].sum()
                    route_costs_bpr.append(cost_of_current_route)
                
                if len(route_costs_bpr) > 1:
                    # Penalizar la varianza de los costos de estas K rutas
                    # Normalizar la varianza por el costo medio al cuadrado podría ser una opción
                    costs_tensor = torch.stack(route_costs_bpr)
                    mean_cost = costs_tensor.mean()
                    # Varianza relativa para evitar que explote si los costos son muy altos
                    variance_penalty = torch.var(costs_tensor) / (mean_cost**2 + self.eps) if mean_cost > self.eps else torch.var(costs_tensor)
                    loss_ue_variance += od_demand_value * variance_penalty
                elif len(route_costs_bpr) == 1:
                    # Si solo hay una ruta, no hay varianza. Podríamos penalizar su costo absoluto * demanda.
                    # Esto es similar a la pérdida UE de ruta más corta si K=1.
                    loss_ue_variance += od_demand_value * route_costs_bpr[0] * 0.01 # Pequeña penalización por costo
                                                                                # o simplemente 0 si no queremos penalizar esto.
                
                num_od_pairs_processed += 1
                
            return loss_ue_variance / (num_od_pairs_processed + self.eps) if num_od_pairs_processed > 0 else torch.tensor(0.0, device=predicted_flows.device)

    def _calculate_ue_wardrop_loss(self, predicted_flows, data, link_travel_times):
        """
        Cálculo de la pérdida de UE.
        Itera sobre data.od_pairs o od_tensor, encuentra la ruta más corta y calcula la pérdida.
        """
        loss_ue = torch.tensor(0.0, device=predicted_flows.device)
        num_od_pairs_processed = 0

        # TODO implementar que sea data.od_pairs o data.od_tensor

        if not hasattr(data, 'od_pairs') or data.od_tensor is None:
            logger.debug("No OD pairs found in data. Skipping UE Wardrop loss.")
            return loss_ue

        # Construir/Actualizar el grafo NetworkX con los costos actuales
        self._build_nx_graph(data, link_travel_times)

        for row in data.od_tensor:
            orig_idx, dest_idx, demand = row[0], row[1], row[2]
            if demand <= self.eps: # Ignorar pares O-D sin demanda significativa
                continue
            
            try:
                # Encontrar la ruta más corta usando los link_travel_times actuales como pesos
                # El grafo _graph_nx ya tiene los pesos actualizados
                shortest_path_nodes = nx.shortest_path(self._graph_nx, source=orig_idx, target=dest_idx, weight='weight')
                
                # Calcular el costo de esta ruta más corta
                cost_shortest_path = torch.tensor(0.0, device=predicted_flows.device)
                for i in range(len(shortest_path_nodes) - 1):
                    u, v = shortest_path_nodes[i], shortest_path_nodes[i+1]
                    # Encontrar el índice del enlace (u,v) en data.edge_index para obtener su costo de link_travel_times
                    # Esto puede ser ineficiente si se hace muchas veces.
                    # Una mejor forma sería tener un mapeo o que _graph_nx almacene el índice original del enlace.
                    # Por ahora, una búsqueda simple (asumiendo que los costos en el grafo son los correctos):
                    cost_shortest_path += self._graph_nx[u][v]['weight']
                
                # La pérdida es la demanda multiplicada por el costo de la ruta más corta
                # Esto incentiva al modelo a encontrar flujos que resulten en costos bajos para las rutas usadas.
                loss_ue += demand * cost_shortest_path 
                num_od_pairs_processed += 1

            except nx.NetworkXNoPath:
                # logger.debug(f"No path found between OD pair: {orig_idx} -> {dest_idx}. Skipping for UE loss.")
                # Podría añadirse una penalización alta si se espera conectividad
                loss_ue += demand * 1e5 # Penalización alta si no hay ruta
                num_od_pairs_processed +=1
            except Exception as e:
                logger.error(f"Error en shortest_path para OD {orig_idx}-{dest_idx}: {e}")
                loss_ue += demand * 1e5 
                num_od_pairs_processed +=1


        return loss_ue / (num_od_pairs_processed + self.eps) if num_od_pairs_processed > 0 else torch.tensor(0.0, device=predicted_flows.device)


    def forward(self, predicted_flows, data):
        # 1. Pérdida por Conservación de Flujo
        loss_cons = self._calculate_flow_conservation_loss(predicted_flows, data)
        
        # 2. Pérdida por Satisfacción de Demanda (solo TAZ)
        loss_demand_sat_total_ga = self._calculate_demand_satisfaction_loss(predicted_flows, data)
        
        # 3. Pérdida de Equilibrio de Usuario (UE) de Wardrop
        #    a. Calcular tiempos de viaje actuales en todos los enlaces usando BPR
        if not hasattr(data, 'vdf_tensor'):
            logger.error("data.vdf_tensor no encontrado. No se puede calcular la pérdida UE.")
            current_link_travel_times = torch.full_like(predicted_flows, self.eps) # Fallback
        else:
            current_link_travel_times = self._calculate_bpr_times_all_links(predicted_flows, data)
        
        #    b. Calcular la pérdida de UE basada en estos tiempos
        loss_ue_wardrop = self._calculate_ue_loss_k_fixed_routes(predicted_flows, data, current_link_travel_times)

        total_loss = (self.training_config.w_conservation * loss_cons +
                      self.training_config.w_demand_satisfaction * loss_demand_sat_total_ga +
                      self.training_config.w_ue_wardrop * loss_ue_wardrop)
        
        self.last_loss_components = {
            "conservation": loss_cons.item(),
            "demand_satisfaction_taz": loss_demand_sat_total_ga.item(), # Solo TAZ
            "ue_wardrop_k_fixed": loss_ue_wardrop.item(),
            "total": total_loss.item()
        }
        
        return total_loss

# --- 

def custom_loss_original(predicted_flows, data, config_various: Various,
                w_observed=1.0,
                w_conservation=1.0,
                w_demand=1.0,
                normalize_losses=False):
    """
    Calcula la pérdida personalizada combinando:
      1. Error respecto a flujos observados.
      2. Error por conservación de flujos en intersecciones.
      3. Error por equilibrio con demanda neta en nodos tipo TAZ y AUX.
         Para nodos AUX, la generación y atracción se toman de data.x (columnas 5 y 6),
         que se asume son actualizadas por el modelo con parámetros aprendibles.

    Args:
        predicted_flows (torch.Tensor): Flujos predichos (shape: [num_edges]).
        data (torch_geometric.data.Data): Información de nodos y aristas. Debe incluir:
            - x: Características de nodo. Para AUX, x[:, 5] es gen, x[:, 6] es attr.
            - node_types: Lista de tipos de nodo.
            - aux_node_indices: Tensor con los índices de los nodos AUX.
            - zat_demands: Diccionario {id_original_taz: [gen, attr]} para TAZ.
            - node_id_map_rev: Mapeo de índice de nodo a ID original.
        config_various (Various): Configuración con los tipos de nodos TAZ, AUX, Intersection.
        w_observed (float): Peso para flujos observados.
        w_conservation (float): Peso para conservación de flujo.
        w_demand (float): Peso para equilibrio en TAZs y AUXs.
        normalize_losses (bool): Indica si se normalizan o no los términos.

    Returns:
        total_loss, loss_observed, loss_conservation, loss_demand
    """
    eps = 1e-8
    device = predicted_flows.device

    # 1. Pérdida por flujos observados
    loss_observed = torch.tensor(0.0, device=device)
    if hasattr(data, 'observed_flow_indices') and data.observed_flow_indices.numel() > 0:
        observed_vals = data.observed_flow_values.to(device)
        pred_vals_at_observed_indices = predicted_flows[data.observed_flow_indices]
        mse_obs = F.mse_loss(pred_vals_at_observed_indices, observed_vals, reduction='mean')

        if normalize_losses and observed_vals.numel() > 0:
            norm_obs = (observed_vals.mean() ** 2) + eps
            loss_observed = mse_obs / norm_obs if norm_obs > eps else mse_obs
        else:
            loss_observed = mse_obs

    # 2. Pérdida por conservación de flujos en intersecciones
    loss_conservation = torch.tensor(0.0, device=device)
    count_intersections = 0
    intersection_types_lower = [it.lower() for it in config_various.intersection_node_types]

    for node_idx in range(data.num_nodes):
        if str(data.node_types[node_idx]).lower() in intersection_types_lower:
            in_idx = data.in_edges_idx_tonode[node_idx].to(device)
            out_idx = data.out_edges_idx_tonode[node_idx].to(device)

            flow_in = predicted_flows[in_idx].sum() if in_idx.numel() > 0 else torch.tensor(0.0, device=device)
            flow_out = predicted_flows[out_idx].sum() if out_idx.numel() > 0 else torch.tensor(0.0, device=device)

            imbalance = flow_in - flow_out
            current_loss_term = imbalance ** 2
            if normalize_losses:
                denominator = (flow_in + flow_out)**2 + eps # O usar (abs(flow_in) + abs(flow_out))**2
                loss_conservation += current_loss_term / denominator if denominator > eps else current_loss_term
            else:
                loss_conservation += current_loss_term
            count_intersections += 1
    
    loss_conservation = loss_conservation / (count_intersections + eps) if count_intersections > 0 else torch.tensor(0.0, device=device)


    # 3. Pérdida por equilibrio con demanda en nodos TAZ y AUX
    loss_demand = torch.tensor(0.0, device=device)
    count_demand_nodes = 0
    taz_types_lower = [t.lower() for t in config_various.taz_node_types]
    aux_types_lower = [a.lower() for a in config_various.aux_node_types]

    for node_idx in range(data.num_nodes):
        node_type_lower = str(data.node_types[node_idx]).lower()
        gen_val, attr_val = 0.0, 0.0
        is_demand_node = False

        if node_type_lower in taz_types_lower:
            node_id_original = data.node_id_map_rev.get(node_idx)
            if node_id_original and node_id_original in data.zat_demands: # zat_demands es para TAZ
                gen_val, attr_val = data.zat_demands[node_id_original]
            else: # logger.warning(f"TAZ node {node_id_original} (idx {node_idx}) not found in data.zat_demands.")
                pass # Mantener G/A en 0 si no se encuentra
            is_demand_node = True
        
        elif node_type_lower in aux_types_lower:
            # Para nodos AUX, la generación y atracción vienen de data.x (columnas 5 y 6)
            # Estas columnas se asume que son actualizadas por el modelo con parámetros aprendibles.
            # data.x tiene forma [num_nodes, num_features]
            # donde las características son [is_taz, is_aux, is_int, gen_taz, attr_taz, gen_aux, attr_aux]
            if data.x.shape[1] >= 7: # Asegurar que las columnas existen
                gen_val = data.x[node_idx, 5].item() # Columna 5 para gen_aux
                attr_val = data.x[node_idx, 6].item() # Columna 6 para attr_aux
            else:
                # logger.warning(f"AUX node {node_idx} - data.x no tiene suficientes columnas para G/A aprendibles.")
                pass
            is_demand_node = True

        if is_demand_node:
            gen_tensor = torch.tensor(gen_val, device=device, dtype=predicted_flows.dtype)
            attr_tensor = torch.tensor(attr_val, device=device, dtype=predicted_flows.dtype)

            in_idx = data.in_edges_idx_tonode[node_idx].to(device)
            out_idx = data.out_edges_idx_tonode[node_idx].to(device)

            flow_in = predicted_flows[in_idx].sum() if in_idx.numel() > 0 else torch.tensor(0.0, device=device)
            flow_out = predicted_flows[out_idx].sum() if out_idx.numel() > 0 else torch.tensor(0.0, device=device)

            # Pérdida para generación: (flujo saliente - generación)^2
            # Pérdida para atracción: (flujo entrante - atracción)^2
            loss_gen = (flow_out - gen_tensor)**2
            loss_attr = (flow_in - attr_tensor)**2
            
            current_demand_loss_term = loss_gen + loss_attr

            if normalize_losses:
                # Normalizar cada término por separado
                den_gen = gen_tensor**2 + eps
                den_attr = attr_tensor**2 + eps
                
                norm_loss_gen = loss_gen / den_gen if den_gen > eps else loss_gen
                norm_loss_attr = loss_attr / den_attr if den_attr > eps else loss_attr
                loss_demand += (norm_loss_gen + norm_loss_attr)
            else:
                loss_demand += current_demand_loss_term
            
            count_demand_nodes += 1

    loss_demand = loss_demand / (count_demand_nodes + eps) if count_demand_nodes > 0 else torch.tensor(0.0, device=device)

    # Combinación ponderada
    total_loss = (w_observed * loss_observed +
                  w_conservation * loss_conservation +
                  w_demand * loss_demand)
    
    # Loggear componentes de la pérdida si es necesario para debugging
    # logger.debug(f"Losses - Obs: {loss_observed.item():.4f}, Cons: {loss_conservation.item():.4f}, Dem: {loss_demand.item():.4f}")

    return total_loss, loss_observed, loss_conservation, loss_demand


def mse_loss_vs_lp(predicted_flows, data):
    """
    Calcula el error cuadrático medio (MSE) entre los flujos predichos y los flujos asignados
    por el método de programación lineal (almacenados en data.lp_assigned_flows).
    """
    if not hasattr(data, 'lp_assigned_flows') or data.lp_assigned_flows is None:
        # logger.warning("El objeto Data no tiene 'lp_assigned_flows' o es None. Retornando MSE de 0.")
        return torch.tensor(0.0, device=predicted_flows.device)
    
    target_flows = data.lp_assigned_flows.to(predicted_flows.device).view_as(predicted_flows)
    
    # Considerar solo donde target_flows no es NaN si es posible que tenga NaNs
    valid_targets_mask = ~torch.isnan(target_flows)
    if valid_targets_mask.sum() == 0:
        # logger.warning("No hay targets válidos (no-NaN) en 'lp_assigned_flows'. Retornando MSE de 0.")
        return torch.tensor(0.0, device=predicted_flows.device)
        
    loss = F.mse_loss(predicted_flows[valid_targets_mask], target_flows[valid_targets_mask])
    return loss


def train_model(data, model, optimizer, loss_criterion: nn.Module, train_config: TrainingConfig, config_various: Various, current_dropout_rate: float):
    device = next(model.parameters()).device 
    # data = data.to(device)
    model.train()
    outputs = model(data)
    
    loss_components = {}
    if isinstance(loss_criterion, UserEquilibriumLoss):
        total_loss = loss_criterion(outputs, data)
        loss_components = loss_criterion.last_loss_components 
    elif callable(loss_criterion) and loss_criterion.__name__ == 'custom_loss_original':
         total_loss, obs_loss, cons_loss, dem_loss = loss_criterion(
             outputs, data, config_various,
             train_config.w_observed, train_config.w_conservation, train_config.w_demand_satisfaction,
             train_config.normalize_losses
         )
         loss_components = {"observed": obs_loss.item(), "conservation": cons_loss.item(), "demand": dem_loss.item()}
    elif callable(loss_criterion) and loss_criterion.__name__ == 'mse_loss_vs_lp':
        total_loss = loss_criterion(outputs, data)
    else:
        logger.error(f"Tipo de loss_criterion no reconocido: {type(loss_criterion)}")
        return model, {"error": "Criterio de pérdida no reconocido", "status": "failure", "loss": float('inf')}

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    training_epoch_info = {"loss": total_loss.item(), "status": "success"}
    if loss_components: training_epoch_info.update(loss_components)
    return model, training_epoch_info


def full_training_loop(data, model_class, optimizer_class, train_config: "TrainingConfig", model_config: "ModelConfig", config_various: "Various", current_dropout_rate: float):
    """
    Realiza un ciclo de entrenamiento completo para una configuración dada (incluyendo un dropout_rate específico).
    Retorna la pérdida final o una métrica de validación para Optuna/Random Search.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Determinar el número de nodos AUX para el modelo
    num_aux_nodes = 0
    if hasattr(data, 'aux_node_indices') and data.aux_node_indices is not None:
        num_aux_nodes = data.aux_node_indices.numel()

    # Instanciar el modelo con el dropout_rate actual
    # Asumimos que model_class es HetGATPyG aquí
    model = model_class(
        node_feat_dim=data.x.size(1),
        embed_dim=model_config.model_embed,
        num_v_layers=model_config.num_v_layers,
        num_r_layers=model_config.num_r_layers,
        num_heads=model_config.model_heads,
        ff_hidden_dim=model_config.ff_hidden,
        pred_hidden_dim=model_config.pred_hidden,
        dropout_rate=current_dropout_rate, # Pasar el dropout actual
        num_aux_nodes=num_aux_nodes,
        aux_learnable_ga_initial_scale=train_config.aux_learnable_ga_initial_scale
    ).to(device)

    optimizer = optimizer_class(model.parameters(), lr=train_config.lr)
    
    all_epoch_losses = []
    best_loss_so_far = float('inf') # O una métrica de validación

    for epoch in range(train_config.num_epochs):
        model.train() # Asegurar modo entrenamiento
        
        # Aquí llamamos a la función train_model que hace una sola pasada (backward/step)
        # y devuelve la información de esa época.
        # def train_model(data, model, optimizer, loss_criterion: nn.Module, train_config: TrainingConfig, config_various: Various, current_dropout_rate: float):
        _, epoch_info = train_model(data, model, optimizer, train_config, config_various, current_dropout_rate)
        
        if epoch_info.get("status") == "failure":
            logger.error(f"Fallo en la época {epoch+1} para dropout {current_dropout_rate:.4f}. Error: {epoch_info.get('error')}")
            return float('inf') # Penalizar esta prueba en Optuna/Random Search

        current_epoch_loss = epoch_info["loss"]
        all_epoch_losses.append(current_epoch_loss)

        if (epoch + 1) % 100 == 0 or epoch == train_config.num_epochs - 1:
            log_msg = f"Dropout: {current_dropout_rate:.4f} - Epoch {epoch + 1}/{train_config.num_epochs} - Loss: {current_epoch_loss:.4f}"
            if "observed" in epoch_info: # Si es custom_loss_original
                log_msg += f" (Obs: {epoch_info['observed']:.4f}, Cons: {epoch_info['conservation']:.4f}, Dem: {epoch_info['demand']:.4f})"
            logger.info(log_msg)
        
        # Aquí podrías implementar lógica de early stopping basada en una pérdida de validación
        # Por ahora, simplemente tomamos la pérdida de la última época o la mejor pérdida de entrenamiento.
        if current_epoch_loss < best_loss_so_far:
            best_loss_so_far = current_epoch_loss
            
    # Para Optuna/Random Search, usualmente se retorna la métrica a optimizar.
    # Podría ser la pérdida de la última época, la mejor pérdida de entrenamiento,
    # o idealmente, una métrica en un conjunto de validación.
    # Aquí retornamos la pérdida de la última época como ejemplo.
    final_loss_for_trial = all_epoch_losses[-1] if all_epoch_losses else float('inf')
    logger.info(f"Entrenamiento completo para dropout {current_dropout_rate:.4f}. Pérdida final de época: {final_loss_for_trial:.4f}")
    
    # Guardar el modelo entrenado con este dropout si es el mejor hasta ahora (lógica externa a esta función)
    # o simplemente retornar la métrica.
    return final_loss_for_trial # O best_loss_so_far, o métrica de validación
