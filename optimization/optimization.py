#!/usr/bin/env python3
"""
optimization.py

Módulo para el entrenamiento y optimización del modelo GNN para asignación de tráfico.
Incluye:
    - Funciones de pérdida: custom_loss y mse_loss_vs_lp.
    - Función train_model: que ejecuta el ciclo de entrenamiento usando la función de pérdida seleccionada.

Se espera que el objeto data (de PyTorch Geometric) incluya atributos esenciales como:
    - observed_flow_indices y observed_flow_values (para pérdida observada)
    - in_edges_idx_tonode y out_edges_idx_tonode, node_types, zat_demands, node_id_map_rev y num_nodes
      (para calcular las pérdidas de conservación y demanda).
    - lp_assigned_flows, cuando se usa la pérdida 'mse_lp'.
"""

import torch
import torch.nn.functional as F
import logging


# Configuración de logging (se asume que se invoca setup_logger en main-script)
# from utils.logger_config import setup_logger

def custom_loss(predicted_flows, data,
                w_observed=1.0,
                w_conservation=1.0,
                w_demand=1.0,
                normalize_losses=False):
    """
    Calcula la pérdida personalizada combinando:
      1. Error respecto a flujos observados.
      2. Error por conservación de flujos en intersecciones.
      3. Error por equilibrio con demanda neta en nodos tipo ZAT.

    Args:
        predicted_flows (torch.Tensor): Flujos predichos (shape: [num_edges]).
        data (torch_geometric.data.Data): Información de nodos y aristas.
        w_observed (float): Peso para flujos observados.
        w_conservation (float): Peso para conservación de flujo.
        w_demand (float): Peso para equilibrio en ZATs.
        normalize_losses (bool): Indica si se normalizan o no los términos.

    Returns:
        total_loss, loss_observed, loss_conservation, loss_demand
    """
    eps = 1e-8
    device = predicted_flows.device

    # ---------------------------------------------------
    # 1. Pérdida por flujos observados
    # ---------------------------------------------------
    loss_observed = torch.tensor(0.0, device=device)
    if hasattr(data, 'observed_flow_indices') and data.observed_flow_indices.numel() > 0:
        observed_vals = data.observed_flow_values.to(device)
        pred_vals = predicted_flows[data.observed_flow_indices]
        mse_obs = F.mse_loss(pred_vals, observed_vals, reduction='mean')

        if normalize_losses:
            norm_obs = (observed_vals.mean() ** 2) + eps
            loss_observed = mse_obs / norm_obs
        else:
            loss_observed = mse_obs

    # ---------------------------------------------------
    # 2. Pérdida por conservación de flujos en intersecciones
    # ---------------------------------------------------
    loss_conservation = torch.tensor(0.0, device=device)
    count_intersections = 0

    for node_idx in range(data.num_nodes):
        if data.node_types[node_idx] == 'intersection':
            in_idx = data.in_edges_idx_tonode[node_idx].to(device)
            out_idx = data.out_edges_idx_tonode[node_idx].to(device)

            flow_in = predicted_flows[in_idx].sum() if in_idx.numel() > 0 else torch.tensor(0.0, device=device)
            flow_out = predicted_flows[out_idx].sum() if out_idx.numel() > 0 else torch.tensor(0.0, device=device)

            imbalance = flow_in - flow_out
            if normalize_losses:
                denominator = (flow_in + flow_out)**2 + eps
                loss_conservation += (imbalance**2) / denominator
            else:
                loss_conservation += imbalance**2

            count_intersections += 1

    loss_conservation = loss_conservation / (count_intersections + eps)

    # ---------------------------------------------------
    # 3. Pérdida por equilibrio con demanda en nodos ZAT
    # ---------------------------------------------------
    loss_demand = torch.tensor(0.0, device=device)
    count_zats = 0

    for node_idx in range(data.num_nodes):
        if data.node_types[node_idx] == 'zat':
            node_id = data.node_id_map_rev[node_idx]
            gen_val, attr_val = data.zat_demands.get(node_id, (0.0, 0.0))
            gen_tensor = torch.tensor(gen_val, device=device, dtype=predicted_flows.dtype)
            attr_tensor = torch.tensor(attr_val, device=device, dtype=predicted_flows.dtype)

            in_idx = data.in_edges_idx_tonode[node_idx].to(device)
            out_idx = data.out_edges_idx_tonode[node_idx].to(device)

            flow_in = predicted_flows[in_idx].sum() if in_idx.numel() > 0 else torch.tensor(0.0, device=device)
            flow_out = predicted_flows[out_idx].sum() if out_idx.numel() > 0 else torch.tensor(0.0, device=device)

            imbalance = flow_in - flow_out
            net_demand = attr_tensor - gen_tensor  # Equilibrio considerando demanda neta

            if normalize_losses:
                denominator = (net_demand**2) + eps
                loss_demand += ((imbalance - net_demand)**2) / denominator
            else:
                loss_demand += (imbalance - net_demand)**2

            count_zats += 1

    loss_demand = loss_demand / (count_zats + eps)

    # ---------------------------------------------------
    # 4. Combinación ponderada de todas las pérdidas
    # ---------------------------------------------------
    total_loss = (w_observed * loss_observed +
                  w_conservation * loss_conservation +
                  w_demand * loss_demand)

    return total_loss, loss_observed, loss_conservation, loss_demand


def mse_loss_vs_lp(predicted_flows, data):
    """
    Calcula el error cuadrático medio (MSE) entre los flujos predichos y los flujos asignados
    por el método de programación lineal (almacenados en data.lp_assigned_flows).

    Args:
        predicted_flows (torch.Tensor): Flujos predichos (shape: [num_edges]).
        data (torch_geometric.data.Data): Objeto Data que debe incluir 'lp_assigned_flows'.

    Returns:
        torch.Tensor: Valor escalar representando el MSE.
    """
    if not hasattr(data, 'lp_assigned_flows'):
        raise AttributeError("El objeto Data no tiene 'lp_assigned_flows' requerido para mse_loss_vs_lp.")
    target_flows = data.lp_assigned_flows.to(predicted_flows.device).view_as(predicted_flows)
    loss = F.mse_loss(predicted_flows, target_flows)
    return loss


def train_model(data, model, optimizer, num_epochs=500,
                w_observed=1.0, w_conservation=1.0, w_demand=1.0,
                loss_function_type='custom', normalize_losses=False):
    """
    Ejecuta el ciclo de entrenamiento del modelo GNN utilizando la función de pérdida seleccionada.

    Args:
        data (torch_geometric.data.Data): Datos de entrada y etiquetas.
        model (torch.nn.Module): Modelo GNN pre-instanciado.
        optimizer (torch.optim.Optimizer): Optimizador para actualizar los parámetros.
        num_epochs (int): Número total de épocas.
        w_observed (float): Peso para la pérdida de flujos observados.
        w_conservation (float): Peso para la pérdida de conservación de flujos.
        w_demand (float): Peso para la pérdida basada en demanda.
        loss_function_type (str): Tipo de pérdida a utilizar ('custom' o 'mse_lp').

    Returns:
        tuple: (modelo entrenado, información de entrenamiento como diccionario con pérdidas por época).
    """
    device = next(model.parameters()).device
    data = data.to(device)

    # Seleccionar la función de pérdida en base al parámetro
    if loss_function_type == 'custom':
        loss_fn = lambda pred: custom_loss(pred, data, w_observed, w_conservation, w_demand, normalize_losses)[0]
        logging.info(
            f"Usando custom_loss: w_observed={w_observed}, w_conservation={w_conservation}, w_demand={w_demand}")
        required_attrs = [
            'observed_flow_indices', 'observed_flow_values',
            'in_edges_idx_tonode', 'out_edges_idx_tonode',
            'node_types', 'zat_demands', 'node_id_map_rev', 'num_nodes'
        ]
        for attr in required_attrs:
            if not hasattr(data, attr):
                logging.error(f"El objeto Data no tiene el atributo '{attr}' requerido para custom_loss.")
                return model, {"error": f"Falta {attr}"}
    elif loss_function_type == 'mse_lp':
        loss_fn = lambda pred: mse_loss_vs_lp(pred, data)
        logging.info("Usando mse_loss_vs_lp para la pérdida.")
        if not hasattr(data, 'lp_assigned_flows'):
            logging.error("El objeto Data no contiene 'lp_assigned_flows' necesario para mse_loss_vs_lp.")
            return model, {"error": "Falta lp_assigned_flows"}
    else:
        logging.error(f"Tipo de función de pérdida inválido: {loss_function_type}")
        return model, {"error": "Función de pérdida inválida"}

    training_info = {"epochs_run": 0, "losses": []}

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(data)
        if loss_function_type == 'custom':
            loss, loss_obs, loss_cons, loss_dem = custom_loss(outputs, data, w_observed, w_conservation, w_demand,
                                                              normalize_losses)
        else:
            loss = mse_loss_vs_lp(outputs, data)
        loss.backward()
        optimizer.step()

        training_info["epochs_run"] = epoch + 1
        training_info["losses"].append(loss.item())
        if (epoch % 100 == 0) or (epoch == num_epochs - 1):
            logging.info(f"Epoch {epoch + 1}/{num_epochs} - Loss: {loss.item():.2f} - Obs: {loss_obs.item():.2f} "
                         f"- Cons: {loss_cons.item():.2f} - Demand {loss_dem.item():.2f}")

    return model, training_info
