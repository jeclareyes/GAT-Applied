# optimization/optimization.py
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
# from evaluation.evaluation import compute_model_metrics # Keep but might need adjustments
import logging
from utils.logger_config import setup_logger

# Assuming models.gat_model TrafficGAT class is imported correctly
# from models.gat_model import TrafficGAT

setup_logger()


# --- Added Custom Loss Function ---
# (Copied from previous response, ensure consistency)
def custom_loss(predicted_flows, data, w_observed=1.0, w_conservation=0.5, w_demand=0.5):
    loss_observed = torch.tensor(0.0, device=predicted_flows.device)
    loss_conservation = torch.tensor(0.0, device=predicted_flows.device)
    loss_demand = torch.tensor(0.0, device=predicted_flows.device)

    # 1. Error en flujos observados
    if hasattr(data, 'observed_flow_indices') and data.observed_flow_indices.numel() > 0:
        observed_vals = data.observed_flow_values.to(predicted_flows.device)
        pred_vals = predicted_flows[data.observed_flow_indices]
        loss_observed = F.mse_loss(pred_vals, observed_vals)

    # 2. Conservación en intersecciones y 3. Demanda en ZATs
    eps = 1e-8
    count_int = 0
    count_zat = 0

    for node_idx in range(data.num_nodes):
        node_type = data.node_types[node_idx]
        in_idx = data.in_edges_idx_tonode[node_idx].to(predicted_flows.device)
        out_idx = data.out_edges_idx_tonode[node_idx].to(predicted_flows.device)

        flow_in = predicted_flows[in_idx].sum() if in_idx.numel() > 0 else 0.0
        flow_out = predicted_flows[out_idx].sum() if out_idx.numel() > 0 else 0.0

        if node_type == 'intersection':
            loss_conservation += torch.abs(flow_in - flow_out)
            count_int += 1

        elif node_type == 'zat':
            gen, attr = data.zat_demands.get(data.node_id_map_rev[node_idx], (0.0, 0.0))
            gen = torch.tensor(gen, device=predicted_flows.device)
            attr = torch.tensor(attr, device=predicted_flows.device)
            loss_demand += torch.abs(flow_out - gen) + torch.abs(flow_in - attr)
            count_zat += 1

    loss_conservation /= (count_int + eps)
    loss_demand /= (count_zat + eps)

    total_loss = w_observed * loss_observed + w_conservation * loss_conservation + w_demand * loss_demand
    return total_loss, loss_observed, loss_conservation, loss_demand


# --- Modified Train Function ---
def train_model(data, model_class, hidden_dim=64, heads=4, num_epochs=500, lr=0.01,
                w_observed=1.0, w_conservation=0.5, w_demand=0.5):
    model = model_class(in_channels=data.x.size(1), hidden_dim=hidden_dim, heads=heads).to(data.x.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    logging.info("Inicio del entrenamiento con pérdida compuesta...")
    model.train()

    for epoch in range(num_epochs):
        optimizer.zero_grad()
        pred = model(data)
        total, l_obs, l_cons, l_dem = custom_loss(pred, data,
                                                  w_observed=w_observed,
                                                  w_conservation=w_conservation,
                                                  w_demand=w_demand)
        total.backward()
        optimizer.step()

        if epoch % 100 == 0 or epoch == num_epochs - 1:
            logging.info(
                f"Epoch {epoch:04d} | Total: {total.item():.4f} | Obs: {l_obs.item():.4f} | Cons: {l_cons.item():.4f} | Dem: {l_dem.item():.4f}")

    logging.info("Entrenamiento finalizado.")
    return model


# --- Pruning and Optimization Functions (NEED ADAPTATION) ---
# NOTE: The functions below (compute_edge_importance, performance_based_pruning,
# optimize_sensor_network) are likely incompatible with the new model structure
# and loss function. They rely on the old way of calculating MSE loss directly
# on edges using train_mask and edge_attr, and potentially on analyzing
# attention weights which might have changed or been removed.
# Adapting these requires rethinking how "importance" is measured (maybe based on
# gradient magnitude, effect on loss components, etc.) and how pruning affects
# the custom loss. Commenting out or careful review/rewrite is needed.

def compute_edge_importance(model, data, num_epochs=100):
    logging.warning("compute_edge_importance needs review/adaptation for the new model/loss structure.")
    # Placeholder: return random importance for now
    return np.random.rand(data.num_edges)
    # --- Original code below is likely incompatible ---
    # device = data.x.device
    # model = model.to(device)
    # data = data.to(device)
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    # model.train()
    # model.reset_attention_weights() # Assumes this method exists
    # for _ in range(num_epochs):
    #     optimizer.zero_grad()
    #     # Original loss calculation based on edge_attr and train_mask
    #     loss = F.mse_loss(model(data)[data.train_mask], data.edge_attr.squeeze()[data.train_mask])
    #     loss.backward()
    #     optimizer.step()
    # attentions = torch.cat(model.attention_weights, dim=1) # Assumes attention_weights are stored
    # edge_importance = attentions.mean(dim=1)
    # return edge_importance.cpu().numpy()


def compute_model_metrics(model, data, metric='mae'):
    """
    Computes basic metrics based on observed flows.
    NOTE: Assumes 'data' contains observed_indices and observed_values.
          The concept of a 'mask' argument is less relevant here unless
          referring to a specific subset of observed values for evaluation.
    """
    model.eval()
    with torch.no_grad():
        pred_flows = model(data)

    if not hasattr(data, 'observed_indices') or data.observed_indices.numel() == 0:
        logging.warning("No observed flows found in data for metric calculation.")
        return 0.0 if metric == 'count' else float('nan')  # Return NaN or 0

    # Get predictions corresponding to observed flows
    pred_observed = pred_flows[data.observed_indices]
    true_observed = data.observed_values.to(pred_observed.device)  # Ensure same device

    if metric == 'mae':
        return F.l1_loss(pred_observed, true_observed).item()
    elif metric == 'rmse':
        return torch.sqrt(F.mse_loss(pred_observed, true_observed)).item()
    elif metric == 'mse':
        return F.mse_loss(pred_observed, true_observed).item()
    elif metric == 'count':
        return data.observed_indices.numel()  # Number of observed sensors
    else:
        logging.error(f"Unsupported metric: {metric}")
        return float('nan')


def performance_based_pruning(model, data, edge_importance,
                              max_error_increase=0.10, metric='mae'):
    logging.warning("performance_based_pruning needs significant review/adaptation.")
    # This function needs a major rewrite. It relies on manipulating train_mask
    # and recalculating metrics based on that mask, which doesn't fit the new
    # loss structure where observations are handled differently.
    # Returning the original observed mask as a placeholder.
    if hasattr(data, 'observed_mask'):
        return data.observed_mask.clone(), compute_model_metrics(model, data, metric), 0
    else:
        # Cannot proceed without observed_mask
        raise AttributeError("Data object missing 'observed_mask' required for placeholder pruning.")


def optimize_sensor_network(data, model_class, hidden_dim=64, heads=4,
                            max_error_increase=0.10, metric='mae', num_epochs=1000):  # Added num_epochs back
    logging.warning("optimize_sensor_network (pruning pipeline) needs significant review/adaptation.")
    logging.warning("Running basic training instead of pruning.")
    # Fallback to just training the model without pruning
    model = train_model(data, model_class, hidden_dim, heads, num_epochs)  # Use the adapted train_model
    # Return the trained model and the original observed mask
    optimized_mask = data.observed_mask.clone() if hasattr(data, 'observed_mask') else None  # Or handle error
    if optimized_mask is None:
        logging.error("Cannot return optimized_mask as it's missing from data.")
    return model, optimized_mask
