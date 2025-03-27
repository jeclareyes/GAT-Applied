# optimization/pruning.py
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from evaluation.evaluation import compute_model_metrics

def train_model(data, model_class, hidden_dim=64, heads=4, num_epochs=100, lr=0.01):
    model = model_class(data.x.size(1), data.edge_attr.size(1), hidden_dim, heads).to(data.x.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    for _ in range(num_epochs):
        optimizer.zero_grad()
        preds = model(data)
        loss = F.mse_loss(preds[data.train_mask], data.edge_attr.squeeze()[data.train_mask])

        #% Conservation

        loss_conservation = model.conservation_loss(preds, data)
        total_loss = loss + loss_conservation

        #%
        total_loss.backward()
        optimizer.step()
    return model

def compute_edge_importance(model, data, num_epochs=100):
    """
    Compute edge importance by training the model and analyzing attention weights.
    
    Args:
        model (TrafficGAT): Graph Attention Network model
        data (Data): PyTorch Geometric Data object
        num_epochs (int, optional): Number of training epochs. Defaults to 100.
    
    Returns:
        numpy.ndarray: Edge importance scores
    """
    device = data.x.device
    model = model.to(device)
    data = data.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    model.train()
    model.reset_attention_weights()  # Clear previous attention weights

    for _ in range(num_epochs):
        optimizer.zero_grad()
        loss = F.mse_loss(model(data)[data.train_mask], data.edge_attr.squeeze()[data.train_mask])
        loss.backward()
        optimizer.step()

    # Combine attention weights across layers
    attentions = torch.cat(model.attention_weights, dim=1)
    edge_importance = attentions.mean(dim=1)
    
    return edge_importance.cpu().numpy()

def performance_based_pruning(model, data, edge_importance, 
                               max_error_increase=0.10, metric='mae'):
    """
    Prune sensors based on their importance while maintaining model performance.
    
    Args:
        model (TrafficGAT): Graph Attention Network model
        data (Data): PyTorch Geometric Data object
        edge_importance (numpy.ndarray): Computed edge importance scores
        max_error_increase (float, optional): Maximum allowed error increase. Defaults to 0.10.
        metric (str, optional): Performance metric to use. Defaults to 'mae'.

    Returns:
        tuple: Optimized mask, final error, number of removed sensors
    """
    sensor_edges = data.train_mask.nonzero().squeeze()
    sorted_indices = np.argsort(edge_importance[sensor_edges])

    current_mask = data.train_mask.clone()
    initial_error = compute_model_metrics(model, data, current_mask, metric)
    current_error = initial_error

    removed_sensors = 0
    pbar = tqdm(total=len(sorted_indices), desc="Sensor Pruning")

    for idx in sorted_indices:
        temp_mask = current_mask.clone()
        temp_mask[sensor_edges[idx]] = False
        new_error = compute_model_metrics(model, data, temp_mask, metric)

        # Check if error increase is within acceptable range
        if (new_error - initial_error) / initial_error <= max_error_increase:
            current_mask = temp_mask
            current_error = new_error
            removed_sensors += 1
        else:
            break
        pbar.update(1)

    pbar.close()

    print(f"Pruning complete. Removed {removed_sensors}/{len(sensor_edges)} sensors.")
    return current_mask, current_error, removed_sensors

def optimize_sensor_network(data, model_class, hidden_dim=64, heads=4, 
                             max_error_increase=0.10, metric='mae', num_epochs=100):
    """
    Complete pipeline to train model, prune sensors, and retrain with optimized sensors.
    
    Args:
        data (Data): PyTorch Geometric Data object
        model_class (type): Model class to instantiate
        hidden_dim (int, optional): Hidden dimension for the model. Defaults to 64.
        heads (int, optional): Number of attention heads. Defaults to 4.
        max_error_increase (float, optional): Maximum allowed error increase. Defaults to 0.10.
        metric (str, optional): Performance metric to use. Defaults to 'mae'.
        num_epochs (int, optional): Number of training epochs. Defaults to 100.
    
    Returns:
        tuple: Optimized model, optimized mask
    """
    # Initial model training
    model = model_class(data.x.size(1), data.edge_attr.size(1), hidden_dim, heads)
    
    # Compute edge importance
    edge_importance = compute_edge_importance(model, data, num_epochs=num_epochs)

    # Perform sensor pruning
    optimized_mask, final_error, removed_sensors = performance_based_pruning(
        model, data, edge_importance, max_error_increase, metric
    )

    # Update data with optimized mask
    data.train_mask = optimized_mask

    # Retrain model after pruning
    model = model_class(data.x.size(1), data.edge_attr.size(1), hidden_dim, heads)
    
    # Recompute edge importance with pruned model
    compute_edge_importance(model, data, num_epochs=num_epochs)

    # Compute final metrics
    final_mae = compute_model_metrics(model, data, optimized_mask, 'mae')
    final_rmse = compute_model_metrics(model, data, optimized_mask, 'rmse')

    print(f"Final MAE: {final_mae:.4f}, Final RMSE: {final_rmse:.4f}")

    return model, optimized_mask
