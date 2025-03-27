# evaluation/metrics.py
import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pandas as pd

def compute_model_metrics(model, data, mask, metric='mae'):
    """
    Compute specified evaluation metric for the model.
    
    Args:
        model (nn.Module): Trained neural network model
        data (Data): PyTorch Geometric Data object
        mask (torch.Tensor): Mask for selecting specific edges
        metric (str, optional): Metric to compute. Defaults to 'mae'.
    
    Returns:
        float: Computed metric value
    
    Raises:
        ValueError: If an unsupported metric is specified
    """
    model.eval()
    with torch.no_grad():
        #  predictions = model(data)[mask].cpu().numpy() #WHY DOES THIS NEED A MASK
        predictions = model(data).cpu().numpy()
        targets = data.y[mask].cpu().numpy()

    edge_index = data.edge_index.cpu().numpy().T
    df_links = pd.DataFrame({
        'source': edge_index[:, 0],
        'target': edge_index[:, 1],
        'predicted_flow': predictions,
        'train_mask': data.train_mask.cpu().numpy(),
        'true_flow': data.y.cpu().numpy(),
        'observed_flow': data.edge_attr.squeeze().cpu().numpy(),
    })

    # Guardar a CSV
    df_links.to_csv("link_predictions.csv", index=False)

    print('Se imprimió dataframe de resultados\n', df_links.head())

def validate_model(model, data, mask):
    """
    Comprehensive model validation with multiple metrics.
    
    Args:
        model (nn.Module): Trained neural network model
        data (Data): PyTorch Geometric Data object
        mask (torch.Tensor): Mask for selecting specific edges
    
    Returns:
        dict: Dictionary of performance metrics
    """
    model.eval()
    with torch.no_grad():
        predictions = model(data)[mask].cpu().numpy()
        targets = data.y[mask].cpu().numpy()
        
        return {
            'rmse': np.sqrt(mean_squared_error(targets, predictions)),
            'mae': mean_absolute_error(targets, predictions),
            'mse': mean_squared_error(targets, predictions),
            'r2': r2_score(targets, predictions)
        }


def print_model_performance(metrics):
    """
    Print model performance metrics in a formatted way.
    
    Args:
        metrics (dict): Dictionary of performance metrics
    """
    print("Model Performance Metrics:")
    for metric, value in metrics.items():
        print(f"{metric.upper()}: {value:.4f}")

