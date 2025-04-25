#!/usr/bin/env python3
"""
evaluation.py

Módulo para la evaluación del desempeño del modelo de asignación de tráfico.
Provee funciones para calcular indicadores de desempeño (MAE, MSE, RMSE, R²) y
para generar un DataFrame con la comparación de los flujos predichos, reales y observados.
El DataFrame se guarda en un archivo CSV para facilitar su análisis.
"""

import torch
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import logging


def compute_model_metrics(model, data, mask=None, ground_truth_type="observed", print_results=False):
    """
    Calcula las métricas de evaluación y genera un DataFrame completo con información para todos los enlaces.

    La columna 'true_flow' tendrá los valores de ground-truth (observados o asignados) únicamente en los enlaces
    en los que se disponen de esos datos, y NaN en el resto. Las métricas se calculan exclusivamente en el
    subconjunto de enlaces con valores de ground-truth.

    Args:
        model (torch.nn.Module): Modelo entrenado con método forward.
        data (torch_geometric.data.Data): Objeto Data que debe incluir:
            - edge_index: Tensor [2, num_edges] con la topología.
            - ground truth: Dependiendo del parámetro, se espera:
                * data.observed_flow_values para "observed".
                * data.assigned_flow_values para "assigned".
            - (Opcional) train_mask: Tensor booleano indicando qué enlaces fueron usados para entrenamiento.
        mask (torch.Tensor, opcional): Máscara booleana para evaluar solo un subconjunto de enlaces.
        ground_truth_type (str, opcional): "observed" o "assigned".
        print_results (bool, opcional): Si True, imprime las primeras filas del DataFrame y las métricas.

    Returns:
        tuple:
          - df_eval (pd.DataFrame): DataFrame con columnas:
            'source', 'target', 'predicted_flow', 'true_flow', 'train_mask'
          - metrics (dict): Diccionario con las métricas calculadas (mae, mse, rmse, r2) sobre los enlaces con target.
    """
    model.eval()
    with torch.no_grad():
        # Obtener las predicciones para todos los enlaces y aplanarlas a un vector unidimensional.
        predictions = model(data).detach().cpu().flatten()

        # Seleccionar el target según ground_truth_type.
        if ground_truth_type == "assigned":
            if hasattr(data, 'assigned_flow_values'):
                available_targets = data.assigned_flow_values.detach().cpu().flatten()
            else:
                logging.error("El objeto Data no tiene 'assigned_flow_values'.")
                raise AttributeError("El objeto Data no tiene 'assigned_flow_values'.")
        else:  # Por defecto, "observed"
            if hasattr(data, 'observed_flow_values'):
                available_targets = data.observed_flow_values.detach().cpu().flatten()
            else:
                logging.error("El objeto Data no tiene 'observed_flow_values'.")
                raise AttributeError("El objeto Data no tiene 'observed_flow_values'.")

    # Número total de enlaces (predicciones).
    num_edges = predictions.shape[0]
    # Crear un vector completo para los targets con NaN.
    full_targets = np.full(num_edges, np.nan)
    # Suponiendo que los datos ground-truth están disponibles en un subconjunto de enlaces,
    # se debe tener un atributo que indique los índices a los que corresponden esos valores.
    # En nuestro caso, para "observed" se usa data.observed_flow_indices.
    # Si se usa "assigned" se asumirá que la ground-truth abarca todos los enlaces,
    # o bien se debe disponer de un vector de índices (a adaptar según tus necesidades).
    if ground_truth_type == "observed":
        #observed_idx = data.observed_flow_indices.cpu().numpy().flatten()
        observed_idx = data.observed_flow_indices.cpu().numpy().flatten().astype(int)
    else:
        # Para "assigned", asumimos que se dispone de ground-truth para todos los enlaces.
        observed_idx = np.arange(num_edges)

    # Asegurarse de que available_targets tiene la misma dimensión que los índices en observed_idx.
    if len(available_targets) != len(observed_idx):
        logging.error("La dimensión de ground-truth no coincide con la cantidad de índices disponibles.")
        raise ValueError("Dimensiones inconsistentes en ground-truth y sus índices.")

    # Asignar los valores disponibles a full_targets en las posiciones correspondientes.
    full_targets[observed_idx] = available_targets.numpy()

    # Si se aplica una máscara adicional, se filtran todas las variables.
    if mask is not None:
        mask_np = mask.cpu().numpy().flatten()
        predictions = predictions.numpy()[mask_np]
        full_targets = full_targets[mask_np]
        edge_index = data.edge_index.cpu().numpy().T[mask_np]
        train_mask = data.train_mask.cpu().numpy().flatten()[mask_np] if hasattr(data, 'train_mask') else np.array(
            [None] * len(predictions))
    else:
        predictions = predictions.numpy()
        edge_index = data.edge_index.cpu().numpy().T  # Forma [num_edges, 2]
        train_mask = data.train_mask.cpu().numpy().flatten() if hasattr(data, 'train_mask') else np.array(
            [None] * num_edges)

    # Crear el DataFrame completo con la posición original de cada enlace.
    df_eval = pd.DataFrame({
        'source': edge_index[:, 0],
        'target': edge_index[:, 1],
        'predicted_flow': predictions,
        'true_flow': full_targets,
        'train_mask': train_mask
    })

    # Calcular métricas sólo en los enlaces en los que el target está definido (no NaN).
    valid_mask = ~np.isnan(full_targets)
    if np.sum(valid_mask) > 0:
        predictions_valid = predictions[valid_mask]
        targets_valid = full_targets[valid_mask]
        mae = mean_absolute_error(targets_valid, predictions_valid)
        mse = mean_squared_error(targets_valid, predictions_valid)
        rmse = np.sqrt(mse)
        r2 = r2_score(targets_valid, predictions_valid)
        metrics = {"mae": mae, "mse": mse, "rmse": rmse, "r2": r2}
    else:
        logging.error("No se encontraron enlaces con ground-truth para evaluar.")
        metrics = {"mae": None, "mse": None, "rmse": None, "r2": None}

    # Guardar el DataFrame en un archivo CSV.
    csv_path = "link_predictions.csv"
    df_eval.to_csv(csv_path, index=False)
    logging.info(f"DataFrame de evaluación guardado en: {csv_path}")

    if print_results:
        print("Primeros 5 registros del DataFrame de evaluación:")
        print(df_eval.head())
        print("\nMétricas de Evaluación (sobre enlaces válidos):")
        for key, value in metrics.items():
            print(f"{key.upper()}: {value:.4f}" if value is not None else f"{key.upper()}: No disponible")

    return df_eval, metrics

def validate_model(model, data, mask=None):
    """
    Evalúa el modelo utilizando compute_model_metrics y retorna las métricas calculadas.

    Args:
        model (torch.nn.Module): Modelo entrenado.
        data (torch_geometric.data.Data): Objeto Data.
        mask (torch.Tensor, opcional): Máscara para seleccionar enlaces a evaluar.

    Returns:
        dict: Diccionario con las métricas (mae, mse, rmse, r2)
    """
    _, metrics = compute_model_metrics(model, data, mask=mask, print_results=False)
    return metrics


def print_model_performance(metrics):
    """
    Imprime de forma formateada las métricas de desempeño del modelo.

    Args:
        metrics (dict): Diccionario con métricas (mae, mse, rmse, r2)
    """
    print("=== Desempeño del Modelo ===")
    for metric, value in metrics.items():
        print(f"{metric.upper()}: {value:.4f}")
