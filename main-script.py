#!/usr/bin/env python3
"""
main-script.py
Flujo principal para la carga de datos, instanciación y entrenamiento del modelo.
"""

import os
import datetime
import logging
import torch
import numpy as np
import random
import optuna
import torch.nn as nn 

from utils.logger_config import setup_logger
setup_logger()

from data.loader import load_traffic_data_pickle, load_real_traffic_data
from models.HeGAT import HeGATPyG
from models.gat_model import TrafficGAT # Asegúrate que exista o comenta si no se usa

from optimization.optimization import (
    full_training_loop, 
    custom_loss_original, # Renombrada
    mse_loss_vs_lp,
    train_model,
    UserEquilibriumLoss # Nueva clase
)
from evaluation.evaluation import compute_model_metrics, print_model_performance
from visualization.network_viz import NetworkVisualizer_Pyvis
from config import TrainingConfig, ModelConfig, Directories, Various

logger = logging.getLogger(__name__)

def objective_optuna(trial, data, model_class, optimizer_class, loss_criterion_instance, base_train_config, model_config, config_various):
    """
    Función objetivo para Optuna.
    loss_criterion_instance ya está inicializado y se pasa aquí.
    """
    current_dropout_rate = trial.suggest_float(
        "dropout_rate", 
        base_train_config.dropout_range[0], 
        base_train_config.dropout_range[1]
    )
    logger.info(f"[Optuna Trial {trial.number}] Probando dropout: {current_dropout_rate:.4f}")

    # Pasamos la instancia de loss_criterion directamente
    metric_to_optimize = full_training_loop(
        data, model_class, optimizer_class, 
        loss_criterion_instance, # Pasar la instancia
        base_train_config, model_config, config_various, 
        current_dropout_rate
    )
    return metric_to_optimize

def main():
    train_config = TrainingConfig()
    model_config = ModelConfig()
    dirs = Directories()
    config_various = Various()

    os.makedirs(dirs.output_eval_dir, exist_ok=True)
    os.makedirs(dirs.output_viz_dir, exist_ok=True)
    os.makedirs(dirs.output_models_dir, exist_ok=True)

    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    data = None
    if config_various.read_pickle:
        try:
            pickle_file = os.path.join("data", dirs.input_pickle)
            data = load_traffic_data_pickle(pickle_file, config_various, train_config.aux_learnable_ga_initial_scale)
            logger.info("Carga de datos desde Pickle completada.")
        except Exception as e:
            logger.error(f"Error al cargar datos desde Pickle: {e}", exc_info=True)
            return
    elif config_various.read_real_data:
        try:
            data = load_real_traffic_data(dirs.input_real_network, dirs.odm_dir_file, config_various, train_config.aux_learnable_ga_initial_scale)
            logger.info("Carga de datos reales completada.")
        except Exception as e:
            logger.error(f"Error al cargar datos reales: {e}", exc_info=True)
            return
    else:
        logger.error("No se especificó un método de carga de datos.")
        return

    if data is None or data.x is None:
        logger.error("La carga de datos falló o data.x no está definido.")
        return
    
    logger.info(f"Datos cargados: {data.num_nodes} nodos, {data.edge_index.shape[1] if data.edge_index is not None and data.edge_index.numel() > 0 else 0} aristas.")
    if hasattr(data, 'aux_node_indices'):
         logger.info(f"Número de nodos AUX identificados para G/A aprendible: {data.aux_node_indices.numel()}")


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Se usará el dispositivo: {device}")

    model_class_to_use = None
    if model_config.model_run == "HetGATPyG":
        model_class_to_use = HeGATPyG
    elif model_config.model_run == "TrafficGAT":
        model_class_to_use = TrafficGAT
    else:
        logger.error(f"Modelo no reconocido: {model_config.model_run}")
        return
    
    optimizer_class_to_use = torch.optim.Adam

    # --- Instanciar el criterio de pérdida ---
    loss_criterion_instance = None
    if train_config.loss_function_type == 'user_equilibrium':
        loss_criterion_instance = UserEquilibriumLoss(config_various, train_config)
        logger.info("Usando UserEquilibriumLoss.")
    elif train_config.loss_function_type == 'custom_original':
        # custom_loss_original es una función, no una clase nn.Module.
        # Se pasará directamente a full_training_loop que la manejará.
        loss_criterion_instance = custom_loss_original 
        logger.info("Usando custom_loss_original.")
    elif train_config.loss_function_type == 'mse_lp':
        loss_criterion_instance = mse_loss_vs_lp 
        logger.info("Usando mse_loss_vs_lp.")
    else:
        logger.error(f"Tipo de función de pérdida no reconocido: {train_config.loss_function_type}")
        return
    
    # Mover la instancia de UserEquilibriumLoss al dispositivo si es un nn.Module
    if isinstance(loss_criterion_instance, nn.Module):
        loss_criterion_instance.to(device)


    best_dropout_rate = train_config.fixed_dropout_rate
    
    if train_config.dropout_optimization_method == 'fixed':
        logger.info(f"Usando dropout fijo: {best_dropout_rate}")

    elif train_config.dropout_optimization_method == 'random_search':
        logger.info(f"Iniciando Random Search para dropout ({train_config.dropout_search_trials} intentos)...")
        best_metric_random = float('inf')
        for i in range(train_config.dropout_search_trials):
            current_dropout = random.uniform(train_config.dropout_range[0], train_config.dropout_range[1])
            logger.info(f"[Random Search Trial {i+1}/{train_config.dropout_search_trials}] Probando dropout: {current_dropout:.4f}")
            metric_value = full_training_loop(
                data, model_class_to_use, optimizer_class_to_use,
                loss_criterion_instance, # Pasar la instancia/función
                train_config, model_config, config_various,
                current_dropout
            )
            if metric_value < best_metric_random:
                best_metric_random = metric_value
                best_dropout_rate = current_dropout
            logger.info(f"[Random Search Trial {i+1}] Dropout: {current_dropout:.4f}, Métrica: {metric_value:.4f}. Mejor: {best_dropout_rate:.4f} ({best_metric_random:.4f})")
        logger.info(f"Random Search finalizado. Mejor dropout: {best_dropout_rate:.4f}, Métrica: {best_metric_random:.4f}")

    elif train_config.dropout_optimization_method == 'optuna':
        logger.info(f"Iniciando Optimización con Optuna para dropout ({train_config.dropout_search_trials} intentos)...")
        study = optuna.create_study(direction="minimize")
        study.optimize(
            lambda trial: objective_optuna(trial, data, model_class_to_use, optimizer_class_to_use, loss_criterion_instance, train_config, model_config, config_various),
            n_trials=train_config.dropout_search_trials
        )
        best_dropout_rate = study.best_trial.params["dropout_rate"]
        logger.info(f"Optuna finalizado. Mejor dropout: {best_dropout_rate:.4f}, Métrica: {study.best_trial.value:.4f}")

    logger.info(f"Dropout final seleccionado para entrenamiento: {best_dropout_rate:.4f}")

    # --- Entrenamiento final ---
    logger.info(f"Iniciando entrenamiento final con dropout = {best_dropout_rate:.4f}")
    num_aux_nodes = data.aux_node_indices.numel() if hasattr(data, 'aux_node_indices') else 0
    final_model = model_class_to_use(
        node_feat_dim=data.x.size(1),
        embed_dim=model_config.model_embed,
        num_v_layers=model_config.num_v_layers,
        num_r_layers=model_config.num_r_layers,
        num_heads=model_config.model_heads,
        ff_hidden_dim=model_config.ff_hidden,
        pred_hidden_dim=model_config.pred_hidden,
        dropout_rate=best_dropout_rate,
        num_aux_nodes=num_aux_nodes,
        aux_learnable_ga_initial_scale=train_config.aux_learnable_ga_initial_scale
    ).to(device)
    final_optimizer = optimizer_class_to_use(final_model.parameters(), lr=train_config.lr)
    
    training_losses_final_run = []
    for epoch in range(train_config.num_epochs):
        # Usar la instancia de loss_criterion para el entrenamiento final también
        _, epoch_info = train_model(data, final_model, final_optimizer, loss_criterion_instance, train_config, config_various, best_dropout_rate)
        
        if epoch_info.get("status") == "failure":
            logger.error(f"Fallo en el entrenamiento final en la época {epoch+1}. Error: {epoch_info.get('error')}")
            return
            
        current_loss = epoch_info["loss"]
        training_losses_final_run.append(current_loss)
        
        if (epoch + 1) % 100 == 0 or epoch == train_config.num_epochs - 1:
            log_msg = f"Entrenamiento Final - Epoch {epoch + 1}/{train_config.num_epochs} - Loss: {current_loss:.4f}"
            # Loguear componentes si están disponibles
            if "conservation" in epoch_info:
                 log_msg += f" (Cons: {epoch_info.get('conservation',0):.4f}, Dem: {epoch_info.get('demand_satisfaction',0):.4f}"
                 if "ue_wardrop_placeholder" in epoch_info: # Nombre de la clave en UserEquilibriumLoss.last_loss_components
                     log_msg += f", UE_Placeholder: {epoch_info.get('ue_wardrop_placeholder',0):.4f}"
                 if "observed" in epoch_info: # Para custom_loss_original
                     log_msg += f", Obs: {epoch_info.get('observed',0):.4f}"
                 log_msg += ")"
            logger.info(log_msg)
            
    logger.info("Entrenamiento final completado.")
    training_info_final = {"epochs_run": train_config.num_epochs, "losses": training_losses_final_run, "final_dropout": best_dropout_rate}

    # Guardado del Modelo Final
    if train_config.save_model_flag and final_model is not None:
        # ... (código de guardado como en tu versión anterior) ...
        pass
    else:
        logger.info("Saltando el guardado del modelo final.")

    # Evaluación
    try:
        # Establecer el modelo en modo evaluación y obtener predicciones
        final_model.eval()
        with torch.no_grad():
            _ = final_model(data)  # Se asume que el forward de model actualiza data o se usa en la evaluación

        # Llamar a la función de evaluación para obtener el DataFrame y las métricas
        df_eval, metrics = compute_model_metrics(final_model, data, print_results=True)
        eval_filepath = os.path.join(dirs.output_eval_dir, "link_predictions.csv")
        df_eval.to_csv(eval_filepath, index=False)
        logging.info(f"Resultados de evaluación guardados en '{eval_filepath}'")
        print_model_performance(metrics)
    except Exception as e:
        logging.error(f"Error durante la evaluación: {e}", exc_info=True)

    # Visualización
    try:
        # ... (código de visualización como en tu versión anterior) ...
        pass
    except Exception as e:
        logger.error(f"Error durante la visualización: {e}", exc_info=True)

if __name__ == "__main__":
    main()
