#!/usr/bin/env python3
"""
main-script.py

Flujo principal para la carga de datos, instanciación y entrenamiento del modelo de asignación de tráfico,
evaluación y visualización. Este script utiliza un archivo de configuración central para agrupar todos los
parámetros de entrada, facilitando su modificación y extensibilidad.
"""

import os
import datetime
import logging
import json
import torch
import numpy as np

import config
# Configuración inicial del logger (aquí o en utils/logger_config.py)
from utils.logger_config import setup_logger

setup_logger()

# Importar funciones de carga y procesamiento de datos
from data.loader import load_traffic_data_pickle, add_virtual_links, load_traffic_data_dict

# Importar definiciones de modelos
from models.gat_model import TrafficGAT
from models.HeGAT import HetGATPyG

# Importar funciones de entrenamiento y optimización
from optimization.optimization import train_model

# Importar funciones de evaluación
from evaluation.evaluation import compute_model_metrics, print_model_performance

# Importar el visualizador de red
from visualization.network_viz import NetworkVisualizer_Pyvis

# Importar configuración centralizada
from config import TrainingConfig, ModelConfig, Directories, Various


# Definición de una clase para encapsular el entrenamiento (opcional pero recomendable)
class TrafficFlowTrainer:
    def __init__(self, data, model, optimizer, train_config: TrainingConfig):
        self.data = data
        self.model = model
        self.optimizer = optimizer
        self.train_config = train_config

    def train(self):
        logging.info(f"Inicio del entrenamiento por {self.train_config.num_epochs} épocas.")
        # Invocar la función de entrenamiento definida en optimization/optimization.py
        # Se usan los parámetros (p.ej., pesos de loss) definidos en la configuración.
        self.model, training_info = train_model(
            data=self.data,
            model=self.model,
            optimizer=self.optimizer,
            num_epochs=self.train_config.num_epochs,
            w_observed=self.train_config.w_observed,
            w_conservation=self.train_config.w_conservation,
            w_demand=self.train_config.w_demand,
            loss_function_type=self.train_config.loss_function_type,
            normalize_losses=self.train_config.normalize_losses
        )
        logging.info("Entrenamiento finalizado.")
        return training_info


def main():
    # Cargar configuraciones centralizadas
    train_config = TrainingConfig()
    model_config = ModelConfig()
    dirs = Directories()

    # Crear directorios de salida (evaluación, visualización, modelos)

    os.makedirs(dirs.output_eval_dir, exist_ok=True)
    os.makedirs(dirs.output_viz_dir, exist_ok=True)
    os.makedirs(dirs.output_models_dir, exist_ok=True)

    # Fijar semilla para reproducibilidad
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    # ----------------------
    # 0. Carga Manual de Datos
    # ----------------------

    if config.Various.read_pickle is False:
        data = load_traffic_data_dict()
        logging.info("Carga de datos completada exitosamente.")

    # ----------------------
    # 1. Carga de Datos
    # ----------------------
    pickle_file = f"data/{config.Directories.input_pickle}"

    if config.Various.read_pickle is True:
        try:
            data = load_traffic_data_pickle(pickle_file)
            logging.info("Carga de datos completada exitosamente.")
            # Verificar atributos esenciales
            if not hasattr(data, 'link_types'):
                logging.error("El objeto Data no tiene el atributo 'link_types'. Verifica 'loader.py'.")
                return
        except Exception as e:
            logging.error(f"Error al cargar datos: {e}", exc_info=True)
            return

    # ----------------------
    # 2. Instanciación del Modelo
    # ----------------------
    # Determinar el dispositivo
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Se usará el dispositivo: {device}")

    # Seleccionar e instanciar el modelo basado en la configuración
    if model_config.model_run == "TrafficGAT":
        logging.info("Instanciando el modelo TrafficGAT...")
        model = TrafficGAT(
            in_channels=data.x.size(1),
            hidden_dim=model_config.model_hidden,
            heads=model_config.model_heads
        ).to(device)
        # Garantizar que data tenga los atributos necesarios para TrafficGAT
        if not hasattr(data, 'virtual_edge_index'):
            data.virtual_edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
    elif model_config.model_run == "HetGATPyG":
        logging.info("Instanciando el modelo HetGATPyG...")
        # Asegurar que se añadan los enlaces virtuales requeridos
        if not hasattr(data, 'virtual_edge_index'):
            logging.info("Agregando enlaces virtuales necesarios para HetGATPyG...")
            if config.Various.read_pickle is True:
                data = add_virtual_links(data)
        if not hasattr(data, 'virtual_edge_index') or data.virtual_edge_index is None:
            logging.error("Error: No se crearon los enlaces virtuales necesarios para HetGATPyG.")
            return

        #data.x = data.x[:, 2:]

        model = HetGATPyG(
            node_feat_dim=data.x.size(1),
            embed_dim=model_config.model_embed,
            num_v_layers=model_config.num_v_layers,
            num_r_layers=model_config.num_r_layers,
            num_heads=model_config.model_heads,
            ff_hidden_dim=model_config.ff_hidden,
            pred_hidden_dim=model_config.pred_hidden,
            dropout=model_config.dropout
        ).to(device)
    else:
        logging.error(f"Valor de 'model_run' no reconocido en la configuración: {model_config.model_run}")
        return

    # ----------------------
    # 3. Configuración del Optimizador y Entrenamiento
    # ----------------------
    optimizer = torch.optim.Adam(model.parameters(), lr=train_config.lr)
    trainer = TrafficFlowTrainer(data, model, optimizer, train_config)
    training_info = trainer.train()

    # ----------------------
    # 4. Guardado del Modelo
    # ----------------------
    if train_config.save_model_flag and model is not None:
        model_save_path = os.path.join(
            dirs.output_models_dir,
            f"gat_model_{datetime.datetime.now():%Y%m%d_%H%M%S}.pt"
        )
        try:
            torch.save({
                'epoch': training_info.get("epochs_run", 0),
                'model_state_dict': model.state_dict(),
                'config': {
                    'model_params': vars(model_config),
                    'train_params': vars(train_config)
                },
            }, model_save_path)
            logging.info(f"Modelo guardado exitosamente en '{model_save_path}'")
        except Exception as e:
            logging.error(f"Error al guardar el modelo: {e}", exc_info=True)
    else:
        logging.info("Saltando el guardado del modelo (flag desactivado o sin entrenamiento).")

    # ----------------------
    # 5. Evaluación
    # ----------------------
    try:
        # Establecer el modelo en modo evaluación y obtener predicciones
        model.eval()
        with torch.no_grad():
            _ = model(data)  # Se asume que el forward de model actualiza data o se usa en la evaluación

        # Llamar a la función de evaluación para obtener el DataFrame y las métricas
        df_eval, metrics = compute_model_metrics(model, data, print_results=True)
        eval_filepath = os.path.join(dirs.output_eval_dir, "link_predictions.csv")
        df_eval.to_csv(eval_filepath, index=False)
        logging.info(f"Resultados de evaluación guardados en '{eval_filepath}'")
        print_model_performance(metrics)
    except Exception as e:
        logging.error(f"Error durante la evaluación: {e}", exc_info=True)

    # ----------------------
    # 6. Visualización
    # ----------------------
    try:
        with torch.no_grad():
            predicted_flows = model(data).detach().cpu()
        visualizer = NetworkVisualizer_Pyvis(data, predicted_flows)
        viz_filepath = os.path.join(dirs.output_viz_dir, "gat_network_visualization.html")
        #visualizer.save_visualization(viz_filepath)
        logging.info(f"Visualización guardada en: {viz_filepath}")
    except Exception as e:
        logging.error(f"Error durante la visualización: {e}", exc_info=True)


if __name__ == "__main__":
    main()
