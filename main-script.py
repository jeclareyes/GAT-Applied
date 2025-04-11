# main-script.py
import torch
# Importaciones actualizadas/revisadas
from data.loader import load_traffic_data_custom
from models.gat_model import TrafficGAT
from models.save_model import save_model # Aún necesita revisión interna
from optimization.optimization import train_model, optimize_sensor_network, compute_model_metrics # Pruning aún necesita revisión
from visualization.network_viz import NetworkVisualizer_Pyvis # Se usará el actualizado
# evaluation necesita adaptación también
# from evaluation.evaluation import validate_model, print_model_performance
import logging
import os
import datetime
import json
from utils.logger_config import setup_logger
import pandas as pd
import numpy as np # Añadido por si acaso

#%% Directorios de salida
output_viz_dir = os.path.join("visualization", "exported_viz_data")
output_eval_dir = "evaluation"
output_models_dir = "models"
os.makedirs(output_viz_dir, exist_ok=True)
os.makedirs(output_eval_dir, exist_ok=True)
os.makedirs(output_models_dir, exist_ok=True) # Asegura que exista para guardar modelos

#%% Control del color de logger
setup_logger()

#%%
def main(pickle_file, do_train, do_prune, save_model_flag, print_graph, interface_mode):
    """
    Flujo principal para optimización/análisis de red de sensores de tráfico:
    1. Carga datos (usando loader adaptado)
    2. Entrena modelo (usando trainer adaptado, pruning necesita revisión)
    3. Realiza predicciones
    4. Valida rendimiento (métricas básicas sobre observados)
    5. Visualiza resultados (pasando objeto Data y predicciones completas)
    """
    # Semilla para reproducibilidad
    torch.manual_seed(42)
    np.random.seed(42) # Para numpy si se usa
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    # Opcional: flags de determinismo (pueden ralentizar)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

    #%% 1. Carga datos usando el loader actualizado
    logging.info(f"1. Cargando datos de tráfico desde '{pickle_file}'...")
    try:
        # Usa la función de carga actualizada
        data = load_traffic_data_custom(pickle_file)
        logging.info("Datos cargados exitosamente.")
        logging.info(f" Resumen: {data.num_nodes} nodos, {data.num_edges} aristas.")
        if not hasattr(data, 'link_types'):
             logging.error("¡Error crítico! El objeto Data cargado no tiene el atributo 'link_types'. Verifica 'data/loader.py'.")
             return # Salir si falta información esencial
    except FileNotFoundError:
        logging.error(f"Error: No se encontró el archivo pickle '{pickle_file}'. Verifica la ruta.")
        return
    except Exception as e:
        logging.error(f"Falló la carga de datos desde {pickle_file}: {e}", exc_info=True)
        return

    #%% 2. Entrenamiento (Parte de Pruning/Optimización necesita revisión)
    model = None
    predicted_flows = None # Inicializar tensor de predicciones

    # Parámetros del modelo y entrenamiento
    # (Puedes moverlos a un archivo de configuración si prefieres)
    model_params = {'hidden_dim': 16, 'heads': 2} # Consistente con GAT usado
    train_params = {'num_epochs': 500, 'lr': 0.01, 'w_observed': 10, 'w_conservation': 50, 'w_demand': 100}


    if do_train:
        if do_prune:
            # --- SECCIÓN PRUNING NECESITA REVISIÓN ---
            logging.warning("\n'do_prune' es True, pero la lógica de poda (optimize_sensor_network) necesita adaptación significativa.")
            logging.warning("Ejecutando entrenamiento estándar en su lugar.")
            # Fallback a entrenamiento estándar
            model = train_model(data,
                                model_class=TrafficGAT,
                                **model_params, # Pasa parámetros del modelo
                                **train_params # Pasa parámetros de entrenamiento
                                )
            # Mascara optimizada no es aplicable aquí, usamos la original si existe
            # optimized_mask = data.observed_flow_mask.clone() if hasattr(data, 'observed_flow_mask') else None
            # --- Llamada original comentada ---
            # model, optimized_mask = optimize_sensor_network(...)
        else:
            # Entrenamiento estándar sin optimización/poda
            logging.info("\n2. Entrenando modelo GAT...")
            model = train_model(data,
                                model_class=TrafficGAT,
                                **model_params,
                                **train_params
                                )

        if save_model_flag and model:
            # Guardado de modelo - save_model.py necesita revisión interna
            # Debería guardar state_dict, configuración, etc.
            model_save_path = os.path.join(output_models_dir, f"gat_model_{datetime.datetime.now():%Y%m%d_%H%M%S}.pt")
            logging.warning(f"Intentando guardar modelo en '{model_save_path}' usando 'save_model' (necesita revisión).")
            # Idealmente, guardarías el state_dict y la configuración
            try:
                 torch.save({
                     'epoch': train_params['num_epochs'],
                     'model_state_dict': model.state_dict(),
                     # 'optimizer_state_dict': optimizer.state_dict(), # Si guardaras el optimizador
                     'config': {'model_params': model_params, 'train_params': train_params},
                     # 'loss': loss, # Podrías guardar la última pérdida
                 }, model_save_path)
                 logging.info(f"Modelo (state_dict) guardado en {model_save_path}")
            except Exception as e:
                 logging.error(f"Error al guardar el state_dict del modelo: {e}", exc_info=True)
            # La función save_model actual parece guardar una estructura diferente
            # save_model(model, optimized_mask, route=model_save_path) # Descomentar si adaptas save_model

    else:
        logging.info("Saltando entrenamiento del modelo (do_train=False).")
        # Aquí podrías cargar un modelo pre-entrenado si 'do_train' es False
        # model_load_path = "path/to/your/saved/model.pt"
        # if os.path.exists(model_load_path):
        #     checkpoint = torch.load(model_load_path)
        #     model = TrafficGAT(in_channels=data.x.size(1), **checkpoint['config']['model_params'])
        #     model.load_state_dict(checkpoint['model_state_dict'])
        #     model.to(data.x.device) # Mover a dispositivo
        #     logging.info(f"Modelo cargado desde {model_load_path}")
        # else:
        #     logging.warning(f"No se encontró modelo pre-entrenado en {model_load_path}")


    #%% 3. Realizar Predicciones y Validar Rendimiento (Básico)
    if model is not None:
        logging.info("\n3. Realizando predicciones y validando rendimiento...")
        model.eval() # Poner modelo en modo evaluación
        with torch.no_grad(): # Desactivar cálculo de gradientes
            # Obtener predicciones para TODAS las aristas
            predicted_flows = model(data) # Tensor de forma [num_edges]

        # Extraer información relevante para la validación y visualización
        observed_indices = data.observed_flow_indices.to(predicted_flows.device) # Asegurar mismo dispositivo
        observed_values = data.observed_flow_values.to(predicted_flows.device)
        predicted_observed = predicted_flows[observed_indices] # Predicciones solo para aristas observadas

        # Crear DataFrame de comparación para aristas observadas
        if observed_indices.numel() > 0: # Si hay aristas observadas
            edge_pairs_observed = data.edge_index[:, observed_indices].t().cpu().numpy()
            node_map_rev = data.node_id_map_rev

            df_eval = pd.DataFrame({
                'edge_from_idx': edge_pairs_observed[:, 0],
                'edge_to_idx': edge_pairs_observed[:, 1],
                'edge_from_id': [node_map_rev.get(idx, 'N/A') for idx in edge_pairs_observed[:, 0]],
                'edge_to_id': [node_map_rev.get(idx, 'N/A') for idx in edge_pairs_observed[:, 1]],
                'predicted_flow': predicted_observed.cpu().numpy(),
                'observed_flow': observed_values.cpu().numpy()
            })
            df_eval['abs_error'] = (df_eval['predicted_flow'] - df_eval['observed_flow']).abs()
            df_eval['rel_error'] = 100 * (df_eval['abs_error'] / df_eval['observed_flow']).replace([np.inf, -np.inf], np.nan)

            eval_save_path = os.path.join(output_eval_dir, "observed_predictions_comparison.csv")
            df_eval.to_csv(eval_save_path, index=False)
            logging.info(f"Comparación de predicciones guardada en: {eval_save_path}")

            # Calcular métricas básicas sobre aristas observadas
            mae = df_eval['abs_error'].mean()
            rmse = np.sqrt((df_eval['abs_error']**2).mean())
            num_observed = len(df_eval)

            logging.info(f"Rendimiento sobre {num_observed} enlaces observados:")
            logging.info(f"  MAE:  {mae:.4f}")
            logging.info(f"  RMSE: {rmse:.4f}")
        else:
            logging.warning("No hay enlaces observados para calcular métricas de rendimiento.")

        # Las funciones originales de validación (validate_model, print_model_performance)
        # probablemente necesiten una revisión completa si se usan, ya que asumían
        # una estructura de datos y máscaras diferente.

    else:
        logging.warning("Saltando predicciones y validación: Modelo no disponible.")


    #%% 4. Visualización (usando network_viz.py actualizado)
    if print_graph:
        if model is not None and predicted_flows is not None and hasattr(data, 'link_types'):
            logging.info("\n4. Visualizando la red con flujos GAT...")

            try:
                # Instanciar el visualizador actualizado
                # Pasamos el objeto 'data' completo y las predicciones COMPLETAS
                viz = NetworkVisualizer_Pyvis(
                    data=data,
                    predicted_flows=predicted_flows # Tensor completo [num_edges]
                )

                # Configuración opcional (ejemplo, adaptado del generador)
                # Puedes cargar esto desde un JSON si prefieres
                viz.update_config(
                    # --- Estilos de Nodos ---
                    node_styles={
                        'zat': {'color': "#FFD700", 'shape': 'diamond', 'size': 20, 'font_weight': 'bold', 'border_color': "black", 'border_width': 2},
                        'intersection': {'color': "#87CEEB", 'shape': 'dot', 'size': 12, 'border_color': "darkgrey", 'border_width': 1}
                    },
                    node_label_font_size=12,
                    node_show_label=True, # Mostrar ID de nodo
                    node_show_demand_title=True, # Mostrar Gen/Attr en tooltip para ZATs

                    # --- Estilos de Enlaces ---
                    link_styles={
                        'road': {'color_observed_match': "#006400", 'color_observed_mismatch': "#FF0000", 'color_unobserved': "#808080", 'width_factor': 0.8, 'dashes': False},
                        'logical': {'color_observed_match': "#ADD8E6", 'color_observed_mismatch': "#FFA07A", 'color_unobserved': "#D3D3D3", 'width_factor': 0.5, 'dashes': [5, 5]}
                    },
                    link_show_label=True,
                    link_label_font_size=9,
                    link_label_prefixes={'predicted': "GAT:", 'observed': "Obs:"},
                    link_width_base=1.0,
                    link_width_scaling_method="log", # 'linear' o 'log'
                    link_max_width=8.0, # Límite superior para ancho
                    link_observed_tolerance=0.1, # Tolerancia relativa para 'match'

                    # --- Física y Exportación ---
                    physics_solver="forceAtlas2Based", # Opciones: barnesHut, repulsion, etc.
                    export_filepath=os.path.join(output_viz_dir, "gat_network_visualization.html"),
                    show_buttons=True # Mostrar controles de PyVis
                )

                # Generar y guardar la visualización HTML
                html_output_path = viz.draw() # Usa la ruta en config por defecto
                logging.info(f"Visualización interactiva guardada en: {html_output_path}")

            except Exception as e:
                 logging.error(f"Error durante la visualización de la red: {e}", exc_info=True)

            # --- Guardar datos para visualización externa (opcional, revisado) ---
            try:
                export_prefix = os.path.join(output_viz_dir, "viz_data")
                # Guardar datos esenciales del objeto Data
                torch.save(data.node_coordinates, f"{export_prefix}_node_coords.pt")
                torch.save(data.edge_index, f"{export_prefix}_edge_index.pt")
                with open(f"{export_prefix}_node_types.json", 'w') as f:
                    json.dump({'types': data.node_types, 'ids': [data.node_id_map_rev[i] for i in range(data.num_nodes)]}, f)
                if hasattr(data, 'link_types'):
                     with open(f"{export_prefix}_link_types.json", 'w') as f:
                        json.dump({'types': data.link_types}, f)
                # Guardar flujos (predichos completos, observados dispersos)
                torch.save(predicted_flows, f"{export_prefix}_predicted_flows_full.pt")
                torch.save({'indices': data.observed_flow_indices, 'values': data.observed_flow_values},
                           f"{export_prefix}_observed_flows_sparse.pt")

                logging.info(f"Datos para visualización externa guardados en '{output_viz_dir}' con prefijo 'viz_data'")
            except Exception as e:
                logging.error(f"Error guardando datos para visualización externa: {e}", exc_info=True)
            # --- Fin Guardar datos ---

            # --- Lanzar GUI (Streamlit/Flask - Necesitan adaptación) ---
            # El código para lanzar Streamlit/Flask necesitaría ser actualizado
            # para cargar y usar los nuevos archivos de datos guardados.
            # Ejemplo (comentado):
            # if interface_mode == "streamlit":
            #     logging.info("\n Lanzando interfaz Streamlit (necesita adaptación)...")
            #     # ... código para lanzar streamlit ...
            # elif interface_mode == "flask":
            #      logging.warning("Flask GUI necesita adaptación.")
            #      # ... código para lanzar flask ...

        elif not hasattr(data, 'link_types'):
             logging.error("Saltando visualización: Falta el atributo 'link_types' en el objeto Data.")
        else:
            logging.warning("Saltando visualización: Modelo no entrenado o predicciones no disponibles.")
    else:
        logging.info("Saltando visualización (print_graph=False).")


#%% Punto de entrada principal
if __name__ == "__main__":
    # Configuración (mejor si se carga desde archivo/argumentos)
    config = {
        "pickle_file": f'data/traffic_data_{10}.pkl', # Ruta relativa a la raíz del proyecto
        "do_train": True,
        "do_prune": False,  # Poda necesita revisión, mantener en False por ahora
        "save_model_flag": True, # Guardar state_dict del modelo
        "print_graph": True, # Generar visualización HTML
        "interface_mode": ""  # Opciones: "streamlit", "flask","" (solo HTML estático)
    }

    # Crear directorio 'data' si no existe (para el pickle de ejemplo)
    data_dir = os.path.dirname(config["pickle_file"])
    if data_dir and not os.path.exists(data_dir):
         os.makedirs(data_dir)
         logging.info(f"Directorio '{data_dir}' creado.")
         # Podrías copiar/crear un pickle de ejemplo aquí si es necesario para la primera ejecución

    # Llamar a la función principal
    main(
        pickle_file=config["pickle_file"],
        do_train=config["do_train"],
        do_prune=config["do_prune"],
        save_model_flag=config["save_model_flag"],
        print_graph=config["print_graph"],
        interface_mode=config["interface_mode"]
    )