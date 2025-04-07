# main.py
from torch_geometric.data import Data
import torch
from data.loader import load_traffic_data
from models.gat_model import TrafficGAT
from models.save_model import save_model
from optimization.optimization import optimize_sensor_network, train_model, compute_model_metrics
from visualization.network_viz import NetworkVisualizer_Pyvis
from evaluation.evaluation import validate_model, print_model_performance
from utils.simple_viz import previsualization_pyvis
import logging
import os
#  import sys
import datetime
import json
from utils.logger_config import setup_logger

#%% Guardado de cosas
output_viz = os.path.join("visualization", "exported_viz_data")
os.makedirs(output_viz, exist_ok=True)

#%% Control del color de logger
setup_logger()


#%%
def main():
    """
    Main workflow for traffic sensor network optimization:
    1. Load data
    2. Optimize sensor network
    3. Visualize results
    4. Validate model performance
    """
    # Set random seed for reproducibility
    torch.manual_seed(42)

    #%% 1. Load traffic data
    print("1. Loading traffic data...")
    data = load_traffic_data(PICKLE_FILE)

    ###

    """x = torch.tensor([
        [1., 1.],
        [0., 0.],
        [0., 0.]
    ])

    edge_index = torch.tensor([
        [0, 1, 2],
        [1, 2, 0]
    ])

    edge_attr = torch.tensor([
        [0.],
        [0.],
        [0.]
    ])

    y = torch.tensor([1., 1., 1.])
    train_mask = torch.tensor([True, True, True])

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, train_mask=train_mask)"""
    ###



    previsualization_pyvis(data)

    #%% 2. Training and Optimization
    if DO_TRAIN:
        if DO_PRUNE:
            # Optimize sensor network
            print("\n2. Optimizing sensor network...")
            model, optimized_mask = optimize_sensor_network(
                data,
                model_class=TrafficGAT,
                hidden_dim=64,
                heads=4,
                max_error_increase=0.10,
                metric='mae',
                num_epochs=1000
            )
        else:
            # Training without optimization
            print("\n2. Training model on all sensors...")
            model = train_model(data, model_class=TrafficGAT, hidden_dim=64, heads=4, num_epochs=1000, lr=0.01)
            optimized_mask = data.train_mask.clone()
        if SAVE_MODEL:
            save_model(model, optimized_mask, route=None)
    else:
        pass

    #%% 3. Visualization
    if PRINT_GRAPH:
        logging.info("3. Visualizando red de sensores...")
        viz = NetworkVisualizer_Pyvis(
            nodes=data.x,
            links=data.edge_index,
            estimated_flows=model(data),
            observed_flows=data.y
        )
        viz.update_config(
            edge_label_mode="combined",
            node_size_scaling_method="flow",
            export_filepath=os.path.join(output_viz, "network_visualization.html")
        )

        #%% Guardado de cosas para su posterior visualización
        torch.save(data.x, os.path.join(output_viz, "nodes.pt"))
        torch.save(data.edge_index, os.path.join(output_viz, "links.pt"))
        torch.save(model(data), os.path.join(output_viz, "estimated_flows.pt"))
        torch.save(data.y, os.path.join(output_viz, "observed_flows.pt"))

        logging.info(f"Datos de visualización exportados en {output_viz}")
        #%%
        if INTERFACE_MODE == "streamlit":
            print("\n Se lanza la interfaz de Streamlit...")
            try:
                print("Ejecuta el siguiente comando para iniciar la interfaz gráfica:")
                print(f"streamlit run {os.path.join('visualization', 'network_gui_streamlit.py')}")

                # Opcional: lanzar automáticamente streamlit (descomenta estas líneas si lo prefieres)
                import subprocess
                subprocess.Popen([
                    "python", "-m", "streamlit", "run",
                    os.path.join("visualization", "network_gui_streamlit.py")
                ])
            except Exception as e:
                print(f"Error al intentar lanzar Streamlit: {e}")
                print("Intenta ejecutar manualmente: streamlit run visualization/network_gui_streamlit.py")

        elif INTERFACE_MODE == "flask":
            from visualization.network_gui_flask import main as gui_main
            gui_main()
        else:
            logging.info("No se eligió una interfaz gráfica. Se exporta HTML estático.")
            html_output = viz.draw()
            logging.info("Visualización generada en 'network_visualization.html'")
    else:
        logging.info("3. Salteando visualización.")

    #%% 4. Validate model performance
    print("\n4. Validating model performance...")
    # Assuming test_mask is the complement of train_mask
    test_mask = ~data.train_mask
    performance_metrics = validate_model(model, data, test_mask)
    print_model_performance(performance_metrics, print_results=False)
    compute_model_metrics(model, data, test_mask, print_results=False)


#%%
if __name__ == "__main__":
    PICKLE_FILE = "traffic_data_5.pkl"
    DO_TRAIN = True
    DO_PRUNE = False
    SAVE_MODEL = False
    PRINT_GRAPH = True
    INTERFACE_MODE = ""  # Cambiar a "flask" para usar el modo Flask
    main()
