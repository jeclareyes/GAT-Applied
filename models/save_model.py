import json
import datetime
import torch
import os
import logging
from utils import logger_config

def save_model(model, mask ,route=None):
    # Definir el modelo y las variables (como en tu ejemplo)
    logger_config.setup_logger()

    model = {
        "TrafficGAT": {
            "conv1": "GATConv(2, 64, heads=4)",
            "conv2": "GATConv(256, 64, heads=1)",
            "regressor": [
                "Linear(in_features=64, out_features=64, bias=True)",
                "ReLU()",
                "Linear(in_features=64, out_features=1, bias=True)"
            ]
        },
        "T_destination": "TypeVar ~T_destination",
        "attention_weights": "tensor([...])",  # Aquí puedes almacenar los pesos reales en formato serializable (tensor -> lista o str)
        "call_super_init": False,
        "dump_patches": False,
        "training": True
    }

    # Obtener el timestamp
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Crear un registro con el timestamp
    record = {
        "timestamp": timestamp,
        "model_details": model
    }

    if route is not None:
        route = ('')

    # Ruta al archivo donde se guardarán los registros
    file_name = "model_records.json"
    file_path = os.path.join(route, file_name)

    # Cargar los registros existentes (si los hay) y agregar el nuevo
    try:
        with open(file_path, "r") as file:
            data = json.load(file)
    except (FileNotFoundError, json.JSONDecodeError):
        data = []

    # Agregar el nuevo registro
    data.append(record)

    # Guardar el archivo con el nuevo registro
    with open(file_path, "w") as file:
        json.dump(data, file, indent=4)

    print(f"Modelo guardado en {file_path} con timestamp {timestamp}.")
