# config.py
from dataclasses import dataclass

@dataclass
class TrainingConfig:
    num_epochs: int = 500
    lr: float = 0.01
    w_observed: float = 1.0
    w_conservation: float = 1.0
    w_demand: float = 1.0
    loss_function_type: str = 'custom'  # 'custom' or 'mse_lp'
    normalize_losses: bool = True  # or True
    save_model_flag: bool = False

@dataclass
class ModelConfig:
    model_run: str = 'HetGATPyG'  # 'HetGATPyG' or 'TrafficGAT'
    model_hidden: int = 8  # 16
    model_heads: int = 2  # 2
    model_embed: int = 16  # 32
    num_v_layers: int = 2  # 2
    num_r_layers: int = 2  # 2
    ff_hidden: int = 16  # 32
    pred_hidden: int = 8  # 16
    dropout: float = 0.2  # 0.6

@dataclass
class Directories:
    input_pickle: str = 'traffic_data_big.pkl'
    output_viz_dir: str = "visualization/exported_viz_data"
    output_eval_dir: str = "evaluation"
    output_models_dir: str = "models"
    odm_dir_file: str = "layer_procesing/data/Mobile Data/odm.csv"
    input_real_network: str = 'layer_procesing/data/geopackages/final_network.gpkg'

@dataclass
class Various:
    read_pickle: bool = False
    read_real_data: bool= True

# You can then import these in your main script:
# from config import TrainingConfig, ModelConfig, Directories
