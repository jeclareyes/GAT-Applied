# config.py
from dataclasses import dataclass, field
from typing import List, Tuple

@dataclass
class TrainingConfig:
    num_epochs: int = 100
    lr: float = 0.01

    w_observed: float = 1.0
    w_conservation: float = 1.0 # Conservacion del flujo en intersecciones
    w_demand_satisfaction: float = 1.0 # Satisfaccion de demanda en TAZs y AUX
    w_ue_wardrop: float = 1.0 # Wardrop

    # Opciones para la función de pérdida: 'custom_original', 'mse_lp', 'user_equilibrium'
    loss_function_type: str = 'custom_original'  # '', 'custom' or 'mse_lp'
    normalize_losses: bool = True  # or True
    save_model_flag: bool = False

    # Nuevas configuraciones para optimización de dropout
    dropout_optimization_method: str = 'fixed'  # 'fixed', 'random_search', 'optuna'
    fixed_dropout_rate: float = 1.0  # Usado si dropout_optimization_method es 'fixed'
    dropout_search_trials: int = 10  # Número de intentos para random_search u optuna
    dropout_range: Tuple[float, float] = (0.1, 0.7) # Rango para buscar dropout

    # Nueva configuración para la inicialización de G/A aprendibles de nodos AUX
    aux_learnable_ga_initial_scale: float = 1.0 # Escala inicial para G/A de AUX

    # Numero de rutas a considerar para cada par OD
    k_routes_for_ue_loss: int = 3

@dataclass
class ModelConfig:
    model_run: str = 'HetGATPyG'  # 'HetGATPyG' or 'TrafficGAT'
    # Parámetros para HetGATPyG
    model_embed: int = 16
    num_v_layers: int = 2
    num_r_layers: int = 2
    model_heads: int = 2
    ff_hidden: int = 16
    pred_hidden: int = 8
    # dropout: float = 0.2 # Este será manejado por TrainingConfig

    # Parámetros para TrafficGAT (si se usa)
    gat_model_hidden: int = 8 # Renombrado para evitar conflicto
    gat_model_heads: int = 2  # Renombrado para evitar conflicto

@dataclass
class Directories:
    input_pickle: str = 'traffic_data_big.pkl'
    output_viz_dir: str = "visualization/exported_viz_data"
    output_eval_dir: str = "evaluation"
    output_models_dir: str = "models"
    odm_dir_file: str = "layer_procesing/data/Mobile Data/odm.csv"
    input_real_network: str = 'layer_procesing/data/geopackages/final_network.gpkg'
    graph_to_load: str = 'layer_procesing/data/output/grafo_vial.graphml'

@dataclass
class Various:
    read_pickle: bool = False
    read_real_data: bool= True
    # Identificadores para tipos de nodos (sensible a mayúsculas/minúsculas según tus datos)
    taz_node_types: List[str] = field(default_factory=lambda: ['zat', 'TAZ'])
    aux_node_types: List[str] = field(default_factory=lambda: ['aux', 'AUX'])
    intersection_node_types: List[str] = field(default_factory=lambda: ['intersection', 'Intersection', 'border_node', 'Border_Node'])
    road_link_types: List[str] = field(default_factory=lambda: ['road'])

@dataclass
class Odm_params:
    aggregation_method: str = 'average' # 'average', 'one_date', 'weekdays', 'weekends', 'dates_range'
    one_date = '2022-10-05'
    dates_range = ('2022-10-05', '2022-10-10')

@dataclass
class Vdf:
    grouped_by: str = 'emme_@vtyp' # 'emme_VDF' or 'emme_@vtyp' or 'composite'
    
    capacity = {
    'VTYP_4.0':            2000,
    'VTYP_5.0':            2000,
    'VTYP_9.0':            2000,
    'VTYP_10.0':           2000}
    
    vdf_dictionary = {
        'composite': {
            'VTYP_9.0_SP_40.0'           :{'alpha': 0.15, 'beta': 4},
            'VTYP_9.0_SP_50.0'           :{'alpha': 0.15, 'beta': 4},
            'VTYP_9.0_SP_70.0'           :{'alpha': 0.15, 'beta': 4},
            'VTYP_9.0_SP_60.0'           :{'alpha': 0.15, 'beta': 4},
            'VTYP_10.0_SP_70.0'          :{'alpha': 0.15, 'beta': 4},
            'VTYP_4.0_SP_60.0'           :{'alpha': 0.15, 'beta': 4},
            'VTYP_10.0_SP_80.0'          :{'alpha': 0.15, 'beta': 4},
            'VTYP_9.0_SP_100.0'          :{'alpha': 0.15, 'beta': 4},
            'VTYP_10.0_SP_100.0'         :{'alpha': 0.15, 'beta': 4},
            'VTYP_4.0_SP_70.0'           :{'alpha': 0.15, 'beta': 4},
            'VTYP_4.0_SP_40.0'           :{'alpha': 0.15, 'beta': 4},
            'VTYP_9.0_SP_80.0'           :{'alpha': 0.15, 'beta': 4},
            'VTYP_9.0_SP_90.0'           :{'alpha': 0.15, 'beta': 4},
            'VTYP_4.0_SP_50.0'           :{'alpha': 0.15, 'beta': 4},
            'VTYP_5.0_SP_120.0'          :{'alpha': 0.15, 'beta': 4},
            'VTYP_4.0_SP_80.0'           :{'alpha': 0.15, 'beta': 4},
            'VTYP_5.0_SP_90.0'           :{'alpha': 0.15, 'beta': 4},
            'VTYP_5.0_SP_70.0'           :{'alpha': 0.15, 'beta': 4},
            'VTYP_5.0_SP_110.0'          :{'alpha': 0.15, 'beta': 4},
            'VTYP_10.0_SP_110.0'         :{'alpha': 0.15, 'beta': 4}
            },
        'emme_@vtyp': {
            'VTYP_4.0': {'alpha': 0.15, 'beta': 4},
            'VTYP_5.0': {'alpha': 0.15, 'beta': 4},
            'VTYP_9.0': {'alpha': 0.15, 'beta': 4},
            'VTYP_10.0': {'alpha': 0.15, 'beta': 4}}
            ,
        'emme_VDF': {
            'VDF_1.0': {'alpha': 0.15, 'beta': 4},
            'VDF_3.0': {'alpha': 0.15, 'beta': 4},
            'VDF_7.0': {'alpha': 0.15, 'beta': 4},
            'VDF_11.0': {'alpha': 0.15, 'beta': 4},
            'VDF_16.0': {'alpha': 0.15, 'beta': 4},
            'VDF_18.0': {'alpha': 0.15, 'beta': 4},
            'VDF_19.0': {'alpha': 0.15, 'beta': 4},
            'VDF_20.0': {'alpha': 0.15, 'beta': 4},
            'VDF_23.0': {'alpha': 0.15, 'beta': 4},
            'VDF_24.0': {'alpha': 0.15, 'beta': 4},
            'VDF_27.0': {'alpha': 0.15, 'beta': 4},
            'VDF_28.0': {'alpha': 0.15, 'beta': 4},
            'VDF_29.0': {'alpha': 0.15, 'beta': 4},
            'VDF_30.0': {'alpha': 0.15, 'beta': 4},
            'VDF_31.0': {'alpha': 0.15, 'beta': 4},
            'VDF_32.0': {'alpha': 0.15, 'beta': 4},
            'VDF_41.0': {'alpha': 0.15, 'beta': 4},
            'VDF_43.0': {'alpha': 0.15, 'beta': 4},
            'VDF_46.0': {'alpha': 0.15, 'beta': 4},
            'VDF_47.0': {'alpha': 0.15, 'beta': 4},
            'VDF_51.0': {'alpha': 0.15, 'beta': 4},
            'VDF_53.0': {'alpha': 0.15, 'beta': 4},
            'VDF_54.0': {'alpha': 0.15, 'beta': 4},
            'VDF_55.0': {'alpha': 0.15, 'beta': 4},
            'VDF_56.0': {'alpha': 0.15, 'beta': 4},
            'VDF_57.0': {'alpha': 0.15, 'beta': 4},
            'VDF_58.0': {'alpha': 0.15, 'beta': 4},
            'VDF_59.0': {'alpha': 0.15, 'beta': 4},
            'VDF_60.0': {'alpha': 0.15, 'beta': 4},
            'VDF_61.0': {'alpha': 0.15, 'beta': 4},
            'VDF_62.0': {'alpha': 0.15, 'beta': 4},
            'VDF_63.0': {'alpha': 0.15, 'beta': 4},
            'VDF_64.0': {'alpha': 0.15, 'beta': 4},
            'VDF_65.0': {'alpha': 0.15, 'beta': 4},
            'VDF_66.0': {'alpha': 0.15, 'beta': 4},
            'VDF_67.0': {'alpha': 0.15, 'beta': 4},
            'VDF_68.0': {'alpha': 0.15, 'beta': 4},
            'VDF_69.0': {'alpha': 0.15, 'beta': 4},
            'VDF_70.0': {'alpha': 0.15, 'beta': 4},
            'VDF_71.0': {'alpha': 0.15, 'beta': 4},
            'VDF_72.0': {'alpha': 0.15, 'beta': 4},
            'VDF_73.0': {'alpha': 0.15, 'beta': 4},
            'VDF_74.0': {'alpha': 0.15, 'beta': 4},
            'VDF_75.0': {'alpha': 0.15, 'beta': 4},
            'VDF_76.0': {'alpha': 0.15, 'beta': 4}
        }
    }
# You can then import these in your main script:
# from config import TrainingConfig, ModelConfig, Directories
