#%%
import geopandas as gpd # No se usa directamente en el núcleo del PINN
import pandas as pd
import logging
import torch
import numpy as np
import networkx as nx # No se usa directamente en el núcleo del PINN
from itertools import product # No se usa directamente en el núcleo del PINN
import os
from collections import defaultdict
import pickle
import csv # No se usa directamente en el núcleo del PINN
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from pathlib import Path
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Constantes para padding
PADDING_LINK_IDX = -1
MAX_ROUTES_PER_OD = 10
MAX_LINKS_PER_ROUTE = 128 # Definido por el usuario

#%%
# Rutas
folder_pickle: str = Path("C:/Users/SEJC94056/Documents/AADT_CodeProject/GAT-Applied/data")
filename_pickle: str = "network_data_final.pkl"

#%%
# Checkpoints
def save_checkpoints_to_file(checkpoints_list, file_path):
    """
    Guarda una lista de diccionarios de checkpoint en un único archivo .pth.

    Args:
        checkpoints_list (list): La lista de checkpoints a guardar.
        file_path (str or Path): La ruta completa del archivo donde se guardarán los checkpoints.
    """
    # Asegurarse de que el directorio de destino exista
    directory = Path(file_path).parent
    directory.mkdir(parents=True, exist_ok=True)
    
    # Guardar la lista completa de checkpoints
    torch.save(checkpoints_list, file_path)
    # print(f"Lista de checkpoints guardada en: {file_path}")

def load_checkpoint(model_path, model, optimizer, epoch_to_load='last'):
    """
    Carga un checkpoint desde un archivo .pth que contiene una lista de checkpoints.

    Args:
        model_path (str or Path): Ruta al archivo .pth.
        model (torch.nn.Module): Una instancia del modelo con la misma arquitectura que el modelo guardado.
        optimizer (torch.optim.Optimizer): Una instancia del optimizador.
        epoch_to_load (int or 'last'): El número de epoch a cargar (basado en el índice 0)
                                     o 'last' para cargar el último checkpoint disponible.

    Returns:
        tuple: Una tupla conteniendo:
            - epoch (int): El número del epoch cargado.
            - histories (dict): Diccionario con los historiales de pérdida y parámetros.
            - model_init_config (dict): Diccionario con los parámetros de inicialización del modelo.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Archivo de checkpoint no encontrado en: {model_path}")

    # Cargar la lista completa de checkpoints (en CPU para evitar problemas de compatibilidad)
    all_checkpoints = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)

    if not all_checkpoints:
        raise ValueError("El archivo de checkpoint está vacío.")

    # Seleccionar el checkpoint deseado
    if epoch_to_load == 'last':
        checkpoint = all_checkpoints[-1]
    else:
        # Buscar el checkpoint por el número de epoch
        # Nota: 'epoch' en el checkpoint es el índice (ej. 0 para el primer epoch)
        checkpoint = next((ckpt for ckpt in all_checkpoints if ckpt['epoch'] == epoch_to_load), None)
        if checkpoint is None:
            raise ValueError(f"No se encontró el checkpoint para el epoch {epoch_to_load}.")

    # Cargar el estado del modelo y del optimizador
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # Extraer la información adicional
    epoch = checkpoint['epoch']
    histories = checkpoint['histories']
    model_init_config = checkpoint['model_init_config']

    print(f"Checkpoint de epoch {epoch + 1} cargado correctamente.")
    
    return epoch, histories, model_init_config

#%%
# Data Loading
## Estructura de Datos 
class DTAData:
    def __init__(self, original_data_pickle_path, device=None,
                pen_rate: bool=False, day_to_hour: bool=False, modal_choice: bool=False,
                numerical_scale_factor= 1.0):
        
        self.device = device or torch.device("cpu")
        self.numerical_scale_factor = numerical_scale_factor
        self.pen_rate = pen_rate
        if self.pen_rate:
            self.adj_pen_rate = 1.0 / 0.36
        else:
            self.adj_pen_rate = 1.0
        self.day_to_hour = day_to_hour
        if self.day_to_hour:
            self.adj_day_to_hour = 1.0 / 24.0
        else:
            self.adj_day_to_hour = 1
        self.modal_choice = modal_choice
        if self.modal_choice:
            self.adj_mode = 0.4
        else:
            self.adj_mode = 1.0

        self.load_original_data(original_data_pickle_path)
        self.preprocess_data()


    def load_original_data(self, pickle_path):
        print(f"Cargando datos originales de: {pickle_path}")
        with open(pickle_path, "rb") as f:
            original_data = pickle.load(f)

        self.edge_index = original_data.edge_index.to(self.device)
        self.ffs = original_data.ffs.to(self.device)
        self.length = original_data.length.to(self.device)
        if self.adj_day_to_hour:
            self.capacity = original_data.capacity.to(self.device)
        else:
            self.capacity = original_data.capacity.to(self.device) * 16
        self.link_group = original_data.link_group.to(self.device)
        self.true_counts = original_data.true_counts.to(self.device)
        self.num_links = original_data.num_links
        self.t0 = original_data.t0.to(self.device)
        self.num_groups = original_data.num_groups
        # Nodos
        self.node_ids = original_data.node_ids
        self.node_types = original_data.node_types
        self.zone_node_ids = original_data.zone_node_ids
        
        self.raw_known_od = original_data.known_od.cpu()
        self.od_pair_masks = original_data.odpairs_mask
        self.raw_routes = original_data.routes
        self.all_od_pairs_list = original_data.all_od_pairs_list
        self.num_od = len(self.all_od_pairs_list)
        
        # Asegurar que all_od_pairs_list (que se usa como self.od_pairs en el código original)
        # esté disponible para las funciones de visualización.
        self.od_pairs = self.all_od_pairs_list

        print(f"Datos originales cargados: {self.num_od} pares OD, {self.num_links} enlaces.")
        print(f"Forma de raw_known_od: {self.raw_known_od.shape}")

      # -------- Ajuste de toma de datos de conteo -------- 
    
    def _counts_integration(self):
        counts_tensor = self.true_counts
        # Máscara True donde no es nan
        mask = ~torch.isnan(counts_tensor)
        # Máximo por fila, remplazando nan por -inf para no afectar el máximo
        tensor_filled = torch.where(torch.isnan(counts_tensor), torch.tensor(float('-inf')), counts_tensor)
        max_vals, _ = torch.max(tensor_filled, dim=1, keepdim=True)
        # Donde el máximo es -inf (todas nan), ponemos nan
        max_vals[max_vals == float('-inf')] = float('nan')
        # Máscara True si hay al menos un valor válido por fila
        mask_any = mask.any(dim=1, keepdim=True)
        return max_vals, mask_any 

    def preprocess_data(self):
        
        # Ajustments
        self.true_counts = self.true_counts * self.adj_day_to_hour
        self.raw_known_od = self.raw_known_od * self.adj_pen_rate * self.adj_day_to_hour * self.adj_mode 

        # Ajuste numérico
        self.true_counts *= self.numerical_scale_factor
        self.raw_known_od *= self.numerical_scale_factor
         #self.capacity *= self.numerical_scale_factor # depronto esto lo tengo que deshacer

        # Padding process

        processed_routes_for_padding = []
        for p_idx in range(self.num_od):
            od_routes = self.raw_routes[p_idx] if p_idx < len(self.raw_routes) else []
            if not isinstance(od_routes, list):
                od_routes = []


            if not od_routes:
                current_p_routes = [[] for _ in range(MAX_ROUTES_PER_OD)]
            else:
                current_p_routes = list(od_routes)
                current_p_routes = current_p_routes[:MAX_ROUTES_PER_OD]
                if current_p_routes:
                    last_route = current_p_routes[-1]
                    while len(current_p_routes) < MAX_ROUTES_PER_OD:
                        current_p_routes.append(list(last_route))
                else:
                    current_p_routes = [[] for _ in range(MAX_ROUTES_PER_OD)]
            processed_routes_for_padding.append(current_p_routes)


        self.padded_routes_tensor = torch.full(
            (self.num_od, MAX_ROUTES_PER_OD, MAX_LINKS_PER_ROUTE),
            PADDING_LINK_IDX, dtype=torch.int32, device=self.device
        )
        self.routes_padding_mask = torch.zeros(
            (self.num_od, MAX_ROUTES_PER_OD, MAX_LINKS_PER_ROUTE),
            dtype=torch.bool, device=self.device
        )


        for p_idx in range(self.num_od):
            for r_idx in range(MAX_ROUTES_PER_OD):
                if p_idx < len(processed_routes_for_padding) and \
                    r_idx < len(processed_routes_for_padding[p_idx]):
                    route_link_indices = processed_routes_for_padding[p_idx][r_idx]
                    if route_link_indices:
                        num_links_in_route = len(route_link_indices)
                        links_to_fill = route_link_indices[:MAX_LINKS_PER_ROUTE]
                        actual_len = len(links_to_fill)
                        
                        self.padded_routes_tensor[p_idx, r_idx, :actual_len] = torch.tensor(
                            links_to_fill, dtype=torch.int32, device=self.device
                        )
                        self.routes_padding_mask[p_idx, r_idx, :actual_len] = True
        
        # del self.raw_routes

        num_total_days = self.raw_known_od.shape[0]
        train_days = int(0.7 * num_total_days)
        val_days = int(0.15 * num_total_days)
        
        self.known_od_train = self.raw_known_od[:train_days, :, :].sum(dim=1).mean(dim=0).to(self.device)
        self.known_od_val = self.raw_known_od[train_days:train_days+val_days, :, :].sum(dim=1).mean(dim=0).to(self.device)
        self.known_od_test = self.raw_known_od[train_days+val_days:, :, :].sum(dim=1).mean(dim=0).to(self.device)

        self.mask_od_train = self.od_pair_masks["taz_to_taz"]
        self.mask_od_val = self.od_pair_masks["taz_to_taz"]
        self.mask_od_test = self.od_pair_masks["taz_to_taz"]

        # self.mask_od_train = self.known_od_train[self.od_pair_masks["taz_to_taz"]]
        # self.mask_od_val = self.known_od_val[self.od_pair_masks["taz_to_taz"]]
        # self.mask_od_test = self.known_od_test[self.od_pair_masks["taz_to_taz"]]
        
        # self.mask_od_train = (~torch.isnan(self.known_od_train)).to(self.device)
        # self.mask_od_val = (~torch.isnan(self.known_od_val)).to(self.device)
        # self.mask_od_test = (~torch.isnan(self.known_od_test)).to(self.device)
        
        # self.obs_mask = (~torch.isnan(self.true_counts)).to(self.device)
        # -------- Ajuste de toma de datos de conteo -------- 
        self.true_counts, self.obs_counts_mask = self._counts_integration()

        temp_known_demands = self.known_od_train[self.mask_od_train]
        # temp_known_demands = self.mask_od_train

        initial_estimate_for_unknown = temp_known_demands.mean() if temp_known_demands.numel() > 0 else torch.tensor(1.0, device=self.device)
        
        self.reference_demands_for_sorting = self.known_od_train.clone()
        # self.reference_demands_for_sorting[~self.mask_od_train] = initial_estimate_for_unknown
        self.reference_demands_for_sorting[~self.mask_od_train] = initial_estimate_for_unknown
        
        # Lógica de identificadores de grupo
        unique_group_ids = torch.unique(self.link_group)
        self.num_actual_link_groups = len(unique_group_ids) # Esto debería ser 5 en tu caso
        # Crear el mapeo
        self.group_id_to_idx_map = {gid.item(): i for i, gid in enumerate(unique_group_ids)}
        # Crear un tensor con los índices 0 a N-1 para cada enlace
        self.link_group_mapped_indices = torch.empty_like(self.link_group, dtype=torch.long)
        for original_gid, mapped_idx in self.group_id_to_idx_map.items():
            self.link_group_mapped_indices[self.link_group == original_gid] = mapped_idx
        self.link_group_mapped_indices = self.link_group_mapped_indices.to(self.device)
        
        print("Preprocesamiento de datos completado.")
        print(f"Forma de padded_routes_tensor: {self.padded_routes_tensor.shape}")
        print(f"Forma de known_od_train: {self.known_od_train.shape}")
#%%
#  Model Definition

class RouteChoiceNN(nn.Module):
   def __init__(self, num_routes_for_od, hidden_dim=4):
       super().__init__()
       self.num_routes = num_routes_for_od
       if self.num_routes > 0:
           self.network = nn.Sequential(
               nn.Linear(1, hidden_dim), nn.Tanh(),
               nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
               nn.Linear(hidden_dim, num_routes_for_od)
           )
       else:
           self.network = nn.Identity()


   def forward(self, t):
       if self.num_routes == 0:
           return torch.empty(t.shape[0], 0, device=t.device)
       logits = self.network(t)
       if self.num_routes > 1:
           return torch.softmax(logits, dim=-1)
       elif self.num_routes == 1:
           return torch.ones_like(logits)
       return torch.empty(t.shape[0], 0, device=t.device)

class PINN_DTA_OD(nn.Module):
    def __init__(self, data_container: DTAData, hidden_dim_x=4, initial_alpha_dyn=0.1,
                 initial_unknown_demand_log_val=0.0, bpr_alpha_param=0.15, bpr_beta_param=4.0, numerical_scale_factor=1.0):
        super().__init__()
        # Store the whole container and device
        self.data_container = data_container
        self.device = data_container.device
        
        # Store tensors directly in the model
        self.num_od = data_container.num_od
        self.num_links = data_container.num_links
        self.padded_routes_tensor = data_container.padded_routes_tensor
        self.routes_padding_mask = data_container.routes_padding_mask
        self.t0 = data_container.t0
        self.capacity = data_container.capacity
        self.link_group_mapped_indices = data_container.link_group_mapped_indices

        # --- BPR Parameters ---
        num_link_groups_for_params = data_container.num_actual_link_groups
        self.raw_bpr_alphas = nn.Parameter(torch.zeros(num_link_groups_for_params, device=self.device))
        self.raw_bpr_betas = nn.Parameter(torch.zeros(num_link_groups_for_params, device=self.device))
        self.alpha_min, self.alpha_max = 0.05, 0.20
        self.beta_min, self.beta_max = 2.0, 10.0

        # --- Route Choice Models ---
        self.route_choice_models = nn.ModuleList()
        for _ in range(self.num_od):
            self.route_choice_models.append(RouteChoiceNN(MAX_ROUTES_PER_OD, hidden_dim_x))

        # --- Unknown Demands Parameters ---
        self.log_unknown_demands_params = nn.ParameterList()
        self.unknown_od_indices_global = []
        for i in range(self.num_od):
            if not data_container.mask_od_train[i]: # The training mask defines what is "unknown"
                self.unknown_od_indices_global.append(i)
                self.log_unknown_demands_params.append(
                    nn.Parameter(torch.tensor(initial_unknown_demand_log_val, dtype=torch.float32))
                )
        
        self.alpha_dyn_param = nn.Parameter(torch.tensor(initial_alpha_dyn, dtype=torch.float32))

    def get_current_demands(self, dataset_type: str):
        """Gets demands based on the dataset type (train, val, test)."""
        if dataset_type == "train":
            known_od_ref = self.data_container.known_od_train
        elif dataset_type == "val":
            known_od_ref = self.data_container.known_od_val
        else: # test
            known_od_ref = self.data_container.known_od_test
            
        demands_full = known_od_ref.clone().to(dtype=torch.float32)
        
        # The learned parameters for unknown demands are always applied.
        current_log_param_idx = 0
        for global_idx in self.unknown_od_indices_global:
            if global_idx < len(demands_full):
                 demands_full[global_idx] = torch.exp(self.log_unknown_demands_params[current_log_param_idx])
            current_log_param_idx += 1
            
        return demands_full

    def _get_route_proportions_at_t(self, p_global_idx, t_tensor):
        return self.route_choice_models[p_global_idx](t_tensor)

    def _bpr_travel_time(self, link_flows_v_e, link_t0, link_capacity, link_group_mapped_indices):
        transformed_alphas_per_group, transformed_betas_per_group = self.get_transformed_bpr_params()
        alpha_for_links = transformed_alphas_per_group[link_group_mapped_indices]
        beta_for_links = transformed_betas_per_group[link_group_mapped_indices]
        alpha_for_links_expanded = alpha_for_links.unsqueeze(0)
        beta_for_links_expanded = beta_for_links.unsqueeze(0)
        ratio = link_flows_v_e / (link_capacity + 1e-6)
        return link_t0 * (1 + alpha_for_links_expanded * torch.pow(ratio, beta_for_links_expanded))

    def forward(self, t_collocation, dataset_type: str):
        """
        The forward pass no longer needs od_subset_indices.
        It operates on all OD pairs by default.
        """
        N_t = t_collocation.shape[0]
        
        # The specific demands (train, val, or test) are fetched based on the dataset_type
        current_demands_d_p = self.get_current_demands(dataset_type)

        x_ap_t_list = []
        dx_ap_dt_list = []

        # This loop now iterates over all ODs
        for p_global_idx in range(self.num_od):
            x_p_t = self._get_route_proportions_at_t(p_global_idx, t_collocation)
            x_ap_t_list.append(x_p_t)

            dx_p_dt_route_list = []
            if x_p_t.requires_grad:
                for r_idx in range(MAX_ROUTES_PER_OD):
                    grad_outputs = torch.ones_like(x_p_t[:, r_idx])
                    dx_pr_dt = torch.autograd.grad(
                        outputs=x_p_t[:, r_idx], inputs=t_collocation,
                        grad_outputs=grad_outputs, create_graph=True, retain_graph=True,
                    )[0]
                    if dx_pr_dt is None: dx_pr_dt = torch.zeros_like(t_collocation.squeeze()).unsqueeze(1)
                    dx_p_dt_route_list.append(dx_pr_dt)
            else:
                for _ in range(MAX_ROUTES_PER_OD):
                    dx_p_dt_route_list.append(torch.zeros(N_t, 1, device=t_collocation.device))
            
            if dx_p_dt_route_list:
                dx_ap_dt_list.append(torch.cat(dx_p_dt_route_list, dim=1))
            else: # Should not happen with padding
                dx_ap_dt_list.append(torch.empty(N_t, 0, device=t_collocation.device))

        x_ap_t = torch.stack(x_ap_t_list, dim=0)
        dx_ap_dt = torch.stack(dx_ap_dt_list, dim=0)

        # Operations now use the full tensors stored in self
        flow_on_routes_pr_t = current_demands_d_p.unsqueeze(1).unsqueeze(2) * x_ap_t
        link_flows_v_e_t = torch.zeros(N_t, self.num_links, device=t_collocation.device)

        for t_idx in range(N_t):
            flow_on_routes_at_t = flow_on_routes_pr_t[:, t_idx, :]
            expanded_flows = flow_on_routes_at_t.unsqueeze(2).expand(-1, -1, MAX_LINKS_PER_ROUTE)
            valid_link_indices_mask = (self.padded_routes_tensor != PADDING_LINK_IDX) & self.routes_padding_mask
            flows_to_add = expanded_flows[valid_link_indices_mask]
            link_indices_for_add = self.padded_routes_tensor[valid_link_indices_mask]
            if flows_to_add.numel() > 0:
                link_flows_v_e_t[t_idx, :].index_add_(0, link_indices_for_add.long(), flows_to_add)

        link_travel_times_t_e_t = self._bpr_travel_time(
            link_flows_v_e_t, self.t0.unsqueeze(0),
            self.capacity.unsqueeze(0), self.link_group_mapped_indices
        )

        safe_padded_routes = torch.where(self.padded_routes_tensor == PADDING_LINK_IDX,
                                           torch.zeros_like(self.padded_routes_tensor),
                                           self.padded_routes_tensor).long()
        T_ap_t_route_list = []
        for t_idx in range(N_t):
            link_times_at_t = link_travel_times_t_e_t[t_idx, :]
            gathered_link_times_for_routes = link_times_at_t[safe_padded_routes]
            masked_gathered_times = gathered_link_times_for_routes * self.routes_padding_mask.float()
            route_travel_times_at_t = torch.sum(masked_gathered_times, dim=2)
            T_ap_t_route_list.append(route_travel_times_at_t)

        T_ap_t = torch.stack(T_ap_t_route_list, dim=1)
        
        num_active_links_per_route = self.routes_padding_mask.sum(dim=2)
        is_invalid_route = (num_active_links_per_route == 0).unsqueeze(1)
        T_ap_t_with_inf = torch.where(
            is_invalid_route, torch.full_like(T_ap_t, float('inf')), T_ap_t
        )
        min_val, _ = torch.min(T_ap_t_with_inf, dim=2, keepdim=True)
        
        return {
            "x_ap_t": x_ap_t, "dx_ap_dt": dx_ap_dt,
            "T_ap_t": T_ap_t, "T_p_min_t": min_val,
            "v_e_t": link_flows_v_e_t, "d_p_estimated": current_demands_d_p
        }
        
    def get_transformed_bpr_params(self):
        transformed_alphas = self.alpha_min + (self.alpha_max - self.alpha_min) * torch.sigmoid(self.raw_bpr_alphas)
        transformed_betas = self.beta_min + (self.beta_max - self.beta_min) * torch.sigmoid(self.raw_bpr_betas)
        return transformed_alphas, transformed_betas

#%%
# Loss Function Computation
def compute_pinn_loss(pinn_model: PINN_DTA_OD, forward_outputs,
                      t_collocation, loss_weights_dict,
                      data_container: DTAData,
                      current_dataset_type: str):
    
    alpha_dynamics = pinn_model.alpha_dyn_param
    # The keys in forward_outputs are now for the full dataset, not a subset
    x_ap_t = forward_outputs["x_ap_t"]
    dx_ap_dt = forward_outputs["dx_ap_dt"]
    T_ap_t = forward_outputs["T_ap_t"]
    T_p_min_t = forward_outputs["T_p_min_t"]
    v_e_t = forward_outputs["v_e_t"]
    d_p_estimated = forward_outputs["d_p_estimated"]
    
    device = t_collocation.device

    # Physics Loss (L_phys)
    diff_times = T_p_min_t - T_ap_t
    diff_times = torch.where(
        torch.isinf(diff_times) | torch.isnan(diff_times),
        torch.zeros_like(diff_times),
        diff_times
    )
    ode_rhs = alpha_dynamics * x_ap_t * diff_times
    residual = dx_ap_dt - ode_rhs
    active_routes_mask = (data_container.routes_padding_mask.sum(dim=2) > 0)
    active_routes_mask_expanded = active_routes_mask.unsqueeze(1)
    L_phys = torch.mean((residual[active_routes_mask_expanded.expand_as(residual)] ** 2)) \
        if active_routes_mask_expanded.any() else torch.tensor(0.0, device=device)

    # Initial Condition Loss (L_ic)
    t0_indices = (t_collocation.squeeze() == 0.0).nonzero(as_tuple=True)[0]
    if t0_indices.numel() > 0:
        t0_idx = t0_indices[0]
        x_at_t0 = x_ap_t[:, t0_idx, :]
        num_active_routes_for_ods = active_routes_mask.sum(dim=1).float()
        mask_active = (num_active_routes_for_ods > 0).unsqueeze(1)
        target_x_val = torch.where(
            mask_active,
            1.0 / num_active_routes_for_ods.unsqueeze(1),
            torch.zeros_like(num_active_routes_for_ods.unsqueeze(1)))
        target_x_at_t0 = target_x_val.expand_as(x_at_t0)
        ic_loss_terms_unmasked = (x_at_t0 - target_x_at_t0) ** 2
        L_ic = torch.mean(ic_loss_terms_unmasked[active_routes_mask]) \
            if active_routes_mask.any() else torch.tensor(0.0, device=device)
    else:
        L_ic = torch.tensor(0.0, device=device)

    # Data Losses (L_od, L_link)
    if current_dataset_type == "train":
        known_od_ref = data_container.known_od_train
        mask_od_ref = data_container.mask_od_train
    elif current_dataset_type == "val":
        known_od_ref = data_container.known_od_val
        mask_od_ref = data_container.mask_od_val
    else: # test
        known_od_ref = data_container.known_od_test
        mask_od_ref = data_container.mask_od_test

    if mask_od_ref.any():
        L_od = torch.mean((d_p_estimated[mask_od_ref] - known_od_ref[mask_od_ref]) ** 2)
    else:
        L_od = torch.tensor(0.0, device=device)

    L_link = torch.tensor(0.0, device=device)
    obs_link_mask = data_container.obs_counts_mask
    if obs_link_mask.any():
        sorted_t, sort_indices = torch.sort(t_collocation.squeeze().detach())
        sorted_v_e_t_observed_links = v_e_t[sort_indices, :][:, obs_link_mask.squeeze(1)]
        if sorted_v_e_t_observed_links.numel() > 0 and sorted_t.numel() > 1:
            integral_v_e_approx = torch.trapezoid(sorted_v_e_t_observed_links, sorted_t, dim=0)
            observed_true_counts = data_container.true_counts[obs_link_mask]
            # L_link = torch.mean((integral_v_e_approx - observed_true_counts) ** 2)
            log_pred = torch.log(integral_v_e_approx + 1.0)
            log_true = torch.log(observed_true_counts + 1.0)
            L_link = torch.mean((log_pred - log_true) ** 2)

    # Regularization Loss (L_reg)
    unknown_mask = ~mask_od_ref
    if unknown_mask.any():
        L_reg = torch.mean(d_p_estimated[unknown_mask] ** 2)
    else:
        L_reg = torch.tensor(0.0, device=device)
    
    total_loss = (L_phys * loss_weights_dict.get('phys', 1.0) +
                  L_ic * loss_weights_dict.get('ic', 1.0) +
                  L_od * loss_weights_dict.get('od', 1.0) +
                  L_link * loss_weights_dict.get('link', 1.0) +
                  L_reg * loss_weights_dict.get('reg', 1.0))
    return {
        "total": total_loss, "phys": L_phys, "ic": L_ic,
        "od": L_od, "link": L_link, "reg": L_reg
    }

#%%
# Función de Entrenamiento

def train_model(
    pinn_model: PINN_DTA_OD, 
    data_container: DTAData, 
    model_init_config: dict,
    epochs=200, 
    lr=0.005,
    num_t_collocation_pts=20, 
    loss_weights=None,
    save_path="best_pinn_model.pth",
    starting_epoch=0,
    loss_history_train=None,
    loss_history_val=None,
    param_histories=None,
    save_every=1
):
    if loss_weights is None:
        loss_weights = {'phys': 1.0, 'ic': 1.0, 'od': 5.0, 'link': 10.0, 'reg': 0.01}

    optimizer = torch.optim.Adam(pinn_model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    device = data_container.device
    pinn_model.to(device)
    
    # Cargar checkpoints previos si se está reanudando un entrenamiento
    checkpoints_list = []
    if starting_epoch > 0 and os.path.exists(save_path):
        print(f"Cargando checkpoints existentes desde {save_path} para continuar.")
        checkpoints_list = torch.load(save_path, map_location=device)

    # Inicializar historiales si no se proporcionan (entrenamiento desde cero)
    if loss_history_train is None:
        loss_history_train = {name: [] for name in ["total", "phys", "ic", "od", "link", "reg"]}
    if loss_history_val is None:
        loss_history_val = {"total": []}
    if param_histories is None:
        param_histories = {"bpr_alphas": [], "bpr_betas": []}

    print(f"Iniciando/continuando entrenamiento en {device} desde el epoch {starting_epoch + 1}...")
    
    t_collocation_points = torch.linspace(0, 1, num_t_collocation_pts, device=device, dtype=torch.float32)
    t_collocation_fwd = t_collocation_points.unsqueeze(1).requires_grad_(True)

    for epoch in range(starting_epoch, starting_epoch + epochs):
        pinn_model.train()
        optimizer.zero_grad()
        
        forward_pass_results = pinn_model(t_collocation_fwd, dataset_type="train")
        current_losses = compute_pinn_loss(pinn_model, forward_pass_results, t_collocation_fwd, loss_weights, data_container, "train")
        total_loss_val = current_losses["total"]

        if torch.isnan(total_loss_val) or torch.isinf(total_loss_val):
            print(f"\nEpoch {epoch+1} | NaN/Inf en training loss. Deteniendo.")
            break

        total_loss_val.backward()
        optimizer.step()

        # Actualizar historiales
        loss_history_train["total"].append(total_loss_val.item())
        for name in ["phys", "ic", "od", "link", "reg"]:
            loss_history_train[name].append(current_losses[name].item())
        
        # Validación
        pinn_model.eval()
        with torch.no_grad():
            val_results = pinn_model(t_collocation_fwd.detach().requires_grad_(False), dataset_type="val")
            val_losses = compute_pinn_loss(pinn_model, val_results, t_collocation_fwd.detach(), loss_weights, data_container, "val")
            val_loss = val_losses["total"].item()
        
        loss_history_val["total"].append(val_loss)
        scheduler.step(val_loss)
        
        # Historial de parámetros BPR
        alphas, betas = pinn_model.get_transformed_bpr_params()
        param_histories["bpr_alphas"].append(alphas.detach().cpu().numpy())
        param_histories["bpr_betas"].append(betas.detach().cpu().numpy())

        print(f"\rEpoch {epoch+1}/{starting_epoch + epochs} | Train Loss: {total_loss_val.item():.4e} | Val Loss: {val_loss:.4e}", end="")
        
        # Guardar checkpoint
        if epoch % save_every == 0:
            checkpoint_data = {
                'epoch': epoch,
                'model_state_dict': pinn_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'histories': {
                    'loss_train': {k: v.copy() for k, v in loss_history_train.items()},
                    'loss_val': {k: v.copy() for k, v in loss_history_val.items()},
                    'params': {k: v.copy() for k, v in param_histories.items()}
                },
                'model_init_config': model_init_config
            }
            # Eliminar checkpoint anterior del mismo epoch si existe (al reanudar)
            checkpoints_list = [ckpt for ckpt in checkpoints_list if ckpt.get('epoch') != epoch]
            checkpoints_list.append(checkpoint_data)

            save_checkpoints_to_file(checkpoints_list, save_path)

            if epoch % 10 == 0:
                t_eval = torch.tensor([1.0], device=device) # Evaluar en el punto de tiempo t=1.0
                print_link_flow_comparison_standalone(
                    model_instance=pinn_model,
                    data_container=dta_data_container,
                    t_eval_tensor=t_eval,
                    output_dir=MAIN_OUTPUT_DIR,
                    save_outputs=False, # Guardar tanto la tabla como el gráfico
)
    
    print("\nEntrenamiento finalizado.")
    return pinn_model, loss_history_train, loss_history_val, param_histories

#%%
# Evaluación y Visualización

import pandas as pd
import matplotlib.pyplot as plt
import torch
import numpy as np
from pathlib import Path
import os
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# --- Funciones Auxiliares ---

def _ensure_dir(directory_path):
    """Asegura que un directorio exista, creándolo si es necesario."""
    Path(directory_path).mkdir(parents=True, exist_ok=True)

def _load_model_if_path(model_instance, model_path, device, data_container_for_init, model_params_for_init):
    
    """Carga un modelo desde model_path si model_instance es None."""
    
    if model_instance is None:

        if model_path is None:
            raise ValueError("Se debe proporcionar model_instance o model_path.")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Archivo de modelo no encontrado en: {model_path}")
        
        print(f"Cargando modelo desde: {model_path}")
        # Cargar la lista de checkpoints y tomar el último
        all_checkpoints = torch.load(model_path, map_location=device, weights_only=False)
        if not all_checkpoints:
            raise ValueError("El archivo de checkpoints está vacío.")
        checkpoint_to_load = all_checkpoints[-1]

        # Usar la configuración guardada para instanciar el modelo
        loaded_model_config = checkpoint_to_load.get('model_init_config', model_params_for_init)
        if loaded_model_config is None:
            raise ValueError("No se encontró la configuración de inicialización en el checkpoint.")

        loaded_model = PINN_DTA_OD(
            data_container=data_container_for_init,
            **loaded_model_config
        )
        loaded_model.load_state_dict(checkpoint_to_load['model_state_dict'])
        loaded_model.to(device)
        loaded_model.eval()
        return loaded_model
        
    model_instance.eval()
    return model_instance

# --- Funciones de Visualización y Evaluación (Actualizadas) ---
def plot_loss_progression_standalone(loss_history_train, loss_history_val, epochs_ran,
                                     output_dir="outputs", save_plot=False):
    """Grafica la progresión de pérdidas y opcionalmente guarda el gráfico."""
    plt.figure(figsize=(12, 8))
    
    for loss_name, history in loss_history_train.items():
        if history:
            clean_history = [h for h in history if pd.notna(h) and np.isfinite(h) and h > 0]
            epochs_for_plot = [i + 1 for i, h in enumerate(history) if pd.notna(h) and np.isfinite(h) and h > 0]
            if clean_history and epochs_for_plot:
                plt.plot(epochs_for_plot, clean_history, label=f"Train {loss_name}")

    if loss_history_val and loss_history_val.get("total"):
        clean_val_history = [h for h in loss_history_val["total"] if pd.notna(h) and np.isfinite(h) and h > 0]
        epochs_for_val_plot = [i + 1 for i, h in enumerate(loss_history_val["total"]) if pd.notna(h) and np.isfinite(h) and h > 0]
        if clean_val_history and epochs_for_val_plot:
            plt.plot(epochs_for_val_plot, clean_val_history, label="Validation Total Loss", linestyle='--')

    plt.xlabel("Epoch")
    plt.ylabel("Log(Loss)")
    plt.title("Progresión de Pérdidas (Entrenamiento y Validación)")
    try:
        plt.yscale('log')
    except ValueError:
        print("Advertencia: No se pudo establecer la escala logarítmica.")
        plt.yscale('linear')
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.tight_layout()
    
    if save_plot:
        _ensure_dir(output_dir)
        save_path = Path(output_dir) / "loss_progression.png"
        plt.savefig(save_path, dpi=300)
        print(f"Gráfico de pérdidas guardado en: {save_path}")
    plt.show()
    plt.close()

def print_od_demand_comparison_standalone(model_instance: PINN_DTA_OD,
                                          data_container: DTAData,
                                          dataset_type="test",
                                          output_dir="outputs", save_table=False,
                                          model_path=None, model_init_params=None,
                                          display_scale='daily'):
    """Compara demandas OD y opcionalmente guarda la tabla en CSV."""
    device = data_container.device
    model = _load_model_if_path(model_instance, model_path, device, data_container, model_init_params)
    
    print(f"\n--- Comparación de Demandas OD ({dataset_type}) en escala {display_scale} ---")

    inverse_numerical_scale = 1.0 / data_container.numerical_scale_factor

    if display_scale == 'daily':
        final_inverse_scale = inverse_numerical_scale 
    elif display_scale == 'hourly':
        inverse_day_hour_scale = 1.0 / data_container.adj_day_to_hour if data_container.day_to_hour else 1.0
        final_inverse_scale = inverse_numerical_scale * inverse_day_hour_scale
    else: # 'scaled'
        final_inverse_scale = inverse_numerical_scale

    with torch.no_grad():
        estimated_demands_all = model.get_current_demands(dataset_type=dataset_type)
        estimated_demands_all_cpu = estimated_demands_all.cpu().numpy()

    if dataset_type == "train":
        known_demands_cpu = data_container.known_od_train.cpu().numpy()
        mask_od_cpu = data_container.mask_od_train.cpu().numpy()
    elif dataset_type == "val":
        known_demands_cpu = data_container.known_od_val.cpu().numpy()
        mask_od_cpu = data_container.mask_od_val.cpu().numpy()
    else: # test
        known_demands_cpu = data_container.known_od_test.cpu().numpy()
        mask_od_cpu = data_container.mask_od_test.cpu().numpy()

    od_data = []
    for i in range(data_container.num_od):
        od_pair_str = f"{data_container.all_od_pairs_list[i][0]}->{data_container.all_od_pairs_list[i][1]}"
        est_demand = estimated_demands_all_cpu[i] * final_inverse_scale
        known_demand_val = known_demands_cpu[i] * final_inverse_scale
        is_known = mask_od_cpu[i]
        od_data.append({
            "Par OD": od_pair_str, "Índice OD": i,
            "Demanda Estimada": f"{est_demand:.2f}",
            "Demanda Conocida (Set Actual)": f"{known_demand_val:.2f}" if is_known else "N/A",
            "Es Conocida (Set Actual)": "Sí" if is_known else "No"
        })
    df = pd.DataFrame(od_data)
    print(df.to_string(index=False))

    if save_table:
        _ensure_dir(output_dir)
        save_path = Path(output_dir) / f"od_demand_comparison_{dataset_type}.csv"
        df.to_csv(save_path, index=False)
        print(f"Tabla de comparación de demandas OD guardada en: {save_path}")
     
    return df

def calculate_regression_metrics_standalone(model_instance: PINN_DTA_OD,
                                            data_container: DTAData,
                                            dataset_type="test",
                                            t_collocation_points_for_eval=None,
                                            output_dir="outputs", save_metrics=False,
                                            model_path=None, model_init_params=None):
    """Calcula R2, MAE, RMSE para demandas OD y flujos en enlaces, y opcionalmente guarda las métricas."""
    device = data_container.device
    model = _load_model_if_path(model_instance, model_path, device, data_container, model_init_params)

    inverse_numerical_scale = 1.0 / data_container.numerical_scale_factor

    metrics_results = {}
    
    # 1. Métricas para Demandas OD
    print(f"\n--- Calculando Métricas de Regresión para Demandas OD ({dataset_type}) ---")
    with torch.no_grad():
        estimated_demands_all = model.get_current_demands(dataset_type=dataset_type)
    
    if dataset_type == "train":
        known_od_ref = data_container.known_od_train
        mask_od_ref = data_container.mask_od_train
    elif dataset_type == "val":
        known_od_ref = data_container.known_od_val
        mask_od_ref = data_container.mask_od_val
    else: # test
        known_od_ref = data_container.known_od_test
        mask_od_ref = data_container.mask_od_test

    y_true_demand = known_od_ref[mask_od_ref].cpu().numpy()
    y_pred_demand = estimated_demands_all[mask_od_ref].cpu().numpy()

    if len(y_true_demand) > 0 and len(y_pred_demand) > 0:
        y_true_demand_rescaled = y_true_demand * inverse_numerical_scale
        y_pred_demand_rescaled = y_pred_demand * inverse_numerical_scale
        r2_od = r2_score(y_true_demand_rescaled, y_pred_demand_rescaled)
        mae_od = mean_absolute_error(y_true_demand_rescaled, y_pred_demand_rescaled)
        rmse_od = np.sqrt(mean_squared_error(y_true_demand_rescaled, y_pred_demand_rescaled))
        metrics_results["OD_Demands"] = {"R2": r2_od, "MAE": mae_od, "RMSE": rmse_od}
        print(f"  Demandas OD - R2: {r2_od:.4f}, MAE: {mae_od:.4f}, RMSE: {rmse_od:.4f}")
    else:
        print("  No hay demandas OD conocidas en este conjunto para calcular métricas.")
        metrics_results["OD_Demands"] = {"R2": np.nan, "MAE": np.nan, "RMSE": np.nan}

    # 2. Métricas para Flujos en Enlaces (Integrados)
    print(f"\n--- Calculando Métricas de Regresión para Flujos en Enlaces ({dataset_type}) ---")
    if t_collocation_points_for_eval is None:
        t_collocation_points_for_eval = torch.linspace(0, 1, 20, device=device, dtype=torch.float32)
    
    t_eval_fwd = t_collocation_points_for_eval.unsqueeze(1).requires_grad_(False)
    
    with torch.no_grad():
        eval_outputs = model(t_eval_fwd, dataset_type=dataset_type)
        v_e_t_eval = eval_outputs["v_e_t"]

    obs_link_mask = data_container.obs_counts_mask
    if obs_link_mask.any() and v_e_t_eval.numel() > 0:
        sorted_t_eval, sort_indices_eval = torch.sort(t_eval_fwd.squeeze())
        sorted_v_e_t_observed_links_eval = v_e_t_eval[sort_indices_eval, :][:, obs_link_mask.squeeze(1)]

        if sorted_v_e_t_observed_links_eval.numel() > 0 and sorted_t_eval.numel() > 1:
            integral_v_e_approx_eval = torch.trapezoid(sorted_v_e_t_observed_links_eval, sorted_t_eval, dim=0)
            
            y_true_link_flow = data_container.true_counts[obs_link_mask].cpu().numpy()
            y_pred_link_flow = integral_v_e_approx_eval.cpu().numpy()

            if len(y_true_link_flow) > 0 and len(y_pred_link_flow) > 0:
                y_true_link_flow_rescaled = y_true_link_flow * inverse_numerical_scale
                y_pred_link_flow_rescaled = y_pred_link_flow * inverse_numerical_scale

                r2_link = r2_score(y_true_link_flow_rescaled, y_pred_link_flow_rescaled)
                mae_link = mean_absolute_error(y_true_link_flow_rescaled, y_pred_link_flow_rescaled)
                rmse_link = np.sqrt(mean_squared_error(y_true_link_flow_rescaled, y_pred_link_flow_rescaled))
                metrics_results["Link_Flows_Integrated"] = {"R2": r2_link, "MAE": mae_link, "RMSE": rmse_link}
                print(f"  Flujos Enlaces (Integrados) - R2: {r2_link:.4f}, MAE: {mae_link:.4f}, RMSE: {rmse_link:.4f}")
    else:
        print("  No hay máscara de observación de enlaces o v_e_t_eval está vacío.")
        metrics_results["Link_Flows_Integrated"] = {"R2": np.nan, "MAE": np.nan, "RMSE": np.nan}

    if save_metrics:
        _ensure_dir(output_dir)
        save_path = Path(output_dir) / f"regression_metrics_{dataset_type}.csv"
        pd.DataFrame.from_records([
            (comp, metric, val) for comp, mets in metrics_results.items() for metric, val in mets.items()
        ], columns=["Componente", "Metrica", "Valor"]).to_csv(save_path, index=False)
        print(f"Métricas de regresión guardadas en: {save_path}")
        
    return metrics_results

def print_link_flow_comparison_standalone(model_instance: PINN_DTA_OD,
                                          data_container: DTAData,
                                          t_eval_tensor: torch.Tensor,
                                          dataset_type="test",
                                          output_dir="outputs", save_outputs=False,
                                          model_path=None, model_init_params=None):
    """Compara flujos en enlaces predichos vs. conocidos."""
    device = data_container.device
    model = _load_model_if_path(model_instance, model_path, device, data_container, model_init_params)
    
    inverse_numerical_scale = 1.0 / data_container.numerical_scale_factor

    print(f"\n--- Flow Comparison per edge (en t={t_eval_tensor.item():.2f}) ---")
    
    with torch.no_grad():
        t_eval_fwd = t_eval_tensor.reshape(1, 1).to(device).requires_grad_(False)
        eval_outputs = model(t_eval_fwd, dataset_type=dataset_type)
        predicted_flows_at_t_eval = eval_outputs["v_e_t"].squeeze().cpu().numpy()
        if predicted_flows_at_t_eval.ndim == 0:
            predicted_flows_at_t_eval = np.array([predicted_flows_at_t_eval.item()])

    link_data_list = []
    edge_index_np = data_container.edge_index.cpu().numpy()
    true_counts_np = data_container.true_counts.cpu().numpy()
    obs_mask_np = data_container.obs_counts_mask.cpu().numpy().squeeze(1)

    for i in range(data_container.num_links):
        pred_flow = predicted_flows_at_t_eval[i] if i < len(predicted_flows_at_t_eval) else float('nan')
        known_flow_val = true_counts_np[i]

        pred_flow_rescaled = pred_flow * inverse_numerical_scale
        known_flow_val_rescaled = known_flow_val * inverse_numerical_scale

        is_observed = obs_mask_np[i]
        topology_str = f"{edge_index_np[0, i]} -> {edge_index_np[1, i]}"
        link_data_list.append({
            "Enlace #": i, "Topology (O->D)": topology_str,
            f"Esimated Flow (t={t_eval_tensor.item():.2f})": f"{pred_flow_rescaled:.2f}",
            "Observed Flow (true_counts)": f"{known_flow_val_rescaled.item():.2f}" if is_observed else "N/A",
            "Measured": "Yes" if is_observed else "No"
        })
    
    df_link_flows = pd.DataFrame(link_data_list)
    print(df_link_flows.to_string(index=False))
    
    if save_outputs:
        _ensure_dir(output_dir)
        df_link_flows.to_csv(Path(output_dir) / f"link_flow_comparison_t{t_eval_tensor.item():.2f}.csv", index=False)

    if np.any(obs_mask_np) and len(predicted_flows_at_t_eval) == data_container.num_links:
       true_obs_flows = true_counts_np[obs_mask_np]
       pred_obs_flows = predicted_flows_at_t_eval[obs_mask_np]

       true_obs_flows_rescaled = true_obs_flows * inverse_numerical_scale
       pred_obs_flows_rescaled = pred_obs_flows * inverse_numerical_scale

       if len(true_obs_flows_rescaled) > 0 and len(pred_obs_flows_rescaled) > 0:
           plt.figure(figsize=(6, 6))
           plt.scatter(true_obs_flows_rescaled, pred_obs_flows_rescaled, alpha=0.7, label=f"Flujos Observados vs Predichos (t={t_eval_tensor.item():.2f})")
          
           valid_true_flows = true_obs_flows_rescaled[~np.isnan(true_obs_flows_rescaled)]
           valid_pred_flows = pred_obs_flows_rescaled[~np.isnan(pred_obs_flows_rescaled)]


           if len(valid_true_flows) > 0 and len(valid_pred_flows) > 0:
               min_val = min(np.min(valid_true_flows), np.min(valid_pred_flows))
               max_val = max(np.max(valid_true_flows), np.max(valid_pred_flows))
               plt.plot([min_val, max_val], [min_val, max_val], 'r--', label="Igualdad Perfecta")
          
           plt.xlabel("Flujo Observado (true_counts)")
           plt.ylabel(f"Flujo Predicho (t={t_eval_tensor.item():.2f})")
           plt.title("Comparación Flujo Predicho vs. Observado (Enlaces Observados)")
           plt.legend()
           plt.grid(True)
           plt.tight_layout()
          
           if save_outputs:
               _ensure_dir(output_dir)
               plot_save_path = Path(output_dir) / f"link_flow_scatter_t{t_eval_tensor.item():.2f}.png"
               plt.savefig(plot_save_path, dpi=300)
               print(f"Gráfico scatter de flujos guardado en: {plot_save_path}")
           plt.show()
           plt.close()
       else:
           print("Advertencia: No hay datos de flujos observados válidos para generar el diagrama de dispersión.")
    elif len(predicted_flows_at_t_eval) != data_container.num_links:
        print(f"Advertencia: El número de flujos predichos ({len(predicted_flows_at_t_eval)}) no coincide con el número de enlaces ({data_container.num_links}).")

def print_link_type_parameters_standalone(model_instance: PINN_DTA_OD,
                                          data_container: DTAData,
                                          output_dir="outputs", save_table=False,
                                          model_path=None, model_init_params=None):
    """Imprime parámetros por tipo de enlace."""
    device = data_container.device
    model = _load_model_if_path(model_instance, model_path, device, data_container, model_init_params)

    inverse_numerical_scale = 1.0 / data_container.numerical_scale_factor

    print("\n--- Parámetros por Tipo de Enlace (con BPR aprendidos) ---")
    with torch.no_grad():
       transformed_alphas, transformed_betas = model.get_transformed_bpr_params()
    
  
    alphas_cpu = transformed_alphas.cpu().numpy()
    betas_cpu = transformed_betas.cpu().numpy()


    link_groups_data_list = []
    capacities_np = data_container.capacity.cpu().numpy()
    link_group_np = data_container.link_group.cpu().numpy() # IDs originales de grupo


    # Usar unique_group_ids y el mapeo para asegurar el orden correcto y la asociación
    for mapped_idx in range(data_container.num_actual_link_groups):
        original_group_id = data_container.link_group_mapped_indices
        original_group_id = data_container.group_id_to_idx_map.get(mapped_idx, f"UnknownMappedIdx{mapped_idx}")
        
        group_mask = (link_group_np == original_group_id)
        group_capacities = capacities_np[group_mask]
        group_capacities_rescaled = group_capacities_rescaled * inverse_numerical_scale

        link_indices_in_group = np.where(group_mask)[0]
        
        alpha_val = alphas_cpu[mapped_idx]
        beta_val = betas_cpu[mapped_idx]


        link_groups_data_list.append({
            "ID Grupo Enlace (Original)":  original_group_id,
            # "Índice Mapeado Param": mapped_idx,
            "Alpha BPR Aprendido": f"{alpha_val:.4f}",
            "Beta BPR Aprendido": f"{beta_val:.3f}",
            "Num. Enlaces en Grupo": len(group_capacities_rescaled),
            "Capacidad Media": f"{np.mean(group_capacities_rescaled):.2f}" if len(group_capacities_rescaled) > 0 else "N/A",
            "Capacidad Min": f"{np.min(group_capacities_rescaled):.2f}" if len(group_capacities_rescaled) > 0 else "N/A",
            "Capacidad Max": f"{np.max(group_capacities_rescaled):.2f}" if len(group_capacities_rescaled) > 0 else "N/A",
            #"Índices de Enlaces (muestra)": ", ".join(map(str, link_indices_in_group[:5])) + ("..." if len(link_indices_in_group) > 5 else "") if len(
            #    link_indices_in_group) > 0 else "N/A"
        })
        
    df_link_groups = pd.DataFrame(link_groups_data_list)
    # Ordenar por ID de Grupo Original para consistencia si es necesario
    if all(isinstance(d["ID Grupo Enlace (Original)"], (int, float)) for d in link_groups_data_list):
        df_link_groups = df_link_groups.sort_values(by="ID Grupo Enlace (Original)")


    print(df_link_groups.to_string(index=False))


    if save_table:
        _ensure_dir(output_dir)
        save_path = Path(output_dir) / "link_type_parameters_with_bpr.csv"
        df_link_groups.to_csv(save_path, index=False)
        print(f"Tabla de parámetros por tipo de enlace (con BPR) guardada en: {save_path}")

def plot_bpr_parameter_evolution_standalone(bpr_alpha_history_per_group, # Lista de arrays numpy
                                           bpr_beta_history_per_group,  # Lista de arrays numpy
                                           epochs_ran: int,
                                           idx_to_group_id_map: dict, # Para leyendas {0: id_orig1, 1: id_orig2}
                                           num_bpr_groups: int,
                                           output_dir="outputs", save_plot=False):
   """
   Grafica la evolución de los parámetros alpha y beta BPR por grupo a lo largo de las epochs.
   """
   if not bpr_alpha_history_per_group or not bpr_beta_history_per_group:
       print("No hay historial de parámetros BPR para graficar.")
       return


   epochs_axis = np.arange(1, epochs_ran + 1)


   # Convertir historial a array numpy para fácil indexación: (epochs, num_groups)
   alpha_history_np = np.array(bpr_alpha_history_per_group)
   beta_history_np = np.array(bpr_beta_history_per_group)


   # Transponer si está en el orden (num_groups, num_epochs)
   if alpha_history_np.shape[0] == num_bpr_groups:
       alpha_history_np = alpha_history_np.T
       beta_history_np = beta_history_np.T


   fig, axs = plt.subplots(2, 1, figsize=(12, 10), sharex=True)


   # Plot Alphas
   for group_idx in range(num_bpr_groups):
       original_group_id = idx_to_group_id_map.get(group_idx, f"GrupoIdx{group_idx}")
       axs[0].plot(epochs_axis, alpha_history_np[:, group_idx], label=f"Alpha Grupo {original_group_id}")
   axs[0].set_title("Evolución de Parámetros Alpha BPR por Grupo")
   axs[0].set_ylabel("Alpha BPR")
   axs[0].legend(loc='center left', bbox_to_anchor=(1, 0.5))
   axs[0].grid(True)


   # Plot Betas
   for group_idx in range(num_bpr_groups):
       original_group_id = idx_to_group_id_map.get(group_idx, f"GrupoIdx{group_idx}")
       axs[1].plot(epochs_axis, beta_history_np[:, group_idx], label=f"Beta Grupo {original_group_id}")
   axs[1].set_title("Evolución de Parámetros Beta BPR por Grupo")
   axs[1].set_xlabel("Epoch")
   axs[1].set_ylabel("Beta BPR")
   axs[1].legend(loc='center left', bbox_to_anchor=(1, 0.5))
   axs[1].grid(True)


   plt.tight_layout(rect=[0, 0, 0.85, 1]) # Ajustar para leyenda


   if save_plot:
       _ensure_dir(output_dir)
       save_path = Path(output_dir) / "bpr_parameter_evolution.png"
       plt.savefig(save_path, dpi=300)
       print(f"Gráfico de evolución de parámetros BPR guardado en: {save_path}")


   plt.show()
   plt.close()

def print_route_flow_progression_standalone(model_instance: 'PINN_DTA_OD',
                                           data_container: 'DTAData',
                                           t_eval_tensor: torch.Tensor,
                                           output_dir="outputs", save_table=False,
                                           model_path=None, model_init_params=None):
    """Muestra el flujo predicho por ruta."""
    device = data_container.device
    model = _load_model_if_path(model_instance, model_path, device, data_container, model_init_params)

    inverse_numerical_scale = 1.0 / data_container.numerical_scale_factor

    print(f"\n--- Flujo Predicho por Ruta (en t={t_eval_tensor.item():.2f}) ---")
    with torch.no_grad():
        t_eval_fwd = t_eval_tensor.reshape(1, 1).to(device).requires_grad_(False)
        eval_outputs = model(t_eval_fwd, dataset_type="test")
        x_ap_at_t_eval = eval_outputs["x_ap_t"][:, 0, :].cpu().numpy()
        current_demands = eval_outputs["d_p_estimated"].cpu().numpy()

    route_flow_data_list = []
    edge_index_np = data_container.edge_index.cpu().numpy()

    for p_idx in range(data_container.num_od):
        od_pair_str = f"{data_container.all_od_pairs_list[p_idx][0]}->{data_container.all_od_pairs_list[p_idx][1]}"
        d_p = current_demands[p_idx]
        
        # Acceder a las rutas desde el padded_routes_tensor y routes_padding_mask
        # padded_routes_tensor: (num_od, MAX_ROUTES, MAX_LINKS_ROUTE)
        # routes_padding_mask: (num_od, MAX_ROUTES, MAX_LINKS_ROUTE)
        
        for r_idx in range(MAX_ROUTES_PER_OD):
            route_link_indices_padded = data_container.padded_routes_tensor[p_idx, r_idx, :].cpu().numpy()
            route_mask_padded = data_container.routes_padding_mask[p_idx, r_idx, :].cpu().numpy()
            
            actual_route_link_indices = route_link_indices_padded[route_mask_padded]


            if not actual_route_link_indices.size > 0: # Ruta vacía o completamente padding
                route_nodes_str = "Vacía/Padding"
                flow_on_route_ap = 0.0
            else:
                x_pr_at_t_eval = x_ap_at_t_eval[p_idx, r_idx]
                flow_on_route_ap = d_p * x_pr_at_t_eval
                flow_on_route_rescaled = flow_on_route_ap * inverse_numerical_scale


                node_sequence = []
                first_link_idx = actual_route_link_indices[0]
                node_sequence.append(edge_index_np[0, first_link_idx])
                for link_idx_in_route in actual_route_link_indices:
                    dest_node = edge_index_np[1, link_idx_in_route]
                    if not node_sequence or node_sequence[-1] != dest_node:
                        node_sequence.append(dest_node)
                route_nodes_str = " -> ".join(map(str, node_sequence)) if node_sequence else "N/A"


            route_flow_data_list.append({
                "Par OD": od_pair_str,
                "Índice Ruta (en padded)": r_idx,
                "Flujo en Ruta (d_p * x_ap)": f"{flow_on_route_rescaled:.3f}",
                "Secuencia Nodos (O->D)": route_nodes_str
            })


    df_route_flows = pd.DataFrame(route_flow_data_list)
    if not df_route_flows.empty:
        print(df_route_flows.to_string(index=False, max_colwidth=60))
        if save_table:
            _ensure_dir(output_dir)
            save_path = Path(output_dir) / f"route_flow_progression_t{t_eval_tensor.item():.2f}.csv"
            df_route_flows.to_csv(save_path, index=False)
            print(f"Tabla de flujo por ruta guardada en: {save_path}")
    else:
        print("No hay datos de flujo de ruta para mostrar.")

def plot_route_choice_proportions_standalone(model_instance: 'PINN_DTA_OD',
                                             data_container: 'DTAData',
                                             od_pair_indices_to_plot: list,
                                             num_t_points=50,
                                             output_dir="outputs", save_plots=False,
                                             model_path=None, model_init_params=None):
    """
    Grafica las proporciones de elección de ruta x_ap(t) para los pares OD seleccionados.

    Args:
        model_instance (PINN_DTA_OD): La instancia del modelo entrenado.
        data_container (DTAData): El contenedor de datos.
        od_pair_indices_to_plot (list): Lista de índices globales de los pares OD a graficar.
        num_t_points (int): Número de puntos de tiempo para la evaluación.
        output_dir (str): Directorio para guardar los gráficos.
        save_plots (bool): Si es True, guarda los gráficos como archivos PNG.
        model_path (str, optional): Ruta al modelo guardado si no se pasa una instancia.
        model_init_params (dict, optional): Parámetros para inicializar el modelo si se carga.
    """
    device = data_container.device
    model = _load_model_if_path(model_instance, model_path, device, data_container, model_init_params)

    print("\n--- Graficando Proporciones de Elección de Ruta x_ap(t) ---")
    t_plot = torch.linspace(0, 1, num_t_points, device=device).unsqueeze(1)

    with torch.no_grad():
        # Ejecutar el forward pass para todos los ODs de una vez
        eval_outputs_full = model(t_plot, dataset_type="test")
        x_ap_t_full = eval_outputs_full["x_ap_t"].cpu().numpy() # Forma: (num_od, num_t_points, MAX_ROUTES_PER_OD)

    edge_index_np_cpu = data_container.edge_index.cpu().numpy()

    for p_global_idx in od_pair_indices_to_plot:
        if p_global_idx >= data_container.num_od:
            print(f"Índice de par OD {p_global_idx} fuera de rango. Omitiendo.")
            continue

        od_pair_str = f"{data_container.all_od_pairs_list[p_global_idx][0]}->{data_container.all_od_pairs_list[p_global_idx][1]}"
        
        # Extraer las proporciones para el OD actual
        x_p_t_plot = x_ap_t_full[p_global_idx] # Forma: (num_t_points, MAX_ROUTES_PER_OD)

        active_routes_mask = data_container.routes_padding_mask[p_global_idx].sum(dim=1).cpu().numpy() > 0

        if not np.any(active_routes_mask):
            print(f"No hay rutas activas definidas para graficar para el par OD {od_pair_str} (índice {p_global_idx}).")
            continue
        
        plt.figure(figsize=(12, 7))
        for r_idx in range(MAX_ROUTES_PER_OD):
            if not active_routes_mask[r_idx]: # No graficar rutas que son solo de padding
                continue

            # Reconstruir la secuencia de nodos para la leyenda
            route_links_padded = data_container.padded_routes_tensor[p_global_idx, r_idx].cpu().numpy()
            route_mask_padded = data_container.routes_padding_mask[p_global_idx, r_idx].cpu().numpy()
            actual_route_indices = route_links_padded[route_mask_padded]
            
            node_sequence_legend = []
            if actual_route_indices.size > 0:
                first_link = actual_route_indices[0]
                node_sequence_legend.append(edge_index_np_cpu[0, first_link])
                for link_idx in actual_route_indices:
                    dest_node = edge_index_np_cpu[1, link_idx]
                    if not node_sequence_legend or node_sequence_legend[-1] != dest_node:
                        node_sequence_legend.append(dest_node)
            
            route_str_legend = " -> ".join(map(str, node_sequence_legend)) if node_sequence_legend else f"Ruta (Padding) {r_idx}"
            
            label_text = f"Ruta {r_idx}: {route_str_legend}"
            plt.plot(t_plot.cpu().numpy().squeeze(), x_p_t_plot[:, r_idx],
                     label=label_text[:80] + "..." if len(label_text) > 80 else label_text)

        plt.xlabel("Tiempo (t)")
        plt.ylabel("Proporción de Ruta (x_ap(t))")
        plt.title(f"Evolución de Proporciones de Ruta para Par OD {od_pair_str} (Índice {p_global_idx})")
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize='small')
        plt.grid(True)
        plt.ylim(-0.05, 1.05)
        plt.tight_layout(rect=[0, 0, 0.75, 1])

        if save_plots:
            _ensure_dir(output_dir)
            save_path = Path(output_dir) / f"route_choice_od_{p_global_idx}.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Gráfico de proporciones de ruta para OD {p_global_idx} guardado en: {save_path}")
        
        plt.show()
        plt.close()

#%%
# --- PREPARACIÓN COMÚN PARA TODOS LOS FLUJOS ---

import pandas as pd
from pathlib import Path

# Configuración de rutas y parámetros
directory_pickle = folder_pickle / filename_pickle

MAIN_OUTPUT_DIR = "C:/Users/SEJC94056/Documents/AADT_CodeProject/GAT-Applied/data/pinn_dta_run_outputs"
model_save_file_path = Path(MAIN_OUTPUT_DIR) / "PINN_03.pth"

device = torch.device("cpu")

numerical_scale_factor = 1e-3
# Instanciar el contenedor de datos
dta_data_container = DTAData(
    original_data_pickle_path=directory_pickle, 
    device=device,
    pen_rate=False, day_to_hour=False, modal_choice=True, 
    numerical_scale_factor=numerical_scale_factor
)

# Configuración de inicialización del modelo (debe ser la misma para cargar y entrenar)
known_demands_train_values = dta_data_container.known_od_train[dta_data_container.mask_od_train]
avg_known_demand = torch.mean(known_demands_train_values) if known_demands_train_values.numel() > 0 else torch.tensor(1.0)
initial_log_demand_val = torch.log(avg_known_demand if avg_known_demand > 0 and avg_known_demand.item() > 0 else torch.tensor(1.0)).item() 

model_init_config = {
    'hidden_dim_x': 4,
    'initial_alpha_dyn': 0.05,
    'initial_unknown_demand_log_val': initial_log_demand_val,
    'numerical_scale_factor': numerical_scale_factor
}

multiplier = 1.0
custom_loss_weights = {
    'phys': 1.0 * multiplier, 
    'ic': 1.0 * multiplier,
    'od': 1.0 * multiplier, 
    'link': 100.0 * multiplier, 
    'reg': 0.01 * multiplier
}

#%%
# --- FLUJO 1: ENTRENAMIENTO DESDE CERO ---

print("Iniciando un nuevo entrenamiento desde cero...")

# 1. Instanciar un nuevo modelo
pinn_instance = PINN_DTA_OD(data_container=dta_data_container, **model_init_config)

# 2. Llamar a la función de entrenamiento
trained_model, train_hist, val_hist, param_histories = train_model(
    pinn_model=pinn_instance,
    data_container=dta_data_container,
    model_init_config=model_init_config,
    epochs=200,  # Define el número de epochs a entrenar
    lr=0.01,
    num_t_collocation_pts=5,
    loss_weights=custom_loss_weights,
    save_path=model_save_file_path,
    save_every=1 # Guardar cada epoch
)

#%%
# --- FLUJO 2: CARGAR Y EVALUAR UN CHECKPOINT ---

print("Cargando checkpoint para evaluación...")

# 1. Instanciar modelo y optimizador (como contenedores vacíos)
pinn_to_load = PINN_DTA_OD(data_container=dta_data_container, **model_init_config)
optimizer_to_load = torch.optim.Adam(pinn_to_load.parameters())

# 2. Cargar el checkpoint deseado ('last' o un número de epoch)
epoch_a_cargar = 'last'  # o un número como 4 para el quinto epoch (índice 0)

try:
    loaded_epoch, loaded_histories, _ = load_checkpoint(
        model_save_file_path, 
        pinn_to_load, 
        optimizer_to_load, 
        epoch_to_load=epoch_a_cargar
    )
except (ValueError, FileNotFoundError) as e:
    print(f"Error al cargar el checkpoint: {e}")
    # Detener la ejecución de la celda si hay un error
    raise

# --- 3. Ejecutar todas las funciones de evaluación y visualización ---
pinn_to_load.eval() # Poner el modelo en modo de evaluación

#%%
# a) Gráfica de la evolución de la pérdida hasta el epoch cargado
print("\n[Visualización 1/6] Gráfica de Pérdidas...")
plot_loss_progression_standalone(
    loaded_histories['loss_train'], 
    loaded_histories['loss_val'], 
    loaded_epoch + 1, 
    output_dir=MAIN_OUTPUT_DIR,
    save_plot=True # Guardar la imagen
)
#%%
# b) Gráfica de la evolución de los parámetros BPR
print("\n[Visualización 2/6] Evolución de Parámetros BPR...")
# Crear el mapeo inverso de ID de grupo para las leyendas del gráfico
idx_to_group_id_map = {v: k for k, v in dta_data_container.group_id_to_idx_map.items()}
plot_bpr_parameter_evolution_standalone(
    loaded_histories['params']['bpr_alphas'],
    loaded_histories['params']['bpr_betas'],
    loaded_epoch + 1,
    idx_to_group_id_map,
    dta_data_container.num_actual_link_groups,
    output_dir=MAIN_OUTPUT_DIR,
    save_plot=True
)
#%%
# c) Tabla de parámetros de enlace y BPR aprendidos
print("\n[Evaluación 3/6] Parámetros de Enlace y BPR...")
print_link_type_parameters_standalone(
    model_instance=pinn_to_load,
    data_container=dta_data_container,
    output_dir=MAIN_OUTPUT_DIR,
    save_table=False # Guardar la tabla
)
#%%
# d) Comparación de demandas OD (mostrando en escala horaria)
print("\n[Evaluación 4/6] Comparación de Demandas OD...")
print_od_demand_comparison_standalone(
    model_instance=pinn_to_load,
    data_container=dta_data_container,
    dataset_type="test", # Evaluar sobre el conjunto de test
    output_dir=MAIN_OUTPUT_DIR,
    save_table=False,
    display_scale='none' # Mostrar en escala horaria (revierte el factor numérico)
)
#%%
# e) Comparación de flujos en enlaces y gráfico de dispersión
print("\n[Evaluación 5/6] Comparación de Flujos en Enlaces...")
t_eval = torch.tensor([1.0], device=device) # Evaluar en el punto de tiempo t=1.0
print_link_flow_comparison_standalone(
    model_instance=pinn_to_load,
    data_container=dta_data_container,
    t_eval_tensor=t_eval,
    output_dir=MAIN_OUTPUT_DIR,
    save_outputs=False, # Guardar tanto la tabla como el gráfico
)
#%%
# f) Métricas de regresión (R2, MAE, RMSE)
print("\n[Evaluación 6/6] Métricas de Regresión...")
calculate_regression_metrics_standalone(
    model_instance=pinn_to_load,
    data_container=dta_data_container,
    dataset_type="test",
    output_dir=MAIN_OUTPUT_DIR,
    save_metrics=False,
    # display_scale='none'
)

#%%
# --- FLUJO 3: CONTINUAR ENTRENAMIENTO ---

print("Reanudando entrenamiento desde el último checkpoint...")

# 1. Instanciar modelo y optimizador
pinn_to_resume = PINN_DTA_OD(data_container=dta_data_container, **model_init_config)
optimizer_to_resume = torch.optim.Adam(pinn_to_resume.parameters())

# 2. Cargar el ÚLTIMO checkpoint
last_epoch, last_histories, _ = load_checkpoint(
    model_save_file_path,
    pinn_to_resume,
    optimizer_to_resume,
    epoch_to_load='last'
)

# 3. Llamar a la función de entrenamiento para entrenar más epochs
epochs_adicionales = 100 # Define cuántos epochs más quieres entrenar
print(f"Entrenando por {epochs_adicionales} epochs adicionales...")

resumed_model, final_train_hist, final_val_hist, final_param_histories = train_model(
    pinn_model=pinn_to_resume,
    data_container=dta_data_container,
    model_init_config=model_init_config,
    epochs=epochs_adicionales,
    lr=0.01, # Puedes ajustar el LR si lo deseas
    num_t_collocation_pts=5,
    loss_weights=custom_loss_weights,
    save_path=model_save_file_path,
    starting_epoch=last_epoch + 1,
    loss_history_train=last_histories['loss_train'],
    loss_history_val=last_histories['loss_val'],
    param_histories=last_histories['params']
)


#%%
"""
PINN_01:

dta_data_container = DTAData(
    original_data_pickle_path=directory_pickle, 
    device=device,
    pen_rate=False, day_to_hour=False, modal_choice=True
)

model_init_config = {
    'hidden_dim_x': 4,
    'initial_alpha_dyn': 0.05,
    'initial_unknown_demand_log_val': initial_log_demand_val,
}


multiplier = 1.0
custom_loss_weights = {
    'phys': 1.0 * multiplier, 
    'ic': 10.0 * multiplier,
    'od': 10.0 * multiplier, 
    'link': 5.0 * multiplier, 
    'reg': 0.01 * multiplier
}

"""


"""
PINN_02:

dta_data_container = DTAData(
    original_data_pickle_path=directory_pickle, 
    device=device,
    pen_rate=False, day_to_hour=False, modal_choice=True
)

model_init_config = {
    'hidden_dim_x': 4,
    'initial_alpha_dyn': 0.05,
    'initial_unknown_demand_log_val': initial_log_demand_val,
}


multiplier = 1.0
custom_loss_weights = {
    'phys': 1.0 * multiplier, 
    'ic': 10.0 * multiplier,
    'od': 10.0 * multiplier, 
    'link': 5.0 * multiplier, 
    'reg': 0.01 * multiplier
}

"""
# %%
