# processing/topology_updater.py

import geopandas as gpd
import pandas as pd
import logging

logger = logging.getLogger(__name__)

def update_node_topology(gdf_links_with_topology, gdf_nodes):
    """
    Updates the 'incoming_links' and 'outgoing_links' lists in gdf_nodes
    based on the topology information in gdf_links_with_topology.

    This function is useful after manually correcting link directions,
    to ensure node connectivity reflects the corrected link topology.

    Args:
        gdf_links_with_topology (gpd.GeoDataFrame): Links GeoDataFrame with
                                       'ELEMENT_ID', 'start_node_id_logico',
                                       'end_node_id_logico', and 'direccion_logica' columns.
        gdf_nodes (gpd.GeoDataFrame): Nodes GeoDataFrame with a 'node_id' column.

    Returns:
        gpd.GeoDataFrame: The gdf_nodes GeoDataFrame with updated
                          'incoming_links' and 'outgoing_links' list columns.
    """
    logger.info("Iniciando actualizaci\u00F3n de topolog\u00EDa de nodos...")

    # Ensure we are working on a copy of gdf_nodes
    gdf_nodes_updated = gdf_nodes.copy()

    # Clear existing incoming/outgoing link lists in gdf_nodes_updated
    gdf_nodes_updated['incoming_links'] = [[] for _ in range(len(gdf_nodes_updated))]
    gdf_nodes_updated['outgoing_links'] = [[] for _ in range(len(gdf_nodes_updated))]

    # Create a mapping from node_id to its index in the nodes GeoDataFrame for faster lookup
    node_id_to_index = pd.Series(gdf_nodes_updated.index, index=gdf_nodes_updated['node_id']).to_dict()

    # Iterate through the links with topology information
    for index, row in gdf_links_with_topology.iterrows():
        segment_id = row['ELEMENT_ID']
        start_node_logico = row['start_node_id_logico']
        end_node_logico = row['end_node_id_logico']
        direccion_logica_val = row['direccion_logica']

        # Skip if topology information is missing for this link
        if pd.isna(start_node_logico) or pd.isna(end_node_logico) or pd.isna(direccion_logica_val):
            logger.warning(f"Link {segment_id} tiene informaci\u00F3n de topolog\u00EDa incompleta. Saltando.")
            continue

        # Get the indices of the start and end nodes using the lookup dictionary
        start_node_index = node_id_to_index.get(start_node_logico)
        end_node_index = node_id_to_index.get(end_node_logico)

        # Ensure both start and end nodes exist in the nodes GeoDataFrame
        if start_node_index is None or end_node_index is None:
            logger.warning(f"Nodos extremos ({start_node_logico} o {end_node_logico}) del link {segment_id} no encontrados en gdf_nodes. Saltando actualizaci\u00F3n para este link.")
            continue

        # Update incoming/outgoing link lists based on direction
        if direccion_logica_val == 1: # Directed: start_node -> end_node
            gdf_nodes_updated.loc[start_node_index, 'outgoing_links'].append(segment_id)
            gdf_nodes_updated.loc[end_node_index, 'incoming_links'].append(segment_id)
        elif direccion_logica_val == 0: # Bidirectional
            gdf_nodes_updated.loc[start_node_index, 'outgoing_links'].append(segment_id)
            gdf_nodes_updated.loc[start_node_index, 'incoming_links'].append(segment_id)
            gdf_nodes_updated.loc[end_node_index, 'outgoing_links'].append(segment_id)
            gdf_nodes_updated.loc[end_node_index, 'incoming_links'].append(segment_id)
        else:
            logger.warning(f"Valor de direcci\u00F3n l\u00F3gica inesperado ({direccion_logica_val}) para link {segment_id}. Saltando actualizaci\u00F3n para este link.")


    logger.info("Actualizaci\u00F3n de topolog\u00EDa de nodos completada.")
    return gdf_nodes_updated

