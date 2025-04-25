# processing/matching.py
import logging
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point, LineString
import numpy as np


logger = logging.getLogger(__name__)


def match_segments(gdf_links, gdf_nodes, node_tolerance=5.0):
    """
    Matches segment endpoints to nodes and determines logical direction based on attributes.

    Args:
        gdf_links (gpd.GeoDataFrame): GeoDataFrame of road segments (links).
                                       Expected columns: 'geometry' (LineString),
                                       'ELEMENT_ID', 'ROLE', 'DIRECTION'.
        gdf_nodes (gpd.GeoDataFrame): GeoDataFrame of nodes (junctions).
                                       Expected columns: 'geometry' (Point), 'node_id'.
        node_tolerance (float): The maximum distance to consider a segment endpoint
                                matching a node. Use the same CRS units as the GeoDataFrames.

    Returns:
        tuple: A tuple containing:
            - gdf_links_with_topology (gpd.GeoDataFrame): Original gdf_links with
              'start_node_id_logico', 'end_node_id_logico', and 'direccion_logica' columns added.
            - gdf_nodes_with_topology (gpd.GeoDataFrame): Original gdf_nodes with
              'incoming_links' and 'outgoing_links' list columns added.
    """

    # Ensure input GeoDataFrames are copied to avoid modifying originals outside the function
    gdf_links_with_topology = gdf_links.copy()
    gdf_nodes_with_topology = gdf_nodes.copy()

    # Add new columns for topology information to gdf_links
    gdf_links_with_topology['start_node_id'] = None
    gdf_links_with_topology['end_node_id'] = None
    # 0: bidirectional, 1: directed (from start_node_id_logico to end_node_id_logico)
    gdf_links_with_topology['direction_log'] = 0

    # Add new list columns for topology information to gdf_nodes
    # Initialize as empty lists
    gdf_nodes_with_topology['incoming_links'] = [[] for _ in range(len(gdf_nodes_with_topology))]
    gdf_nodes_with_topology['outgoing_links'] = [[] for _ in range(len(gdf_nodes_with_topology))]

    # --- Step 1: Identify Nodes Connected to Segment Endpoints ---
    # Extract start and end points of each LineString geometry
    # Handle potential empty geometries or non-LineStrings gracefully
    start_points = []
    end_points = []
    valid_link_indices = []
    for i, geom in gdf_links_with_topology.geometry.items():
        if geom and isinstance(geom, LineString) and not geom.is_empty:
            start_points.append(Point(geom.coords[0]))
            end_points.append(Point(geom.coords[-1]))
            valid_link_indices.append(i)
        else:
             # Log a warning or handle invalid geometry if necessary
             print(f"Warning: Invalid or empty geometry for link index {i}")


    gdf_start_points = gpd.GeoDataFrame(geometry=start_points, index=valid_link_indices, crs=gdf_links_with_topology.crs)
    gdf_end_points = gpd.GeoDataFrame(geometry=end_points, index=valid_link_indices, crs=gdf_links_with_topology.crs)

    # Perform spatial join to find the nearest node to each start and end point
    # sjoin_nearest requires geopandas version 0.10 or later
    try:
        start_match = gpd.sjoin_nearest(gdf_start_points, gdf_nodes_with_topology[['node_id', 'geometry']],
                                        how='left', max_distance=node_tolerance, distance_col='distance_start')
        end_match = gpd.sjoin_nearest(gdf_end_points, gdf_nodes_with_topology[['node_id', 'geometry']],
                                      how='left', max_distance=node_tolerance, distance_col='distance_end')
    except Exception as e:
        print(f"Error during sjoin_nearest: {e}")
        print("Please ensure you have a recent version of geopandas (>=0.10) and that your CRS units are appropriate for node_tolerance.")
        return gdf_links_with_topology, gdf_nodes_with_topology # Return current state

    # Merge the matched node IDs back to the links GeoDataFrame
    # Use original index to merge correctly
    gdf_links_with_topology['matched_start_node_id'] = start_match['node_id']
    gdf_links_with_topology['matched_end_node_id'] = end_match['node_id']


    # --- Step 2 & 3: Determine Logical Direction and Populate Topology ---
    # Iterate through links to set logical start/end nodes and direction
    for index, row in gdf_links_with_topology.iterrows():
        segment_id = row['ELEMENT_ID']
        role = row['ROLE']
        direction = row['DIRECTION']
        geom = row['geometry']

        # Get the nodes matched to the geometric start and end points
        node_id_cercano_inicio_geom = row['matched_start_node_id']
        node_id_cercano_fin_geom = row['matched_end_node_id']

        # Skip if endpoints didn't match any nodes within tolerance
        if pd.isna(node_id_cercano_inicio_geom) or pd.isna(node_id_cercano_fin_geom):
            # Log a warning or handle segments not connected to nodes
            print(f"Warning: Segment {segment_id} could not be matched to nodes at one or both ends within tolerance {node_tolerance}.")
            continue

        # --- Apply Logical Direction Logic ---
        start_node_logico = None
        end_node_logico = None
        direccion_logica_val = 0 # Default to bidirectional for now, will set to 1 for directed

        if role == 'Normal':
            # Normal segments are bidirectional in the network model
            # The 'med'/'mot' indicates geometry order relation, but network is bidirectional
            start_node_logico = node_id_cercano_inicio_geom
            end_node_logico = node_id_cercano_fin_geom
            direccion_logica_val = 0 # Bidirectional

        elif role == 'Syskon fram' or role == 'Syskon bak':
            # For Syskon segments, use 'med'/'mot' relative to geometric order
            # Assumption: 'med' follows geometric order, 'mot' reverses it.
            if direction == 'Med':
                start_node_logico = node_id_cercano_inicio_geom
                end_node_logico = node_id_cercano_fin_geom
            elif direction == 'Mot':
                start_node_logico = node_id_cercano_fin_geom
                end_node_logico = node_id_cercano_inicio_geom
            else:
                 # Handle unexpected DIRECTION values for Syskon if necessary
                 print(f"Warning: Unexpected DIRECTION '{direction}' for Syskon segment {segment_id}. Assuming 'med' logic.")
                 start_node_logico = node_id_cercano_inicio_geom
                 end_node_logico = node_id_cercano_fin_geom

            direccion_logica_val = 1 # Directed

        else:
            # Handle other ROLE values if necessary
            print(f"Warning: Unexpected ROLE '{role}' for segment {segment_id}. Treating as bidirectional.")
            start_node_logico = node_id_cercano_inicio_geom
            end_node_logico = node_id_cercano_fin_geom
            direccion_logica_val = 0 # Default to bidirectional


        # Update the topology columns in the links GeoDataFrame
        gdf_links_with_topology.loc[index, 'start_node_id'] = start_node_logico
        gdf_links_with_topology.loc[index, 'end_node_id'] = end_node_logico
        gdf_links_with_topology.loc[index, 'direction_log'] = direccion_logica_val

        # --- Update Node Topology (incoming/outgoing links) ---
        # Find the index of the start and end nodes in gdf_nodes_with_topology
        start_node_index = gdf_nodes_with_topology[gdf_nodes_with_topology['node_id'] == start_node_logico].index[0]
        end_node_index = gdf_nodes_with_topology[gdf_nodes_with_topology['node_id'] == end_node_logico].index[0]


        if direccion_logica_val == 1: # Directed
            # Add segment_id to outgoing links of start node and incoming links of end node
            gdf_nodes_with_topology.loc[start_node_index, 'outgoing_links'].append(segment_id)
            gdf_nodes_with_topology.loc[end_node_index, 'incoming_links'].append(segment_id)
        else: # Bidirectional (direccion_logica_val == 0)
            # Add segment_id to both incoming and outgoing lists for both nodes
            gdf_nodes_with_topology.loc[start_node_index, 'outgoing_links'].append(segment_id)
            gdf_nodes_with_topology.loc[start_node_index, 'incoming_links'].append(segment_id)
            gdf_nodes_with_topology.loc[end_node_index, 'outgoing_links'].append(segment_id)
            gdf_nodes_with_topology.loc[end_node_index, 'incoming_links'].append(segment_id)


    # Drop the temporary matched node ID columns
    gdf_links_with_topology = gdf_links_with_topology.drop(columns=['matched_start_node_id', 'matched_end_node_id'])


    return gdf_links_with_topology, gdf_nodes_with_topology

