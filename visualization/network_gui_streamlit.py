# visualization/network_gui_streamlit.py
import streamlit as st
import os
import torch
import sys
import importlib.util


# Find and import NetworkVisualizer_Pyvis from the correct location
def import_network_viz():
    # Approach 1: Try relative import from current directory
    try:
        # Try to import from current directory first
        spec = importlib.util.spec_from_file_location(
            "network_viz",
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "network_viz.py")
        )
        network_viz = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(network_viz)
        return network_viz.NetworkVisualizer_Pyvis
    except Exception as e:
        st.warning(f"First import approach failed: {e}")

    # Approach 2: Try importing using sys.path manipulation
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    if parent_dir not in sys.path:
        sys.path.append(parent_dir)

    try:
        from visualization.network_viz import NetworkVisualizer_Pyvis
        return NetworkVisualizer_Pyvis
    except Exception as e:
        st.warning(f"Second import approach failed: {e}")

    # Last resort: direct import without namespace
    try:
        import network_viz
        return network_viz.NetworkVisualizer_Pyvis
    except Exception as e:
        st.error(f"All import attempts failed: {e}")
        raise ImportError("Could not import NetworkVisualizer_Pyvis")


# Import the visualizer class
NetworkVisualizer_Pyvis = import_network_viz()


def load_viz_data(path=None):
    """
    Loads exported visualization data.

    Args:
        path (str, optional): Path to the folder with the data. If None, will use
                             the default path: visualization/exported_viz_data/
    """
    if path is None:
        # Try several possible paths, depending on how the script is run
        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(current_dir)

        paths_to_try = [
            os.path.join(current_dir, "exported_viz_data"),  # Relative to this script
            os.path.join(parent_dir, "visualization", "exported_viz_data"),  # From parent dir
            "visualization/exported_viz_data",  # If run from main directory
            "exported_viz_data",  # If run from visualization/ directory
        ]

        for p in paths_to_try:
            if os.path.exists(p) and os.path.isdir(p):
                path = p
                st.success(f"Found data path: {path}")
                break

        if path is None:
            raise FileNotFoundError("Folder 'exported_viz_data' not found. "
                                    "Run the main script first.")

    try:
        nodes = torch.load(os.path.join(path, "nodes.pt"))
        links = torch.load(os.path.join(path, "links.pt"))
        estimated_flows = torch.load(os.path.join(path, "estimated_flows.pt"))
        observed_flows = torch.load(os.path.join(path, "observed_flows.pt"))
        return nodes, links, estimated_flows, observed_flows
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        st.error(f"Looking for files in: {path}")
        st.error(f"Directory contents: {os.listdir(path) if os.path.exists(path) else 'Does not exist'}")
        raise


def main(viz=None):
    """
    Main Streamlit interface function.

    Args:
        viz (NetworkVisualizer_Pyvis, optional): Pre-loaded visualizer. If None,
                                               will try to load exported data.
    """
    st.title("Network Visualizer - Streamlit")
    st.sidebar.header("Visualization Configuration")

    # If no visualizer was provided, try to load the data
    if viz is None:
        try:
            nodes, links, estimated_flows, observed_flows = load_viz_data()
            viz = NetworkVisualizer_Pyvis(nodes, links, estimated_flows, observed_flows)
            st.success("Data loaded successfully")
        except Exception as e:
            st.error(f"Could not load visualization: {str(e)}")
            st.info("Run main-script.py first to generate visualization data")
            return

    # Visualization configuration controls
    show_node_labels = st.sidebar.checkbox("Show node labels", value=viz.config["show_node_labels"])
    node_label_font_size = st.sidebar.slider("Node label font size", 8, 40, viz.config["node_label_font_size"])
    node_size_multiplier = st.sidebar.slider("Node size multiplier", 0.1, 5.0, viz.config["node_size_multiplier"],
                                             0.1)
    node_color = st.sidebar.color_picker("Node color", viz.config.get("node_default_color", "#97c2fc"))

    edge_label_mode = st.sidebar.selectbox("Edge label mode", options=["combined", "estimated", "observed"],
                                           index=["combined", "estimated", "observed"].index(
                                               viz.config["edge_label_mode"]))
    show_edge_labels = st.sidebar.checkbox("Show edge labels", value=viz.config.get("show_edge_labels", True))
    base_edge_width = st.sidebar.slider("Base edge width", 0.5, 5.0, viz.config["base_edge_width"], 0.1)
    edge_width_scaling = st.sidebar.slider("Edge width scaling", 0.001, 0.05, viz.config["edge_width_scaling"], 0.001)
    flow_tolerance = st.sidebar.slider("Flow tolerance", 0.0, 20.0, viz.config["flow_tolerance"], 0.5)

    gravitational_constant = st.sidebar.slider("Gravity", -1000, 0, viz.config["gravitational_constant"], 50)
    spring_length = st.sidebar.slider("Spring length", 10, 300, viz.config["spring_length"], 10)
    bidirectional = st.sidebar.checkbox("Bidirectional mode", value=viz.config["bidirectional"])

    export_filepath = st.sidebar.text_input("HTML export path",
                                            value=viz.config.get("export_filepath", "network_visualization.html"))

    if st.sidebar.button("Update Visualization"):
        viz.update_config(
            show_node_labels=show_node_labels,
            node_label_font_size=node_label_font_size,
            node_size_multiplier=node_size_multiplier,
            node_default_color=node_color,
            edge_label_mode=edge_label_mode,
            show_edge_labels=show_edge_labels,
            base_edge_width=base_edge_width,
            edge_width_scaling=edge_width_scaling,
            flow_tolerance=flow_tolerance,
            gravitational_constant=gravitational_constant,
            spring_length=spring_length,
            bidirectional=bidirectional,
            export_filepath=export_filepath
        )

        try:
            html_data = viz.draw()
            st.components.v1.html(html_data, height=700, scrolling=True)
        except Exception as e:
            st.error(f"Error generating visualization: {str(e)}")


if __name__ == "__main__":
    # Provide better error information for debugging
    try:
        main()
    except Exception as e:
        import traceback

        st.error(f"Error running application: {str(e)}")
        st.code(traceback.format_exc())