# visualization/network_viz.py
import matplotlib

matplotlib.use("TkAgg")  # Or try "Qt5Agg"
import matplotlib.pyplot as plt
import networkx as nx


def visualize_sensor_network(data, optimized_mask, title="Sensor Network"):
    """
    Visualize the sensor network with color-coded edges indicating sensor status.

    Args:
        data (Data): PyTorch Geometric Data object
        optimized_mask (torch.Tensor): Mask indicating optimized sensors
        title (str, optional): Title for the visualization. Defaults to "Sensor Network".
    """
    # Convert PyG graph to NetworkX DiGraph
    G = nx.DiGraph()
    edge_list = data.edge_index.cpu().numpy().T
    G.add_edges_from(edge_list)

    # Color edges based on sensor status
    edge_colors = []
    for i in range(len(edge_list)):
        if data.train_mask[i] and not optimized_mask[i]:
            edge_colors.append('red')  # Sensors removed by pruning
        elif optimized_mask[i]:
            edge_colors.append('green')  # Essential sensors retained
        else:
            edge_colors.append('grey')  # Never measured edges

    # Create visualization
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(G, seed=42)
    nx.draw(
        G,
        pos,
        edge_color=edge_colors,
        width=1.5,
        arrowsize=12,
        node_size=50
    )

    plt.title(title)
    plt.legend(handles=[
        plt.Line2D([0], [0], color='green', lw=2, label='Essential Sensors'),
        plt.Line2D([0], [0], color='red', lw=2, label='Removed Sensors'),
        plt.Line2D([0], [0], color='grey', lw=2, label='Never Measured')
    ])
    plt.tight_layout()
    plt.show()


def save_network_visualization(data, optimized_mask, filename="sensor_network.png", title="Sensor Network"):
    """
    Save the sensor network visualization to a file.

    Args:
        data (Data): PyTorch Geometric Data object
        optimized_mask (torch.Tensor): Mask indicating optimized sensors
        filename (str, optional): Output filename. Defaults to "sensor_network.png".
        title (str, optional): Title for the visualization. Defaults to "Sensor Network".
    """
    # Convert PyG graph to NetworkX DiGraph
    G = nx.DiGraph()
    edge_list = data.edge_index.cpu().numpy().T
    G.add_edges_from(edge_list)

    # Color edges based on sensor status
    edge_colors = []
    for i in range(len(edge_list)):
        if data.train_mask[i] and not optimized_mask[i]:
            edge_colors.append('red')  # Sensors removed by pruning
        elif optimized_mask[i]:
            edge_colors.append('green')  # Essential sensors retained
        else:
            edge_colors.append('grey')  # Never measured edges

    # Create visualization
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(G, seed=42)
    nx.draw(
        G,
        pos,
        edge_color=edge_colors,
        width=1.5,
        arrowsize=12,
        node_size=50
    )

    plt.title(title)
    plt.legend(handles=[
        plt.Line2D([0], [0], color='green', lw=2, label='Essential Sensors'),
        plt.Line2D([0], [0], color='red', lw=2, label='Removed Sensors'),
        plt.Line2D([0], [0], color='grey', lw=2, label='Never Measured')
    ])
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()