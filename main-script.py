# main.py
import torch

from data.loader import load_traffic_data
from models.gat_model import TrafficGAT
from optimization.optimization import optimize_sensor_network
from visualization.network_viz import visualize_sensor_network, save_network_visualization
from evaluation.evaluation import validate_model, print_model_performance

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

    # Load traffic data
    print("Loading traffic data...")
    data = load_traffic_data()

    # Optimize sensor network
    print("\nOptimizing sensor network...")
    model, optimized_mask = optimize_sensor_network(
        data, 
        model_class=TrafficGAT, 
        hidden_dim=64, 
        heads=4, 
        max_error_increase=0.10, 
        metric='mae', 
        num_epochs=100
    )

    # Visualize optimized sensor network
    print("\nVisualizing sensor network...")
    visualize_sensor_network(data, optimized_mask)
    save_network_visualization(data, optimized_mask, filename="sensor_network_optimization.png")

    # Validate model performance
    print("\nValidating model performance...")
    # Assuming test_mask is the complement of train_mask
    test_mask = ~data.train_mask
    performance_metrics = validate_model(model, data, test_mask)
    print_model_performance(performance_metrics)

if __name__ == "__main__":
    main()
