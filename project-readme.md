# Traffic Sensor Network Optimization

## Project Overview
This project implements a Graph Attention Network (GAT) for optimizing traffic sensor networks. The main goal is to identify and retain the most crucial sensors while maintaining high prediction accuracy.

## Key Features
- Graph Attention Network (GAT) for traffic flow prediction
- Performance-based sensor pruning
- Network visualization
- Comprehensive model evaluation

## Project Structure
```
traffic_sensor_optimization/
├── data/           # Data loading and preprocessing
├── models/         # Neural network model definitions
├── optimization/   # Sensor network pruning strategies
├── evaluation/     # Model performance metrics
├── visualization/  # Network visualization tools
├── main.py         # Main script to run the optimization pipeline
└── requirements.txt
```

## Installation
1. Clone the repository
2. Create a virtual environment
3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage
Run the main script to perform sensor network optimization:
```bash
python main.py
```

## Key Components
- **Data Loader**: Prepares traffic network data from pickle files
- **GAT Model**: Graph Attention Network for predicting traffic flows
- **Optimization**: Prunes sensors based on importance while maintaining performance
- **Visualization**: Generates network graphs showing sensor status

## Methodology
1. Load traffic network data
2. Train initial Graph Attention Network
3. Compute edge importance using attention weights
4. Prune sensors with minimal performance impact
5. Retrain model with optimized sensor set
6. Visualize and evaluate results

## Performance Metrics
- Mean Absolute Error (MAE)
- Root Mean Square Error (RMSE)
- Mean Squared Error (MSE)

## Dependencies
- PyTorch
- PyTorch Geometric
- NetworkX
- Matplotlib
- scikit-learn

## Contributing
Please read CONTRIBUTING.md for details on our code of conduct and the process for submitting pull requests.

## License
This project is licensed under the MIT License - see the LICENSE.md file for details.
