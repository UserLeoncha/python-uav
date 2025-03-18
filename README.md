Algorithm Implementations
1. Ant Colony Optimization (ACO)
Simple Version: Sequential implementation in ACO_simple.py
Multi-processing Version: Parallel implementation in ACO_multiprogress.py for improved performance
Features: Random node generation, pheromone matrix calculation, path construction and visualization
2. Nearest Neighbor Algorithm (NNA)
Simple heuristic approach for path planning
Fast computation but suboptimal results
3. K-Means Clustering
Used for grouping waypoints into clusters
Can help simplify complex path planning problems
4. Artificial Potential Field (APF)
Obstacle avoidance-based path planning
Creates virtual forces to guide UAV navigation
5. Genetic Algorithm (MATLAB)
Population-based optimization for path planning
Implements selection, crossover and mutation operations

python-uav/
├── algorithm/             # Core algorithm implementations
│   ├── ACO_simple.py      # Ant Colony Optimization (basic)
│   ├── ACO_multiprogress.py  # Parallel Ant Colony Optimization
│   ├── APF.py             # Artificial Potential Field
│   ├── kmeans.py          # K-means clustering
│   ├── NNA.py             # Nearest Neighbor Algorithm
│   └── ok_path.py         # Path validation
├── charts/                # Visualization tools
│   ├── draw.py            # Main visualization interface
│   ├── kmeans.py          # K-means visualization
│   ├── NNA.py             # NNA visualization
│   └── *.ipynb            # Jupyter notebooks for analysis
└── result/                # Output images and data
    └── *.png              # Result visualizations

matlab-uav/
├── allinone.m             # Comprehensive MATLAB implementation
└── README.md              # MATLAB project documentation