import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random

def euclidean_distance(p1, p2):
    """Calculate the Euclidean distance between two points."""
    return np.hypot(p1[0] - p2[0], p1[1] - p2[1])

def tsp_path_planning(a, b, nodes, speeds):
    """
    Plan the shortest path that visits ALL nodes with fixed start (0,0) and end (a,b).
    
    参数:
        a, b: 整个区域尺寸（用于验证固定起点与终点）
        nodes: 节点列表，包含固定起点和终点以及内部节点，节点格式为(x,y)
        speeds: 每一段路程的速度列表
    """
    # Build a complete graph using node indices
    n_nodes = len(nodes)
    G = nx.complete_graph(n_nodes)
    for i in range(n_nodes):
        for j in range(i+1, n_nodes):
            dist = euclidean_distance(nodes[i], nodes[j])
            G.add_edge(i, j, weight=dist)
    
    # Solve TSP cycle on the complete graph.
    cycle = nx.approximation.traveling_salesman_problem(G, cycle=True, weight='weight')
    
    # Ensure fixed start and end: start node is index 0 (i.e. (0,0)), end node is last index (i.e. (a,b))
    while cycle[0] != 0:
        cycle = cycle[1:] + cycle[:1]
    
    # Truncate cycle so that it ends at (a,b)
    end_idx = n_nodes - 1
    if end_idx in cycle:
        idx = cycle.index(end_idx)
        tsp_idx = cycle[:idx+1]
    else:
        tsp_idx = cycle  # fallback
    
    tsp_path = [nodes[i] for i in tsp_idx]
    
    # Calculate total time using speed for each segment (wrap speeds if needed)
    total_time = 0
    for i in range(len(tsp_path)-1):
        dist = euclidean_distance(tsp_path[i], tsp_path[i+1])
        speed = speeds[i % len(speeds)]
        total_time += dist / speed
    
    return tsp_path, total_time

def draw_tsp_path(a, b, tsp_path, nodes, total_time):
    """
    Draw the nodes and the TSP route.
    
    该函数不再绘制六边形背景，而是直接绘制遍历所有节点的路径
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_xlim(-10, a+10)
    ax.set_ylim(-10, b+10)
    
    # Draw nodes
    for node in nodes:
        ax.plot(node[0], node[1], 'ro')
    
    # Draw TSP path
    for i in range(len(tsp_path) - 1):
        x1, y1 = tsp_path[i]
        x2, y2 = tsp_path[i + 1]
        ax.plot([x1, x2], [y1, y2], 'b-', linewidth=2)
    
    # Display total time as text
    ax.text(0.5, 1.05, f"Total Time: {total_time:.2f}", transform=ax.transAxes, 
            ha="center", fontsize=14)
    
    plt.show()

def main(a, b, n, d):
    if n < 2:
        raise ValueError("n must be at least 2 to include start and end points.")
    
    # Generate n-2 random internal nodes
    internal_nodes = [(random.uniform(0, a), random.uniform(0, b)) for _ in range(n - 2)]
    # Fixed start and end points
    nodes = [(0, 0)] + internal_nodes + [(a, b)]
    
    # Generate random speeds for (n-1) segments
    speeds = [random.uniform(1, 5) for _ in range(len(nodes) - 1)]
    
    tsp_path, total_time = tsp_path_planning(a, b, nodes, speeds)
    
    draw_tsp_path(a, b, tsp_path, nodes, total_time)

# Example parameters
if __name__ == "__main__":
    a = 100
    b = 100
    n = 100   # total nodes including start and end
    d = 5   # d is no longer used in TSP planning; kept for compatibility
    main(a, b, n, d)