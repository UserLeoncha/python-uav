import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean


# 生成随机节点
def generate_random_nodes(num_nodes, x_max, y_max):
    nodes = np.random.rand(num_nodes, 2)
    nodes[:, 0] *= x_max
    nodes[:, 1] *= y_max
    return nodes


# 计算两点之间的欧几里得距离
def calculate_distance(a, b):
    return euclidean(a, b)


# 使用最近邻算法解决TSP问题
def nearest_neighbor_tsp(nodes):
    path = []
    num_nodes = len(nodes)
    visited = [False] * num_nodes

    # 起点
    current = np.array([100.0, 100.0])
    path.append(current)

    for _ in range(num_nodes):
        min_distance = float('inf')
        nearest_node_index = -1

        for i in range(num_nodes):
            if not visited[i]:
                distance = calculate_distance(current, nodes[i])
                if distance < min_distance:
                    min_distance = distance
                    nearest_node_index = i

        visited[nearest_node_index] = True
        current = nodes[nearest_node_index]
        path.append(current)

    # 终点
    path.append(np.array([0.0, 0.0]))

    return np.array(path)


# 绘制路径
def draw_path(path):
    plt.figure(figsize=(8, 8))
    plt.plot(path[:, 0], path[:, 1], marker='o', linestyle='-', color='b')
    plt.scatter(path[:, 0], path[:, 1], color='r')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('TSP Path')
    plt.grid(True)
    plt.show()


def main():
    num_nodes = 1000

    x_max = 100.0
    y_max = 100.0

    # 生成随机节点
    nodes = generate_random_nodes(num_nodes, x_max, y_max)

    # 计算最近邻路径
    path = nearest_neighbor_tsp(nodes)

    # 绘制路径
    draw_path(path)


if __name__ == "__main__":
    main()
