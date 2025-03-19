import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from multiprocessing import Pool, cpu_count

# 生成随机节点
np.random.seed(42)  # 为了结果可重复
num_nodes = 1000
nodes = np.random.rand(num_nodes, 2) * 100

# 添加起点和终点
start = np.array([0, 0])
end = np.array([100, 100])
nodes = np.vstack([start, nodes, end])


# 计算距离矩阵
def calculate_distance_matrix(nodes):
    num_nodes = len(nodes)
    dist_matrix = np.zeros((num_nodes, num_nodes))
    for i in range(num_nodes):
        for j in range(i, num_nodes):
            dist_matrix[i, j] = dist_matrix[j, i] = np.linalg.norm(nodes[i] - nodes[j])
    return dist_matrix


dist_matrix = calculate_distance_matrix(nodes)


# 蚁群算法
class ACO:
    def __init__(self, dist_matrix, num_ants, num_iterations, alpha=1, beta=5, evaporation_rate=0.5, Q=100):
        self.dist_matrix = dist_matrix
        self.num_ants = num_ants
        self.num_iterations = num_iterations
        self.alpha = alpha
        self.beta = beta
        self.evaporation_rate = evaporation_rate
        self.Q = Q
        self.num_nodes = dist_matrix.shape[0]
        self.pheromone_matrix = np.ones((self.num_nodes, self.num_nodes))

    def run(self):
        best_path = None
        best_length = np.inf

        for _ in range(self.num_iterations):
            print(f"Iteration {_ + 1}/{self.num_iterations}")
            with Pool(cpu_count()) as pool:
                all_paths = pool.map(self.construct_path, range(self.num_ants))

            self.update_pheromones(all_paths)
            for path, length in all_paths:
                if length < best_length:
                    best_path = path
                    best_length = length

        return best_path, best_length

    def construct_path(self, _):
        path = [0]
        visited = set(path)
        for _ in range(self.num_nodes - 1):
            current = path[-1]
            probabilities = self.calculate_probabilities(current, visited)
            next_node = np.random.choice(range(self.num_nodes), p=probabilities)
            path.append(next_node)
            visited.add(next_node)
        path.append(self.num_nodes - 1)  # 强制终点
        length = self.calculate_path_length(path)
        return (path, length)

    def calculate_probabilities(self, current, visited):
        pheromones = np.copy(self.pheromone_matrix[current])
        pheromones[list(visited)] = 0
        heuristics = 1 / (self.dist_matrix[current] + 1e-10)
        heuristics[list(visited)] = 0
        probabilities = (pheromones ** self.alpha) * (heuristics ** self.beta)
        probabilities /= probabilities.sum()
        return probabilities

    def calculate_path_length(self, path):
        length = 0
        for i in range(len(path) - 1):
            length += self.dist_matrix[path[i], path[i + 1]]
        return length

    def update_pheromones(self, all_paths):
        self.pheromone_matrix *= (1 - self.evaporation_rate)
        for path, length in all_paths:
            for i in range(len(path) - 1):
                self.pheromone_matrix[path[i], path[i + 1]] += self.Q / length
                self.pheromone_matrix[path[i + 1], path[i]] += self.Q / length


# 运行蚁群算法
aco = ACO(dist_matrix, num_ants=50, num_iterations=100)
best_path, best_length = aco.run()


# GUI展示
def draw_path():
    window = tk.Tk()
    window.title("旅行商问题路径")

    canvas = tk.Canvas(window, width=600, height=600)
    canvas.pack()

    for i in range(len(best_path) - 1):
        x1, y1 = nodes[best_path[i]]
        x2, y2 = nodes[best_path[i + 1]]
        canvas.create_line(x1 * 6, y1 * 6, x2 * 6, y2 * 6, fill="blue", width=2)
        canvas.create_oval(x1 * 6 - 3, y1 * 6 - 3, x1 * 6 + 3, y1 * 6 + 3, fill="red")
        canvas.create_oval(x2 * 6 - 3, y2 * 6 - 3, x2 * 6 + 3, y2 * 6 + 3, fill="red")

    window.mainloop()


draw_path()
