import numpy as np
import math
from collections import deque


class HexGridPlanner:
    def __init__(self, a, b, n, d):
        self.a = a  # 场地长度
        self.b = b  # 场地宽度
        self.n = n  # 节点个数
        self.d = d  # 通信范围
        self.hex_centers = []  # 存储六边形中心坐标
        self.node_positions = []  # 节点的实际坐标
        self.hex_with_nodes = set()  # 含有节点的六边形

    def generate_hex_grid(self):
        # 按照d划分整个区域为六边形网格，计算六边形的中心
        hex_height = math.sqrt(3) * self.d
        cols = int(self.a / (3 / 2 * self.d))
        rows = int(self.b / hex_height)

        for r in range(rows):
            for c in range(cols):
                x = c * 3 / 2 * self.d
                y = r * hex_height + (self.d * (r % 2))  # 每行偏移
                self.hex_centers.append((x, y))

    def generate_random_nodes(self):
        # 随机生成n个节点
        for _ in range(self.n):
            x = np.random.uniform(0, self.a)
            y = np.random.uniform(0, self.b)
            self.node_positions.append((x, y))

        # 将节点映射到最近的六边形中心
        for node in self.node_positions:
            nearest_hex = self.find_nearest_hex(node)
            self.hex_with_nodes.add(nearest_hex)

    def find_nearest_hex(self, node):
        # 找到距离node最近的六边形中心
        nearest = min(self.hex_centers, key=lambda hex: np.linalg.norm(np.array(hex) - np.array(node)))
        return nearest

    def bfs_plan(self):
        # 使用BFS从(0,0)开始规划路径，遍历所有包含节点的六边形
        start = (0, 0)
        visited = set()
        queue = deque([start])
        path = []

        while queue:
            current = queue.popleft()
            if current in visited:
                continue
            visited.add(current)
            path.append(current)

            # 检查相邻的六边形
            for neighbor in self.get_neighbors(current):
                if neighbor not in visited and neighbor in self.hex_with_nodes:
                    queue.append(neighbor)

        return path

    def get_neighbors(self, hex_center):
        # 获取相邻的六边形中心
        neighbors = []
        # 使用六边形相邻关系获取
        # 可以根据坐标系（如odd-r或轴坐标系）计算邻居
        return neighbors

    def plan_path(self):
        self.generate_hex_grid()  # 生成六边形网格
        self.generate_random_nodes()  # 随机生成节点
        path = self.bfs_plan()  # 规划路径
        return path

# 使用示例
planner = HexGridPlanner(a=100, b=100, n=20, d=10)
path = planner.plan_path()
print("规划的路径为：", path)
