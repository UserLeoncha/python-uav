import random
import numpy as np

class RandomPointsGenerator:
    def __init__(self, a, b, n):
        self.a = a  # 地图的宽度
        self.b = b  # 地图的高度
        self.n = n  # 节点数量

    def uniform_distribution(self):
        """生成均匀分布的n个点。"""
        return [(random.uniform(0, self.a), random.uniform(0, self.b)) for _ in range(self.n)]

    def clustered_distribution(self, clusters=5):
        """生成聚集分布的n个点，聚集在指定数量的簇中。"""
        points = []
        # 随机生成聚集中心
        cluster_centers = [(random.uniform(0, self.a), random.uniform(0, self.b)) for _ in range(clusters)]
        for _ in range(self.n):
            center_x, center_y = random.choice(cluster_centers)
            # 使用正态分布生成偏移量，控制点围绕聚集中心分布
            offset_x, offset_y = np.random.normal(0, self.a * 0.1), np.random.normal(0, self.b * 0.1)
            x = min(max(center_x + offset_x, 0), self.a)  # 确保点在地图范围内
            y = min(max(center_y + offset_y, 0), self.b)
            points.append((x, y))
        return points

    def generate(self, mode="uniform"):
        """根据传入模式生成点，支持'mode'为 'uniform' 或 'clustered'。"""
        if mode == "clustered":
            return self.clustered_distribution()
        else:
            return self.uniform_distribution()

# 示例
if __name__ == "__main__":
    generator = RandomPointsGenerator(a=100, b=100, n=1000)
    points = generator.generate(mode="clustered")
    print(points[:10])  # 打印前10个点作为样例