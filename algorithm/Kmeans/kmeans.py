import random
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.spatial import distance
import numpy as np
import tkinter as tk

# 生成100个随机点
random_points = [(random.randint(0, 100), random.randint(0, 100)) for _ in range(100)]

# 添加起点和终点
points = [(100, 100)] + random_points + [(0, 0)]

# 定义一个函数来检查聚类直径是否满足条件
def check_clusters_diameter(clustered_points):
    for cluster in clustered_points:
        if not cluster:  # 如果聚类为空，跳过
            continue
        max_distance = 0
        for point1 in cluster:
            for point2 in cluster:
                max_distance = max(max_distance, distance.euclidean(point1, point2))
        if max_distance > 10:
            return False
    return True

# 使用KMeans进行聚类，直到满足直径条件
max_iter = 100  # 防止无限循环
for _ in range(max_iter):
    kmeans = KMeans(n_clusters=5, random_state=0).fit(random_points)
    clusters = kmeans.labels_
    clustered_points = [[] for _ in range(5)]
    for i, point in enumerate(random_points):
        clustered_points[clusters[i]].append(point)
    if check_clusters_diameter(clustered_points):
        break
else:
    raise ValueError("未能满足聚类直径条件")

# 绘制聚类结构图
plt.figure(figsize=(10, 10))
for i, (label, point) in enumerate(zip(clusters, random_points)):
    plt.scatter(point[0], point[1], color=plt.cm.nipy_spectral(label / 10.0), s=50)
plt.scatter(points[0][0], points[0][1], color='red', s=100, label='Start')
plt.scatter(points[-1][0], points[-1][1], color='green', s=100, label='End')
plt.title('Cluster Structure')
plt.legend()
plt.show()

# 最近邻算法实现
def nearest_neighbor(points, start_point):
    path = [start_point]
    unvisited = points.copy()
    current = start_point
    while unvisited:
        nearest = min(unvisited, key=lambda point: distance.euclidean(current, point))
        path.append(nearest)
        unvisited.remove(nearest)
        current = nearest
    return path

# 对每个聚类结果应用最近邻算法
cluster_paths = []
start_point = points[0]
for cluster in clustered_points:
    path = nearest_neighbor(cluster, start_point)
    cluster_paths.append(path)
    start_point = path[-1]  # 更新起点为当前聚类的终点

# 连接所有聚类路径
full_path = [points[0]] + [point for cluster_path in cluster_paths for point in cluster_path[1:]] + [points[-1]]

# 绘制最短路径
plt.figure(figsize=(10, 10))
plt.scatter(*zip(*points), color='blue')
plt.plot(*zip(*full_path), color='red')
plt.show()

# 使用tkinter展示GUI
root = tk.Tk()
root.title("无人机聚类路径")

canvas = tk.Canvas(root, width=1000, height=1000)
canvas.pack()

# 将坐标映射到画布上
def map_to_canvas(x, y):
    return x * 10, 1000 - y * 10

# 绘制点
for point in points:
    canvas.create_oval(map_to_canvas(point[0], point[1]), map_to_canvas(point[0]+1, point[1]+1), fill='blue')

# 绘制路径
for i in range(len(full_path)-1):
    canvas.create_line(map_to_canvas(full_path[i][0], full_path[i][1]),
                       map_to_canvas(full_path[i+1][0], full_path[i+1][1]), fill='red')

root.mainloop()
