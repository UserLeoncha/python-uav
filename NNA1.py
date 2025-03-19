import random
import matplotlib.pyplot as plt
from scipy.spatial import distance
import tkinter as tk
import tsp_solver.greedy as greedy

# 生成100个随机点
random_points = [(random.randint(0, 100), random.randint(0, 100)) for _ in range(1000)]

# 添加起点和终点
points = [(100, 100)] + random_points + [(0, 0)]

# 计算点之间的距离矩阵
distances = distance.cdist(points, points, 'euclidean')

# 使用贪心算法求解TSP问题
path = greedy.solve_tsp(distances)

# 绘制最短路径
plt.figure(figsize=(10, 10))
plt.scatter(*zip(*points), color='blue')
plt.plot(*zip(*[points[i] for i in path]), color='red')
plt.show()

# 使用tkinter展示GUI
root = tk.Tk()
root.title("无人机最短路径")

canvas = tk.Canvas(root, width=1000, height=1000)
canvas.pack()

# 将坐标映射到画布上
def map_to_canvas(x, y):
    return x * 10, 1000 - y * 10

# 绘制点
for point in points:
    canvas.create_oval(map_to_canvas(point[0], point[1]), map_to_canvas(point[0]+1, point[1]+1), fill='blue')

# 绘制路径
for i in range(len(path)-1):
    canvas.create_line(map_to_canvas(points[path[i]][0], points[path[i]][1]),
                       map_to_canvas(points[path[i+1]][0], points[path[i+1]][1]), fill='red')

root.mainloop()
