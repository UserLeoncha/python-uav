import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count
import time
import os
from datetime import datetime
import json
from model.Point import Point
from model.Grid import Grid

class Map:
    """
    地图类，包含点和六边形栅格，是整个系统的主要类
    """
    def __init__(self, length, width, hex_size=1):
        """
        初始化地图
        
        参数:
            length: float - 地图长度
            width: float - 地图宽度
            hex_size: float - 六边形栅格大小
        """
        self.length = length
        self.width = width
        self.hex_size = hex_size
        self.density = None # 点的密度函数，暂时为0

        # 存储数据
        self.points = []
        self.grids = {(0,0)}  # 使用字典存储栅格，键为(q, r)坐标
        
        # 存储算法路径结果
        self.paths = {}
        self.performance = {}
        
        # 确定栅格尺寸
        self._calculate_grid_dimensions()
    
    def _calculate_grid_dimensions(self):
        """
        计算地图需要的六边形栅格数量
        """
        self.grid_width = int(self.length / (self.hex_size * 1.5)) + 1
        self.grid_height = int(self.width / (self.hex_size * np.sqrt(3))) + 1
        
        # 创建空栅格
        self._create_grids()
        
    def _create_grids(self):
        """
        创建六边形栅格网络
        """
        for q in range(self.grid_width):
            for r in range(self.grid_height):
                self.grids[(q, r)] = Grid(q, r, self.hex_size)
    
    def generate_points(self, num_points, density=None):
        """
        生成随机点
        
        参数:
            num_points: int - 生成点的数量
            density: float or function - 生成点的密度或密度函数
        """
        self.points = []
        
        if density is None:
            # 均匀分布
            x_coords = np.random.uniform(0, self.length, num_points)
            y_coords = np.random.uniform(0, self.width, num_points)
        else:
            # 基于密度函数的分布（如果density是函数）
            if callable(density):
                points_generated = 0
                while points_generated < num_points:
                    x = np.random.uniform(0, self.length)
                    y = np.random.uniform(0, self.width)
                    
                    # 接受-拒绝采样
                    if np.random.random() < density(x, y):
                        x_coords = np.append(x_coords, x)
                        y_coords = np.append(y_coords, y)
                        points_generated += 1
            else:
                # 如果density是数值，在高密度区域生成更多点
                # 这里简单实现为中心区域高密度
                x_coords = []
                y_coords = []
                center_x = self.length / 2
                center_y = self.width / 2
                max_dist = np.sqrt(center_x**2 + center_y**2)
                
                for _ in range(num_points):
                    while True:
                        x = np.random.uniform(0, self.length)
                        y = np.random.uniform(0, self.width)
                        dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
                        prob = 1 - (dist / max_dist) * (1 - density)
                        
                        if np.random.random() < prob:
                            x_coords.append(x)
                            y_coords.append(y)
                            break
        
        # 创建点对象
        for i in range(num_points):
            self.points.append(Point(x_coords[i], y_coords[i], i))
        
        # 将点分配到对应的栅格中
        self._assign_points_to_grids()
        
        return self.points
        
    def _assign_points_to_grids(self):
        """
        将所有点分配到对应的六边形栅格中
        """
        for point in self.points:
            grid_coord = self._pixel_to_hex(point.x, point.y)
            if grid_coord in self.grids:
                self.grids[grid_coord].add_point(point)
                
    def _pixel_to_hex(self, x, y):
        """
        将平面直角坐标转换为六边形栅格坐标
        """
        q = (2/3) * x / self.hex_size
        r = (-1/3) * x / self.hex_size + (np.sqrt(3)/3) * y / self.hex_size
        
        # 取最近的六边形栅格（四舍五入到整数）
        q_rounded = round(q)
        r_rounded = round(r)
        
        return (q_rounded, r_rounded)
    
    def run_algorithm(self, algorithm_name, algorithm_func, *args, **kwargs):
        """
        运行路径规划算法
        
        参数:
            algorithm_name: str - 算法名称
            algorithm_func: function - 算法函数
            *args, **kwargs - 传递给算法函数的参数
        """
        start_time = time.time()
        path = algorithm_func(self, *args, **kwargs)
        end_time = time.time()
        
        # 存储路径和性能信息
        self.paths[algorithm_name] = path
        self.performance[algorithm_name] = {
            "runtime": end_time - start_time,
            "path_length": self._calculate_path_length(path),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        return path, self.performance[algorithm_name]
    
    def _calculate_path_length(self, path):
        """
        计算路径长度
        """
        if not path or len(path) < 2:
            return 0
            
        length = 0
        for i in range(len(path) - 1):
            length += path[i].distance_to(path[i+1])
            
        return length
    
    def evaluate_performance(self, baseline_algorithm=None):
        """
        评估算法性能
        
        参数:
            baseline_algorithm: str - 基准算法名称，如果为None则使用第一个算法
        """
        if not self.performance:
            return {}
        
        baseline = baseline_algorithm if baseline_algorithm in self.performance else list(self.performance.keys())[0]
        baseline_length = self.performance[baseline]["path_length"]
        baseline_time = self.performance[baseline]["runtime"]
        
        results = {}
        for alg, perf in self.performance.items():
            results[alg] = {
                "absolute_runtime": perf["runtime"],
                "absolute_length": perf["path_length"],
                "relative_runtime": perf["runtime"] / baseline_time,
                "relative_length": perf["path_length"] / baseline_length,
                "efficiency": baseline_length / perf["path_length"] * baseline_time / perf["runtime"]
            }
            
        return results
    
    def save_results(self, folder_name=None):
        """
        保存结果到文件夹
        
        参数:
            folder_name: str - 文件夹名称，如果为None则使用当前时间
        """
        if folder_name is None:
            folder_name = datetime.now().strftime("%Y%m%d_%H%M%S")
            
        folder_path = os.path.join("results", folder_name)
        os.makedirs(folder_path, exist_ok=True)
        
        # 保存性能数据
        with open(os.path.join(folder_path, "performance.json"), "w") as f:
            json.dump(self.performance, f, indent=2)
            
        # 保存评估结果
        evaluation = self.evaluate_performance()
        with open(os.path.join(folder_path, "evaluation.json"), "w") as f:
            json.dump(evaluation, f, indent=2)
            
        # 生成结果图
        self.plot_results(save_path=folder_path)
        
        return folder_path
    
    def plot_results(self, save_path=None):
        """
        绘制结果图表
        
        参数:
            save_path: str - 保存路径，如果为None则显示图表
        """
        if not self.paths:
            return
            
        # 找出最好和最差的结果
        evaluation = self.evaluate_performance()
        sorted_algs = sorted(evaluation.keys(), key=lambda x: evaluation[x]["efficiency"])
        best_alg = sorted_algs[-1]  # 效率最高的
        worst_alg = sorted_algs[0]   # 效率最低的
        
        # 绘制最好的路径
        plt.figure(figsize=(12, 10))
        
        # 绘制第一个子图 - 最佳路径
        plt.subplot(2, 1, 1)
        plt.title(f"最佳路径: {best_alg}")
        self._plot_path(self.paths[best_alg])
        
        # 绘制第二个子图 - 最差路径
        plt.subplot(2, 1, 2)
        plt.title(f"最差路径: {worst_alg}")
        self._plot_path(self.paths[worst_alg])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(os.path.join(save_path, "paths_comparison.png"))
        else:
            plt.show()
            
        # 绘制性能对比图
        self._plot_performance_comparison(save_path)
    
    def _plot_path(self, path):
        """
        绘制路径
        """
        # 绘制所有点
        x_points = [p.x for p in self.points]
        y_points = [p.y for p in self.points]
        plt.scatter(x_points, y_points, c='gray', s=10, alpha=0.5)
        
        # 绘制路径
        x_path = [p.x for p in path]
        y_path = [p.y for p in path]
        plt.plot(x_path, y_path, 'r-', linewidth=1.5)
        
        # 标记起点和终点
        plt.scatter([path[0].x], [path[0].y], c='g', s=100, marker='o')
        plt.scatter([path[-1].x], [path[-1].y], c='r', s=100, marker='x')
        
        # 绘制栅格（可选，如果栅格数量不太多）
        if len(self.grids) < 100:
            for grid in self.grids.values():
                vertices = grid.vertices + [grid.vertices[0]]  # 闭合六边形
                xs, ys = zip(*vertices)
                plt.plot(xs, ys, 'b-', linewidth=0.5, alpha=0.3)
        
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.grid(True, alpha=0.3)
    
    def _plot_performance_comparison(self, save_path=None):
        """
        绘制性能对比图
        """
        evaluation = self.evaluate_performance()
        algs = list(evaluation.keys())
        
        plt.figure(figsize=(12, 8))
        
        # 路径长度对比
        plt.subplot(2, 2, 1)
        plt.title("路径长度对比")
        plt.bar(algs, [evaluation[a]["absolute_length"] for a in algs])
        plt.xticks(rotation=45)
        plt.ylabel("路径长度")
        
        # 运行时间对比
        plt.subplot(2, 2, 2)
        plt.title("运行时间对比")
        plt.bar(algs, [evaluation[a]["absolute_runtime"] for a in algs])
        plt.xticks(rotation=45)
        plt.ylabel("运行时间 (秒)")
        
        # 相对效率对比
        plt.subplot(2, 2, 3)
        plt.title("相对效率