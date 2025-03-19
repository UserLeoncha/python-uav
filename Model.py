import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count
import time
import os
from datetime import datetime
import json

class Map:
    """
    地图类，包含点和六边形栅格，是整个系统的主要类
    """
    def __init__(self, length=100, width=100, hex_size=5, point_num=1000):
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
        self.density = None  # 点的密度函数，暂时为0
        self.point_num = point_num

        # 存储数据
        self.points = []
        self.grids = {(0, 0)}  # 使用字典存储栅格

        # 存储算法路径结果
        self.paths = {}
        self.performance = {}

        # 生成随机点
        self.generate_nodes(self.point_num)

        # 确定栅格尺寸
        # self._calculate_grid_dimensions()

    def generate_nodes(self, n):
        self.points = [
            Point(np.random.uniform(0, self.length), np.random.uniform(0, self.width))
            for _ in range(n)
        ]
        return self.points
    
class Point:
    """
    节点类，表示地图中的一个点
    """
    def __init__(self, x, y, id=None):
        """
        初始化一个点
        
        参数:
            x: float - 平面直角坐标x
            y: float - 平面直角坐标y
            id: int - 节点标识符（可选）
        """
        self.x = x
        self.y = y
        self.id = id
        self.grid_coord = None  # 所在六边形栅格坐标，由Map类分配
        
    def distance_to(self, other_point):
        """
        计算到另一个点的欧氏距离
        """
        return np.sqrt((self.x - other_point.x)**2 + (self.y - other_point.y)**2)
    
    def get_coordinates(self):
        """
        获取平面直角坐标
        """
        return np.array([self.x, self.y])
    
    def set_grid_coord(self, q, r):
        """
        设置所在六边形栅格坐标
        
        参数:
            q: int - 六边形栅格的q坐标
            r: int - 六边形栅格的r坐标
        """
        self.grid_coord = (q, r)
        
    def __str__(self):
        """
        字符串表示
        """
        if self.grid_coord:
            return f"点({self.x}, {self.y}) 位于六边形栅格{self.grid_coord}"
        else:
            return f"点({self.x}, {self.y})"
        
class Grid:
    """
    六边形栅格类，用于表示地图中的六边形栅格
    """
    def __init__(self, q, r, hex_size):
        """
        初始化六边形栅格

        参数:
            q: int - 六边形栅格在轴坐标系中的 q 坐标
            r: int - 六边形栅格在轴坐标系中的 r 坐标
            hex_size: float - 六边形的边长
        """
        self.q = q
        self.r = r
        self.hex_size = hex_size
        self.center = self.calculate_center()
        self.points = []  # 存储位于该栅格内的节点

    def calculate_center(self):
        """
        根据六边形轴坐标转换成像素坐标，计算并返回六边形栅格的中心坐标

        转换公式:
            x = hex_size * 3/2 * q
            y = hex_size * sqrt(3) * (r + q/2)
        """
        x = self.hex_size * 3/2 * self.q
        y = self.hex_size * np.sqrt(3) * (self.r + self.q/2)
        return (x, y)

    def add_point(self, point):
        """
        将一个节点添加到当前栅格中，并设置该节点的栅格坐标
        
        参数:
            point: Point 对象
        """
        self.points.append(point)
        point.set_grid_coord(self.q, self.r)

    def __str__(self):
        """
        返回当前栅格的字符串表示
        """
        return f"Grid(q={self.q}, r={self.r}, center={self.center}, points={len(self.points)})"