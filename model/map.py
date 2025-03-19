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
    def __init__(self, length=100, width=100, hex_size=5, point_num = 1000):
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
        self.point_num = point_num
        
        # 存储数据
        self.points = []
        self.grids = {(0,0)}  # 使用字典存储栅格，键为(q, r)坐标
        
        # 存储算法路径结果
        self.paths = {}
        self.performance = {}
        
        # 生成随机点
        self.generate_points(1000)

        # 确定栅格尺寸
        self._calculate_grid_dimensions()
    
    def generate_nodes():
        return [(np.random.uniform(0, self.length), np.random.uniform(0, self.width)) for _ in range(n)]

    def generate_points(self, n):
        