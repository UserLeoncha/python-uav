import numpy as np

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