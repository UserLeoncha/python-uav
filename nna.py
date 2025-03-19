import sys
import math
import random
from PyQt5.QtGui import QPainter, QPen
from PyQt5.QtCore import Qt, QPointF
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton


class PointGenerator(QWidget):
    def __init__(self, a, b, c):
        super().__init__()
        self.a = a  # 地图的宽度
        self.b = b  # 地图的高度
        self.c = c  # 随机生成的点的数量
        self.points = []  # 存储生成的随机点
        self.path = []  # 存储计算出的最短路径
        self.initUI()

    def initUI(self):
        """初始化用户界面"""
        self.setGeometry(100, 100, self.a, self.b)  # 设置窗口的位置和大小
        self.setWindowTitle('Point Generator')

        layout = QVBoxLayout()  # 垂直布局

        # 创建按钮，点击时生成随机点并计算最短路径
        generateButton = QPushButton('Generate Points and Find Path', self)
        generateButton.clicked.connect(self.generatePointsAndFindPath)
        layout.addWidget(generateButton)

        self.setLayout(layout)  # 将布局应用到窗口

    def generatePointsAndFindPath(self):
        """生成随机点并计算最短路径"""
        # 在范围 [0, a] 和 [0, b] 内生成 c 个随机点
        self.points = [(random.randint(0, self.a), random.randint(0, self.b)) for _ in range(self.c)]

        # 加入起点和终点
        start_point = (0, 0)
        end_point = (self.a, self.b)
        self.points.insert(0, start_point)
        self.points.append(end_point)

        # 使用最近邻算法计算最短路径
        self.path = self.nearestNeighborPath()
        self.update()  # 触发重新绘制事件

    def nearestNeighborPath(self):
        """使用最近邻算法计算最短路径"""
        unvisited = self.points[:]  # 剩余未访问的点
        path = []
        current_point = unvisited.pop(0)  # 从起点开始
        path.append(current_point)

        while unvisited:
            # 寻找距离当前点最近的未访问点
            nearest_point = min(unvisited, key=lambda p: self.distance(current_point, p))
            path.append(nearest_point)
            unvisited.remove(nearest_point)
            current_point = nearest_point

        return path

    def distance(self, p1, p2):
        """计算两点之间的欧几里得距离"""
        return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

    def paintEvent(self, event):
        """处理绘制事件"""
        qp = QPainter()  # 创建 QPainter 对象
        qp.begin(self)  # 开始绘制
        self.drawPath(qp)  # 调用绘制路径的方法
        qp.end()  # 结束绘制

    def drawPath(self, qp):
        """绘制最短路径"""
        if not self.path:
            return

        qp.setPen(QPen(Qt.black, 1, Qt.SolidLine))  # 设置画笔，线宽为1

        # 绘制最短路径
        for i in range(len(self.path) - 1):
            current_point = QPointF(self.path[i][0], self.path[i][1])
            next_point = QPointF(self.path[i + 1][0], self.path[i + 1][1])
            qp.drawEllipse(current_point, 2, 2)  # 绘制点
            qp.drawLine(current_point, next_point)  # 绘制从当前点到下一个点的线


if __name__ == '__main__':
    app = QApplication(sys.argv)  # 创建应用程序对象
    a, b, c = 1000, 1000, 100  # 设置地图大小和随机点数量
    ex = PointGenerator(a, b, c)  # 创建 PointGenerator 对象
    ex.show()  # 显示窗口
    sys.exit(app.exec_())  # 进入应用程序主循环
