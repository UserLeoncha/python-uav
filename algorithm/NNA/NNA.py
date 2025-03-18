import sys
import random
import math
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton
from PyQt5.QtGui import QPainter, QPen
from PyQt5.QtCore import Qt, QPointF


class PointGenerator(QWidget):
    def __init__(self, a, b, c):
        super().__init__()
        self.a = a  # Map width
        self.b = b  # Map height
        self.c = c  # Number of random points to generate
        self.points = []  # List to store the generated random points
        self.path = []  # List to store the calculated shortest path
        self.initUI()

    def initUI(self):
        """Initialize the user interface"""
        self.setGeometry(100, 100, self.a, self.b)  # Set the window's position and size
        self.setWindowTitle('Point Generator')

        layout = QVBoxLayout()  # Vertical layout

        # Create a button that generates random points and finds the shortest path when clicked
        generateButton = QPushButton('Generate Points and Find Path', self)
        generateButton.clicked.connect(self.generatePointsAndFindPath)
        layout.addWidget(generateButton)

        self.setLayout(layout)  # Apply the layout to the window

    def generatePointsAndFindPath(self):
        """Generate random points and calculate the shortest path"""
        # Generate c random points within the range [0, a] and [0, b]
        self.points = [(random.randint(0, self.a), random.randint(0, self.b)) for _ in range(self.c)]

        # Add the start and end points
        start_point = (0, 0)
        end_point = (self.a, self.b)
        self.points.insert(0, start_point)
        self.points.append(end_point)

        # Calculate the shortest path using the nearest neighbor algorithm
        self.path = self.nearestNeighborPath(start_point) + [end_point]

        self.update()  # Trigger a repaint event

    def nearestNeighborPath(self, start_point):
        """Calculate the shortest path using the nearest neighbor algorithm"""
        unvisited = self.points[1:]  # Remaining unvisited points (excluding the start point)
        path = [start_point]
        current_point = start_point

        while unvisited:
            # Find the nearest unvisited point to the current point
            nearest_point = min(unvisited, key=lambda p: self.distance(current_point, p))
            path.append(nearest_point)
            unvisited.remove(nearest_point)
            current_point = nearest_point

        return path

    def distance(self, p1, p2):
        """Calculate the Euclidean distance between two points"""
        return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

    def paintEvent(self, event):
        """Handle the paint event"""
        qp = QPainter()  # Create a QPainter object
        qp.begin(self)  # Begin drawing
        self.drawPath(qp)  # Call the method to draw the path
        qp.end()  # End drawing

    def drawPath(self, qp):
        """Draw the shortest path"""
        if not self.path:
            return

        qp.setPen(QPen(Qt.black, 1, Qt.SolidLine))  # Set the pen with line width 1

        # Draw the shortest path
        for i in range(len(self.path) - 1):
            current_point = QPointF(self.path[i][0], self.path[i][1])
            next_point = QPointF(self.path[i + 1][0], self.path[i + 1][1])
            qp.drawEllipse(current_point, 2, 2)  # Draw the points
            qp.drawLine(current_point, next_point)  # Draw the line from the current point to the next point


if __name__ == '__main__':
    app = QApplication(sys.argv)  # Create an application object
    a, b, c = 1000, 1000, 100  # Set the map size and number of random points
    ex = PointGenerator(a, b, c)  # Create a PointGenerator object
    ex.show()  # Show the window
    sys.exit(app.exec_())  # Enter the main loop of the application
