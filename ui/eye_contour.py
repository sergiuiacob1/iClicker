from PyQt5 import QtWidgets, QtGui, QtCore


class EyeContour(QtWidgets.QWidget):
    def __init__(self, parent):
        QtWidgets.QWidget.__init__(self, parent)
        self.points = None
        self.setFixedSize(parent.size())

    def paintEvent(self, event):
        if self.points is None:
            return
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        painter.setPen(QtGui.QPen(QtCore.Qt.blue, 1))
        # sort points py x ascending, y descending (so they go counter clockwise if iterated)
        # TODO do points need to be sorted?
        # self.points = sort_points_counter_clockwise(self.points)
        for i in range(0, len(self.points) - 1):
            painter.drawLine(self.points[i][0], self.points[i]
                             [1], self.points[i + 1][0], self.points[i + 1][1])
        # complete the circle
        painter.drawLine(
            self.points[-1][0], self.points[-1][1], self.points[0][0], self.points[0][1])


# TODO do i need this?
def sort_points_counter_clockwise(points):
    if len(points) == 0:
        return []

    def point_comparator(a, b):
        a1 = (math.degrees(math.atan2(
            a[0] - x_center, a[1] - y_center)) + 360) % 360
        a2 = (math.degrees(math.atan2(
            b[0] - x_center, b[1] - y_center)) + 360) % 360
        return (int)(a1 - a2)

    x_center = sum([x[0] for x in points])/len(points)
    y_center = sum([x[1] for x in points])/len(points)

    return sorted(points, key=functools.cmp_to_key(point_comparator))
