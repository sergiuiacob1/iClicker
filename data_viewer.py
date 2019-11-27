from PyQt5 import QtWidgets
from PyQt5 import QtCore



class DataViewer(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Data Viewer')

    def displaySample(self, item):
        label = QtWidgets.QLabel("This is a PyQt5 window!")
        label.setAlignment(QtCore.Qt.AlignCenter)
        self.setCentralWidget(label)
        self.show()