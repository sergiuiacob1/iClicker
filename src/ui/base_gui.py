from PyQt5 import QtWidgets, QtCore

# TODO close all children of the main window

class BaseGUI(QtWidgets.QMainWindow):
    def __init__(self, controller):
        super().__init__()
        self.controller = controller
        self.create_window()

    def create_window(self):
        raise NotImplementedError

    def closeEvent(self, event):
        """This function is ran when the training data window is closed"""
        self.controller.ui_was_closed()

    def keyPressEvent(self, e):
        if e.key() == QtCore.Qt.Key_Escape:
            self.close()
