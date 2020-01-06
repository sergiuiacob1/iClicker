import sys
from PyQt5 import QtWidgets
from app import App

# TODO in init functions, only load resources that are REALLY necessary!!!
# TODO restructure everything with Model-View architecture


def main():
    q_app = QtWidgets.QApplication(sys.argv)
    app = App()
    app.display_main_menu()
    sys.exit(q_app.exec())


if __name__ == '__main__':
    main()
