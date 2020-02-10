import sys
from PyQt5 import QtWidgets
from src.app import App

# TODO in init functions, only load resources that are REALLY necessary!!!
# TODO restructure everything with Model-View architecture

# TODO use binary threshold for eye images?
# TODO glob ocular extins

# TODO LBP (Local Binary Patterns)


def main():
    q_app = QtWidgets.QApplication(sys.argv)
    app = App()
    app.display_main_menu()
    sys.exit(q_app.exec())


if __name__ == '__main__':
    main()



# TODO use some tool to remove unused libraries at the end