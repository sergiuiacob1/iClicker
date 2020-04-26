import sys
from PyQt5 import QtWidgets
from src.app import App
from src.utils import attach_logger_to_stdout

def main():
    attach_logger_to_stdout()
    
    q_app = QtWidgets.QApplication(sys.argv)
    app = App()
    app.display_main_menu()
    sys.exit(q_app.exec())


if __name__ == '__main__':
    main()


# TODO use some tool to remove unused libraries at the end
# TODO add the dlib model on github