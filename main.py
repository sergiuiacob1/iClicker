import sys
from PyQt5 import QtWidgets
from src.app import App
import logging

def main():
    # print every message that I log
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
    # tell the handler to use this format
    console.setFormatter(formatter)
    # add the handler to the root logger
    logging.getLogger('').addHandler(console)
    
    q_app = QtWidgets.QApplication(sys.argv)
    app = App()
    app.display_main_menu()
    sys.exit(q_app.exec())


if __name__ == '__main__':
    main()


# TODO use some tool to remove unused libraries at the end
# TODO add the dlib model on github