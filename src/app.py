from PyQt5 import QtWidgets, QtCore
import random
import os
import numpy as np
import joblib
import sys
import time
import logging


# My files
from src import trainer as Trainer
from src.data_collector import DataCollector, DataObject
from src.data_processing import main as data_processing_main, dp_logger, can_process_data
from src.data_viewer import DataViewer
from src.utils import get_screen_dimensions, run_function_on_thread
from src.predictor import Predictor
import config as Config

class QTextEditLogger(logging.Handler, QtCore.QObject):
    appendPlainText = QtCore.pyqtSignal(str)

    def __init__(self, parent):
        super().__init__()
        QtCore.QObject.__init__(self)
        self.widget = QtWidgets.QPlainTextEdit(parent)
        self.widget.setReadOnly(True)
        self.appendPlainText.connect(self.widget.appendPlainText)

    def emit(self, record):
        msg = self.format(record)
        self.appendPlainText.emit(msg)

class App(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.data_collector = DataCollector()
        self.data_viewer = DataViewer()
        self.predictor = Predictor()

    def keyPressEvent(self, e):
        if e.key() == QtCore.Qt.Key_Escape:
            self.close()

    def create_log_widget(self):
        logTextBox = QTextEditLogger(self)
        logTextBox.setFormatter(logging.Formatter('%(asctime)s: %(message)s'))
        logging.getLogger().addHandler(logTextBox)
        logging.getLogger().setLevel(logging.DEBUG)

        return logTextBox.widget


    def display_main_menu(self):
        self.setWindowTitle('iClicker')
        # self.resize(*get_screen_dimensions())
        self.resize(800, 600)
        # creating layout
        main_widget = QtWidgets.QWidget()
        main_widget.setLayout(QtWidgets.QVBoxLayout())
        self.top_menu_part = QtWidgets.QWidget()
        self.top_menu_part.setLayout(QtWidgets.QVBoxLayout())
        # self.top_menu_part.layout().addWidget(QtWidgets.QLabel('iClicker'))
        self.log_widget = self.create_log_widget()
        self.top_menu_part.layout().addWidget(self.log_widget)
        self.top_menu_part.resize(100, 200)

        self.bottom_menu_part = QtWidgets.QWidget()
        main_widget.layout().addWidget(self.top_menu_part)
        main_widget.layout().addWidget(self.bottom_menu_part)
        self.add_control_buttons()
        self.setCentralWidget(main_widget)
        self.show()

    def add_control_buttons(self):
        self.bottom_menu_part.setLayout(QtWidgets.QGridLayout())
        collect_data_button = QtWidgets.QPushButton('Colectare date')
        collect_data_button.setToolTip('Colectează date')
        collect_data_button.clicked.connect(self.collect_data)

        process_data_button = QtWidgets.QPushButton('Procesare date')
        process_data_button.setToolTip('Procesează datele colectate')
        process_data_button.clicked.connect(self.process_collected_data)

        train_button = QtWidgets.QPushButton('Antrenare model')
        train_button.setToolTip('Antrenează modelul bazat pe datele procesate')
        train_button.clicked.connect(self.train_model)

        predict_button = QtWidgets.QPushButton('Simulare')
        predict_button.setToolTip('Simulează funcționalitățile mouse-ului')
        predict_button.clicked.connect(self.predictor.start)

        # view_data_button = QtWidgets.QPushButton('View data')
        # view_data_button.setToolTip('View collected data')
        # view_data_button.clicked.connect(self.view_data)

        buttons = [collect_data_button, process_data_button, train_button, predict_button]
        for i in range(0, 2):
            for j in range(0, 2):
                self.bottom_menu_part.layout().addWidget(
                    buttons[i * 2 + j], i, j)

    def process_collected_data(self):
        if can_process_data() == False:
            self.error_dialog = QtWidgets.QErrorMessage()
            self.error_dialog.showMessage('Nu există date pentru a putea fi procesate.\n Apăsați pe butonul "Colectare date" mai întâi pentru a obține date.')
            return
        run_function_on_thread(data_processing_main)
        
    def view_data(self):
        print('Getting collected data...')
        data = self.data_collector.get_collected_data()
        print(f'Displaying random photos from {len(data)} samples')
        self.data_viewer.view_data(data)

    def train_model(self):
        if Trainer.can_train_model() == False:
            self.error_dialog = QtWidgets.QErrorMessage()
            self.error_dialog.showMessage('Nu există date procesate pentru a putea antrena un model.\n Apăsați pe butonul "Procesare date" pentru a procesa datele.')
            return
        run_function_on_thread(Trainer.main)

    def collect_data(self):
        self.data_collector.collect_data()
