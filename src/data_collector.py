import logging
import threading
import joblib
import json
import os
import time
import numpy as np
from cv2 import cv2

# My files
from src.mouse_listener import MouseListener
from src.webcam_capturer import WebcamCapturer
from src.data_object import DataObject
import config as Config
import src.utils as Utils
# ui
from src.ui.data_collector_gui import DataCollectorGUI


screen_width, screen_height = Utils.get_screen_dimensions()


class DataCollector():
    def __init__(self, webcam_capturer: WebcamCapturer):
        super().__init__()
        self.collected_data = []
        self.mouse_listener = MouseListener(self.on_mouse_click)
        self.webcam_capturer = webcam_capturer
        logging.basicConfig(filename=("mouse_logs.txt"), level=logging.DEBUG,
                            format='%(asctime)s: %(message)s')
        self.collect_data_lock = threading.Lock()
        self.gui = DataCollectorGUI(self, webcam_capturer)

    def start_collecting(self):
        print('Started collecting training data...')
        self.gui.start()
        print ('GUI started')
        self.mouse_listener.start_listening()
        print ('Mouse listener started')
        self.webcam_capturer.start_capturing()
        print ('Webcam capturer started')

    def save_collected_data(self):
        if len(self.collected_data) == 0:
            return
        self.collect_data_lock.acquire()
        session_no = self.get_session_number()
        print (f"Saving data for session_{session_no}")
        self.save_session_info(session_no)
        self.save_images_info(session_no)
        self.save_images(session_no)
        self.collect_data_lock.release()

    def save_images(self, session_no):
        dir_path = os.path.join(Config.data_directory_path, 'images')
        os.makedirs(dir_path, exist_ok=True)
        for (index, obj) in enumerate(self.collected_data):
            path = os.path.join(dir_path, f'{session_no}_{index}.png')
            cv2.imwrite(path, obj.image)

    def save_images_info(self, session_no):
        """Saves the info about each captured image"""
        path = os.path.join(Config.data_directory_path, "sessions")
        os.makedirs(path, exist_ok=True)

        data = {}
        for (index, obj) in enumerate(self.collected_data):
            data[index] = {
                "mouse_position": list(obj.mouse_position),
            }
        with open(os.path.join(path, f'session_{session_no}.json'), 'w') as f:
            json.dump(data, f)

    def save_session_info(self, session_no):
        path = os.path.join(Config.data_directory_path, 'sessions.json')

        if os.path.exists(path) is False:
            open(path, 'w').write('{}')
        with open(path, 'r') as f:
            data = json.load(f)

        data['total_sessions'] = data.get('total_sessions', 0) + 1
        data[f'session_{session_no}'] = {
            "items": len(self.collected_data),
            "timestamp": time.time(),
            "screen_size": [screen_width, screen_height],
            "webcam_size": [Config.WEBCAM_IMAGE_WIDTH, Config.WEBCAM_IMAGE_HEIGHT],
        }
        with open(path, 'w') as f:
            json.dump(data, f)

    def get_session_number(self):
        path = self._get_data_directory_path()
        path = os.path.join(path, 'sessions.json')
        if os.path.exists(path) is False:
            return 1
        with open(path, 'r') as f:
            data = json.load(f)
        return data.get('total_sessions', 0) + 1

    def end_data_collection(self):
        print('Saving collected data...')
        threading.Thread(target=self.save_collected_data).start()
        self.mouse_listener.stop_listening()
        self.webcam_capturer.stop_capturing()

    def on_mouse_click(self, x, y, button, pressed):
        if pressed:
            logging.info(f'{button} pressed at ({x}, {y})')
            success, webcamImage = self.webcam_capturer.get_webcam_image()
            if success is True:
                dataObject = DataObject(
                    webcamImage, (x, y), (screen_width, screen_height))
                self.collect_data_lock.acquire()
                self.collected_data.append(dataObject)
                self.collect_data_lock.release()

    def _get_data_directory_path(self):
        dataDirectoryPath = Config.data_directory_path
        dataDirectoryPath = os.path.abspath(
            os.path.join(os.getcwd(), dataDirectoryPath))
        os.makedirs(dataDirectoryPath, exist_ok=True)
        return dataDirectoryPath
