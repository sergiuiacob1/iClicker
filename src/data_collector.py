import logging
import threading
import joblib
import json
import os
import sys
import time
import numpy as np
from cv2 import cv2
import pynput

# My files
from src.mouse_listener import MouseListener
import src.webcam_capturer as WebcamCapturer
from src.data_object import DataObject
import config as Config
import src.utils as Utils
# ui
from src.ui.data_collector_gui import DataCollectorGUI


screen_width, screen_height = Utils.get_screen_dimensions()


class DataCollector():
    def __init__(self):
        super().__init__()
        self.collected_data = []
        self.mouse_listener = MouseListener(self.on_mouse_click)
        logging.basicConfig(filename=("mouse_logs.txt"), level=logging.DEBUG,
                            format='%(asctime)s: %(message)s')
        self.collect_data_lock = threading.Lock()
        self.cursor_move_delay = 0.1
        self.is_paused = False
        self.pause_event = threading.Lock()
        self.gui = DataCollectorGUI(self)

    def start_collecting(self, collection_type="background"):
        # TODO document background/active
        print(f'Start collecting data in {collection_type} mode')
        WebcamCapturer.start_capturing()
        self.gui.start()
        print('DataCollectorGUI started')
        if collection_type == "background":
            self.mouse_listener.start_listening()
            print('Mouse listener started')
        else:
            threading.Thread(target=self.start_active_collection).start()

    def start_active_collection(self):
        mouse_controller = pynput.mouse.Controller()
        step_x = int(screen_width/100)
        step_y = int(screen_height/50)
        start, end = 0, screen_width

        # Move it to the corner first, starting from the center
        x, y = screen_width/2, screen_height/2
        animation_steps = 50
        dx = screen_width/2/animation_steps
        dy = screen_height/2/animation_steps
        mouse_controller.position = (x, y)
        for i in range(0, animation_steps):
            mouse_controller.position = (x, y)
            x -= dx
            y -= dy
            time.sleep(1/animation_steps)

        for y in range(0, screen_height, step_y):
            for x in range(start, end, step_x):
                print(self.gui)
                print(self.gui.isEnabled())
                print(self.gui.isVisible())
                print(self.gui.isActiveWindow())
                if self.gui.isVisible() == False:
                    return

                time.sleep(self.cursor_move_delay)

                acquired_lock = False
                if self.is_paused is True:
                    self.pause_event.acquire()
                    acquired_lock = True

                mouse_controller.position = (x, y)
                self.add_new_collected_item((x, y))

                if acquired_lock is True:
                    self.pause_event.release()

            step_x = -step_x
            start, end = end, start

        # If I reached this point and the gui isn't closed, auto-close it
        if self.gui.isVisible():
            self.gui.close()

    def increase_speed(self):
        print('Increasing speed')
        self.cursor_move_delay -= 0.005
        self.cursor_move_delay = max(0, self.cursor_move_delay)

    def decrease_speed(self):
        print('Decreasing speed')
        self.cursor_move_delay += 0.005

    def pause(self):
        print('Triggered pause')
        self.is_paused = not self.is_paused
        if self.is_paused:
            self.pause_event.acquire()
        else:
            self.pause_event.release()

    def save_collected_data(self):
        if len(self.collected_data) == 0:
            return
        self.collect_data_lock.acquire()
        session_no = self.get_session_number()
        print(f"Saving data for session_{session_no}")
        self.save_session_info(session_no)
        self.save_images_info(session_no)
        self.save_images(session_no)
        self.collect_data_lock.release()
        print('Saving data done')

    def save_images(self, session_no):
        dir_path = os.path.join(Config.data_directory_path, 'images')
        os.makedirs(dir_path, exist_ok=True)
        threads = []
        print('\n')
        for (index, obj) in enumerate(self.collected_data):
            path = os.path.join(dir_path, f'{session_no}_{index}.png')
            cv2.imwrite(path, obj.image)
            sys.stdout.write('\r')
            sys.stdout.write(
                f'Saved {index}/{len(self.collected_data)} images')

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
        print(f'Saving collected data: {len(self.collected_data)} items')
        self.is_paused = False
        if self.pause_event.locked():
            self.pause_event.release()

        self.mouse_listener.stop_listening()
        WebcamCapturer.stop_capturing()
        threading.Thread(target=self.save_collected_data).start()

    def on_mouse_click(self, x, y, button, pressed):
        """This function is only used when the type of data collection is \"background\""""
        if pressed:
            logging.info(f'{button} pressed at ({x}, {y})')
            self.add_new_collected_item((x, y))

    def add_new_collected_item(self, mouse_position):
        success, webcam_image = WebcamCapturer.get_webcam_image()
        if success is False:
            return
        item = DataObject(webcam_image, mouse_position,
                          (screen_width, screen_height))
        self.collect_data_lock.acquire()
        self.collected_data.append(item)
        self.collect_data_lock.release()

    def _get_data_directory_path(self):
        data_directory_path = Config.data_directory_path
        data_directory_path = os.path.abspath(
            os.path.join(os.getcwd(), data_directory_path))
        os.makedirs(data_directory_path, exist_ok=True)
        return data_directory_path
