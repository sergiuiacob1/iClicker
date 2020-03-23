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
from enum import Enum

# My files
from src.mouse_listener import MouseListener
import src.webcam_capturer as WebcamCapturer
from src.data_object import DataObject
import config as Config
import src.utils as Utils
# ui
from src.ui.data_collector_gui import DataCollectorGUI


screen_width, screen_height = Utils.get_screen_dimensions()


class DataCollectionType:
    BACKGROUND = 1
    ACTIVE = 2


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

    def start_collecting(self, collection_type=DataCollectionType.BACKGROUND):
        # TODO document background/active
        print(f'Start collecting data in {collection_type} mode')
        WebcamCapturer.start_capturing()
        self.gui.start()
        print('DataCollectorGUI started')
        if collection_type == DataCollectionType.BACKGROUND:
            self.mouse_listener.start_listening()
            print('Mouse listener started')
        elif collection_type == DataCollectionType.ACTIVE:
            threading.Thread(target=self.start_active_collection).start()

    def start_active_collection(self):
        mouse_controller = pynput.mouse.Controller()
        step_x = int(screen_width/100)
        step_y = int(screen_height/25)
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

        print('Collecting data items ended')
        assert (self.collect_data_lock.locked(
        ) == False), "The lock for data collection should be unreleased, because I finished collecting items"

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
        print(
            f'Acquiring lock for data collection. Locked = {self.collect_data_lock.locked()}')
        self.collect_data_lock.acquire()
        print('Lock acquired')
        session_no = self.get_session_number()
        print(f"Saving data for session_{session_no}")
        self.save_session_info(session_no)
        self.save_images_info(session_no)
        self.save_images(session_no)
        self.collect_data_lock.release()
        print('Saving data done')

    def save_images(self, session_no):
        print('\n')
        dir_path = os.path.join(Config.data_directory_path, 'images')
        os.makedirs(dir_path, exist_ok=True)
        threads = []
        start = 0
        diff = len(self.collected_data) // os.cpu_count()
        for i in range(0, os.cpu_count() - 1):
            t = threading.Thread(target=self.save_images_per_thread, args=(
                session_no, start, start + diff, i + 1))
            threads.append(t)
            start += diff
        # last thread takes what's left
        t = threading.Thread(target=self.save_images_per_thread, args=(
            session_no, start, len(self.collected_data), os.cpu_count()))
        threads.append(t)
        # start all threads
        for t in threads:
            t.start()
        # wait for all threads
        for t in threads:
            t.join()
        print('\n')

    def save_images_per_thread(self, session_no, start, end, thread_no):
        """This is run on a different thread; only saves images from [start, end] indexes"""
        dir_path = os.path.join(Config.data_directory_path, 'images')
        print(f'Thread {thread_no}: Saving images from {start} to {end}')
        time.sleep(0.01)  # just to make prints prettier
        last_print = time.time()
        for (index, obj) in enumerate(self.collected_data[start:end]):
            path = os.path.join(dir_path, f'{session_no}_{index}.png')
            cv2.imwrite(path, obj.image)
            # every 2 seconds, print progress
            if time.time() - last_print >= 2:
                last_print = time.time()
                print(
                    f'Thread {thread_no}: saved {index}/{end - start} images')

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
        assert (self.collect_data_lock.locked() ==
                False), "This lock should be false"
        threading.Thread(target=self.save_collected_data).start()

    def on_mouse_click(self, x, y, button, pressed):
        """This function is only used when the type of data collection is DataCollectionType.BACKGROUND"""
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
