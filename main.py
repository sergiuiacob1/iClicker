import asyncio
import cv2
import time
import json
from pynput.mouse import Listener
import logging
from app import App


def main():
    app = App()
    app.collectTrainingData()
    input("Press Enter to continue...")


if __name__ == '__main__':
    main()
