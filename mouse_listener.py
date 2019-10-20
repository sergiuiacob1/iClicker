from pynput.mouse import Listener
import logging
import asyncio


class MouseListener:
    def __init__(self, onClickFunction):
        self.onClickFunction = onClickFunction
        logging.basicConfig(filename=("mouse_logs.txt"), level=logging.DEBUG,
                            format='%(asctime)s: %(message)s')

    def startListening(self):
        with Listener(on_click=self.onClickFunction) as listener:
            listener.join()
