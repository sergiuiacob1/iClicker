from pynput.mouse import Listener


class MouseListener:
    def __init__(self, onClickFunction):
        self._listener = Listener(on_click=onClickFunction)

    def startListening(self):
        # non-blocking
        self._listener.start()

    def stopListening(self):
        self._listener.stop()
