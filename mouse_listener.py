from pynput.mouse import Listener


class MouseListener:
    def __init__(self, on_click_function):
        self.on_click = on_click_function

    def start_listening(self):
        # non-blocking
        # a Listener can only be started once, so I must create a new one every time
        if hasattr(self, "_listener") and self._listener.running is True:
            self._listener.stop()
        self._listener = Listener(on_click=self.on_click)
        self._listener.start()

    def stop_listening(self):
        if hasattr(self, "_listener") and self._listener.running is True:
            self._listener.stop()
