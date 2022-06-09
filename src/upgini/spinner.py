import threading
from typing import Optional, List
import time


class Spinner:

    DEFAULT_FRAMES = [
        "-",
        "\\",
        "|",
        "/"
    ]

    def __init__(self, frames: List[str] = DEFAULT_FRAMES, step_time: float = 0.2):
        self.stop = False
        self.frames = frames
        self.iterations = len(self.frames)
        self.step_time = step_time
        self.ok_msg = "Done"
        self.thread: Optional[threading.Thread] = None

    def __enter__(self) -> "Spinner":
        self.thread = threading.Thread(target=self.spin)
        self.thread.start()
        return self

    def __exit__(self, exc_type, exc_value, tb):
        self.stop = True
        if self.thread is not None:
            self.thread.join()
        if exc_value is not None:
            print(f"Failed: {exc_value}")
        else:
            print(self.ok_msg)

    def ok(self, msg: str):
        self.ok_msg = msg

    def spin(self):
        i = 0
        while not self.stop:
            print(self.frames[i % self.iterations], end="\r", flush=True)
            time.sleep(self.step_time)
            i += 1
