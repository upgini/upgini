import os
from binascii import hexlify
from typing import Tuple

from IPython.display import DisplayObject, display


class CustomProgressBar(DisplayObject):
    """Progressbar supports displaying a progressbar like element"""

    def __init__(self, total=100):
        """Creates a new progressbar

        Parameters
        ----------
        total : int
            maximum size of the progressbar
        """
        self.total = total
        self._progress = 0
        self.html_width = "60ex"
        self.text_width = 60
        self._display_id = hexlify(os.urandom(8)).decode("ascii")
        self._stage = ""
        self._eta = ""

    def __repr__(self):
        fraction = self.progress / self.total
        filled = "=" * int(fraction * self.text_width)
        rest = " " * (self.text_width - len(filled))
        return "[{}{}] {}% {}".format(filled, rest, self.progress, self._stage)

    def _repr_html_(self):
        return "<progress style='width:{}' max='{}' value='{}'></progress>  {}% {}</br>{}".format(
            self.html_width, self.total, self.progress, int(self.progress), self._stage, self._eta
        )

    def display(self):
        display(self, display_id=self._display_id)

    def update(self):
        display(self, display_id=self._display_id, update=True)

    @property
    def progress(self):
        return self._progress

    @progress.setter
    def progress(self, value: Tuple[float, str, str]):
        self._progress = value[0]
        self._stage = value[1]
        self._eta = value[2]
        self.update()
