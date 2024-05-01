from typing import Tuple


class CustomFallbackProgressBar:
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
        self.text_width = 60
        self._stage = ""
        self._eta = ""

    def __repr__(self):
        fraction = self.progress / self.total
        filled = "=" * int(fraction * self.text_width)
        rest = " " * (self.text_width - len(filled))
        return f"[{filled}{rest}] {self.progress}% {self._stage} {self._eta}"

    def display(self):
        print(self)

    def update(self):
        print("\r")
        print(self)

    @property
    def progress(self):
        return self._progress

    @progress.setter
    def progress(self, value: Tuple[float, str, str]):
        self._progress = value[0]
        self._stage = value[1]
        self._eta = value[2]
        self.update()
