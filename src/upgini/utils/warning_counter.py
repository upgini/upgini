class WarningCounter:
    def __init__(self):
        self._count = 0

    def increment(self):
        self._count += 1
        return self._count

    def reset(self):
        self._count = 0

    def has_warnings(self):
        return self._count > 0
