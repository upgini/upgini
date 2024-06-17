import importlib


class LazyImport:
    def __init__(self, module_name, class_name):
        self.module_name = module_name
        self.class_name = class_name
        self._module = None
        self._class = None

    def _load(self):
        if self._module is None:
            self._module = importlib.import_module(self.module_name)
            self._class = getattr(self._module, self.class_name)

    def __call__(self, *args, **kwargs):
        self._load()
        return self._class(*args, **kwargs)

    def __getattr__(self, name):
        self._load()
        return getattr(self._class, name)
