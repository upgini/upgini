import importlib
import importlib.util
import importlib.machinery


class LazyImport:
    def __init__(self, module_name, class_name):
        self.module_name = module_name
        self.class_name = class_name
        self._module = None
        self._class = None

    def _load(self):
        if self._module is None:
            # Load module and save link to it
            spec = importlib.util.find_spec(self.module_name)
            if spec is None:
                raise ImportError(f"Module {self.module_name} not found")

            # Create module
            self._module = importlib.util.module_from_spec(spec)

            # Execute module
            spec.loader.exec_module(self._module)

            # Get class from module
            self._class = getattr(self._module, self.class_name)

    def __call__(self, *args, **kwargs):
        self._load()
        return self._class(*args, **kwargs)

    def __getattr__(self, name):
        self._load()
        return getattr(self._class, name)
