import os


def default_n_jobs() -> int:
    """Return a sensible default worker count for CPU-bound parallel work.

    On Python 3.13+, uses ``os.process_cpu_count()`` so the value reflects CPUs
    usable by the current process (affinity, container limits, ``PYTHON_CPU_COUNT``).
    On earlier versions, falls back to ``os.cpu_count()``.
    """
    if hasattr(os, "process_cpu_count"):
        count = os.process_cpu_count()
    else:
        count = os.cpu_count()
    return count or 1
