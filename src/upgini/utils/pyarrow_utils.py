import warnings

PYARROW_RUNTIME_RESTART_MESSAGE = (
    "pyarrow was upgraded but the runtime still has an older version loaded. "
    "Restart the runtime after installing upgini (Runtime -> Restart session in Colab), "
    "then import upgini again."
)


def is_pyarrow_binary_incompatibility(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return "binary incompatibility" in msg or "size changed" in msg or "ipcreadoptions" in msg


def raise_pyarrow_runtime_error(exc: BaseException) -> None:
    warnings.warn(PYARROW_RUNTIME_RESTART_MESSAGE, UserWarning, stacklevel=3)
    raise ImportError(PYARROW_RUNTIME_RESTART_MESSAGE) from exc


def import_pyarrow_modules():
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except (ValueError, ImportError) as exc:
        if is_pyarrow_binary_incompatibility(exc):
            raise_pyarrow_runtime_error(exc)
        raise
    return pa, pq
