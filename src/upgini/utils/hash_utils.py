import hashlib
import os
import platform
import shutil
import subprocess
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd


def file_hash(path: str | os.PathLike, algo: str = "sha256") -> str:
    """
    Returns file hash using system utilities, working consistently on Windows/macOS/Linux.
    If no suitable utility is found, gracefully falls back to hashlib.

    Supported algo values (depend on OS and available utilities):
      - "md5", "sha1", "sha224", "sha256", "sha384", "sha512"
    On Windows uses `certutil`.
    On Linux uses `sha*sum` (e.g., sha256sum) or `shasum -a N`.
    On macOS uses `shasum -a N` or `md5` for MD5.
    """
    p = str(Path(path))

    sysname = platform.system().lower()
    algo = algo.lower()

    # -------- command attempts depending on OS --------
    candidates: list[list[str]] = []

    if sysname == "windows":
        # certutil supports: MD5, SHA1, SHA256, SHA384, SHA512
        name_map = {
            "md5": "MD5",
            "sha1": "SHA1",
            "sha224": None,  # certutil doesn't support
            "sha256": "SHA256",
            "sha384": "SHA384",
            "sha512": "SHA512",
        }
        cert_name = name_map.get(algo)
        if cert_name:
            candidates.append(["certutil", "-hashfile", p, cert_name])
    else:
        # Unix-like systems
        # 1) specialized *sum utility if available (usually present on Linux)
        sum_cmd = f"{algo}sum"  # md5sum, sha256sum, etc.
        if shutil.which(sum_cmd):
            candidates.append([sum_cmd, p])

        # 2) universal shasum with -a parameter (available on macOS and often on Linux)
        shasum_bits = {
            "sha1": "1",
            "sha224": "224",
            "sha256": "256",
            "sha384": "384",
            "sha512": "512",
        }
        if algo in shasum_bits and shutil.which("shasum"):
            candidates.append(["shasum", "-a", shasum_bits[algo], p])

        # 3) for MD5 on macOS there's often a separate `md5` utility
        if algo == "md5" and shutil.which("md5"):
            candidates.append(["md5", p])

    # -------- try system utilities --------
    for cmd in candidates:
        try:
            out = subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT)
            digest = _parse_hash_output(out, cmd[0])
            if digest:
                return digest.lower()
        except (subprocess.CalledProcessError, FileNotFoundError):
            continue  # try next candidate

    # -------- reliable fallback to hashlib --------
    import hashlib

    try:
        h = getattr(hashlib, algo)
    except AttributeError:
        raise ValueError(f"Algorithm not supported: {algo}")

    hasher = h()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest().lower()


def _parse_hash_output(output: str, tool: str) -> Optional[str]:
    """
    Converts output from different utilities to clean hash.
    Supports:
      - sha*sum / shasum: '<hex>  <filename>'
      - certutil (Windows): line with second element as hash (spaces inside are removed)
      - md5 (macOS): 'MD5 (file) = <hex>'
    """
    tool = tool.lower()
    lines = [ln.strip() for ln in output.splitlines() if ln.strip()]

    if not lines:
        return None

    if tool in {"sha1sum", "sha224sum", "sha256sum", "sha384sum", "sha512sum", "md5sum", "shasum"}:
        # format: '<hex>  <filename>'
        first = lines[0]
        parts = first.split()
        return parts[0] if parts else None

    if tool == "certutil":
        # format:
        # SHA256 hash of file <path>:
        # <AA BB CC ...>
        # CertUtil: -hashfile command completed successfully.
        if len(lines) >= 2:
            # Second line contains hex with spaces
            candidate = lines[1].replace(" ", "")
            # ensure it's hex
            if all(c in "0123456789abcdefABCDEF" for c in candidate):
                return candidate
        return None

    if tool == "md5":
        # format: 'MD5 (<file>) = <hex>'
        last = lines[-1]
        if "=" in last:
            return last.split("=", 1)[1].strip()
        # sometimes md5 can return just the hash
        parts = last.split()
        if parts and all(c in "0123456789abcdefABCDEF" for c in parts[-1]):
            return parts[-1]
        return None

    # as a last resort: take the first "looks like hash" word
    for ln in lines:
        for token in ln.split():
            if all(c in "0123456789abcdefABCDEF" for c in token) and len(token) >= 32:
                return token
    return None


def hash_input(X: pd.DataFrame, y: Optional[pd.Series] = None, eval_set: Optional[List[Tuple]] = None) -> str:
    hashed_objects = []
    try:
        hashed_objects.append(pd.util.hash_pandas_object(X, index=False).values)
        if y is not None:
            hashed_objects.append(pd.util.hash_pandas_object(y, index=False).values)
        if eval_set is not None:
            if isinstance(eval_set, tuple):
                eval_set = [eval_set]
            for eval_X, eval_y in eval_set:
                hashed_objects.append(pd.util.hash_pandas_object(eval_X, index=False).values)
                hashed_objects.append(pd.util.hash_pandas_object(eval_y, index=False).values)
        common_hash = hashlib.sha256(np.concatenate(hashed_objects)).hexdigest()
        return common_hash
    except Exception:
        return ""
