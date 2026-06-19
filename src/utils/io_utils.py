# utils/io_utils.py

from pathlib import Path


def ensure_dirs(paths: dict):
    """
    Ensure all values in the paths dictionary are directories and exist.
    Ignores non-string or non-path-like values (like filenames).
    """
    for key, val in paths.items():
        try:
            path = Path(val)
            if "." not in path.name:  # crude heuristic: skip if looks like a file
                path.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            print(f"Could not create path for {key}: {val} ({e})")
