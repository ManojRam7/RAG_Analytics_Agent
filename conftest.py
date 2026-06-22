"""Ensure the `src` layout is importable when running tests/scripts without install."""
import pathlib
import sys

SRC = pathlib.Path(__file__).resolve().parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
