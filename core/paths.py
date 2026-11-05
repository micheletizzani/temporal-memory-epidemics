"""Repository paths. Importing this module also makes core/ and every experiments/* folder importable."""
import glob
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for _d in [os.path.join(ROOT, "core")] + sorted(glob.glob(os.path.join(ROOT, "experiments", "*"))):
    if _d not in sys.path:
        sys.path.insert(0, _d)
for _d in ("results", "figures", "generated", "data"):
    os.makedirs(os.path.join(ROOT, _d), exist_ok=True)
