from __future__ import annotations

import os
import sys

if __package__ is None and __name__ == "__main__":
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rl.analysis.compare_baselines import main

__all__ = ["main"]


if __name__ == "__main__":
    main()
