from __future__ import annotations

import os
import sys

if __package__ is None and __name__ == "__main__":
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rl.evaluate.continuous_policies import main, plot_policy_comparison, summarise

__all__ = ["main", "plot_policy_comparison", "summarise"]


if __name__ == "__main__":
    main()
