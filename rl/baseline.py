from __future__ import annotations

import os
import sys

if __package__ is None and __name__ == "__main__":
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rl.baselines.discrete import heuristic_policy, random_policy, run_heuristic_episode, run_policy_episode

__all__ = [
    "random_policy",
    "heuristic_policy",
    "run_policy_episode",
    "run_heuristic_episode",
]


if __name__ == "__main__":
    summary = run_heuristic_episode()
    for key, value in summary.items():
        print(f"{key}: {value}")
