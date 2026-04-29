from __future__ import annotations

import os
import sys

if __package__ is None and __name__ == "__main__":
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rl.train.dqn_saoi import build_parser, train

__all__ = ["build_parser", "train"]


if __name__ == "__main__":
    train(build_parser().parse_args())
