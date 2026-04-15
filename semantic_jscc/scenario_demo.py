from __future__ import annotations

import argparse
import os
import sys

import torch

if __package__ is None and __name__ == "__main__":
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from semantic_jscc import DeepJSCCScenarioModule


def main() -> None:
    parser = argparse.ArgumentParser(description="DeepJSCC scenario module demo")
    parser.add_argument("--sinr-db", dest="sinr_db", type=float, default=10.0)
    parser.add_argument("--compression-ratio", type=float, default=0.5)
    parser.add_argument("--quality-weight", type=float, default=1.0)
    parser.add_argument("--cost-weight", type=float, default=0.1)
    parser.add_argument("--checkpoint", type=str, default="semantic_jscc/checkpoints/deepjscc_best.pth")
    args = parser.parse_args()

    module = DeepJSCCScenarioModule(checkpoint_path=args.checkpoint)
    image = torch.rand(3, 32, 32)
    result = module.transmit(
        image=image,
        sinr_db=args.sinr_db,
        compression_ratio=args.compression_ratio,
        task_weights={"quality": args.quality_weight, "cost": args.cost_weight},
    )

    print(f"checkpoint_loaded={module.checkpoint_loaded}")
    print(f"sinr_db={result.sinr_db:.2f}")
    print(f"requested_compression_ratio={result.requested_compression_ratio:.3f}")
    print(f"effective_compression_ratio={result.compression_ratio:.3f}")
    print(f"mse={result.mse:.6f}")
    print(f"psnr={result.psnr:.3f}")
    print(f"semantic_quality={result.semantic_quality:.3f}")
    print(f"semantic_utility={result.semantic_utility:.3f}")
    print(f"transmission_cost={result.transmission_cost:.3f}")
    print(f"transmitted_symbols={result.transmitted_symbols}")
    print(f"reconstructed_shape={tuple(result.reconstructed.shape)}")


if __name__ == "__main__":
    main()
