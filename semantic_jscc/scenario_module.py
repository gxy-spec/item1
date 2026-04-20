from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from semantic_jscc.models import DeepJSCCModel


@dataclass
class SemanticTaskWeights:
    quality: float = 1.0
    cost: float = 0.1


@dataclass
class SemanticTransmissionResult:
    reconstructed: torch.Tensor
    mse: float
    psnr: float
    semantic_quality: float
    semantic_utility: float
    transmission_cost: float
    transmitted_symbols: int
    total_symbols: int
    compression_ratio: float
    requested_compression_ratio: float
    sinr_db: float
    mask_mode: str


class DeepJSCCScenarioModule:
    """面向仿真场景的 DeepJSCC 调用模块。"""

    def __init__(
        self,
        checkpoint_path: str | Path = "semantic_jscc/checkpoints/deepjscc_best.pth",
        latent_dim: int = 256,
        default_sinr_db: float = 10.0,
        mask_mode: str = "uniform",
        device: str | torch.device | None = None,
        strict_checkpoint: bool = False,
    ) -> None:
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.checkpoint_path = Path(checkpoint_path)
        self.checkpoint_loaded = False
        checkpoint_mask_mode = mask_mode

        if self.checkpoint_path.exists():
            checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
            checkpoint_mask_mode = checkpoint.get("mask_mode", checkpoint_mask_mode)
            checkpoint_latent_dim = checkpoint.get("latent_dim", latent_dim)
            self.model = DeepJSCCModel(
                latent_dim=checkpoint_latent_dim,
                sinr_db=default_sinr_db,
                mask_mode=checkpoint_mask_mode,
            ).to(self.device)
            state_dict = checkpoint.get("model_state", checkpoint)
            self.model.load_state_dict(state_dict, strict=False)
            self.checkpoint_loaded = True
        elif strict_checkpoint:
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")
        else:
            self.model = DeepJSCCModel(
                latent_dim=latent_dim,
                sinr_db=default_sinr_db,
                mask_mode=checkpoint_mask_mode,
            ).to(self.device)

        self.model.eval()

    @staticmethod
    def _normalize_task_weights(task_weights: Any) -> SemanticTaskWeights:
        if task_weights is None:
            return SemanticTaskWeights()
        if isinstance(task_weights, (int, float)):
            quality = float(task_weights)
            return SemanticTaskWeights(quality=quality, cost=max(0.0, 1.0 - quality))
        if isinstance(task_weights, dict):
            return SemanticTaskWeights(
                quality=float(task_weights.get("quality", 1.0)),
                cost=float(task_weights.get("cost", 0.1)),
            )
        if isinstance(task_weights, SemanticTaskWeights):
            return task_weights
        raise TypeError("task_weights must be None, scalar, dict, or SemanticTaskWeights")

    @staticmethod
    def _prepare_input(image: torch.Tensor | np.ndarray) -> torch.Tensor:
        if isinstance(image, np.ndarray):
            tensor = torch.from_numpy(image).float()
        elif isinstance(image, torch.Tensor):
            tensor = image.detach().float()
        else:
            raise TypeError("image must be a torch.Tensor or numpy.ndarray")

        if tensor.ndim == 3:
            if tensor.shape[0] in (1, 3):
                tensor = tensor.unsqueeze(0)
            else:
                tensor = tensor.permute(2, 0, 1).unsqueeze(0)
        elif tensor.ndim == 4:
            if tensor.shape[-1] in (1, 3) and tensor.shape[1] not in (1, 3):
                tensor = tensor.permute(0, 3, 1, 2)
        else:
            raise ValueError("image tensor must have shape [C,H,W], [H,W,C], [B,C,H,W], or [B,H,W,C]")

        if tensor.max() > 1.0:
            tensor = tensor / 255.0
        return tensor.clamp(0.0, 1.0)

    @staticmethod
    def _compute_mse_and_psnr(reference: torch.Tensor, reconstructed: torch.Tensor) -> tuple[float, float]:
        mse = torch.mean((reference - reconstructed) ** 2).item()
        if mse <= 1e-12:
            return 0.0, float("inf")
        psnr = 10.0 * np.log10(1.0 / mse)
        return float(mse), float(psnr)

    @staticmethod
    def _semantic_quality_score(mse: float, psnr: float) -> float:
        mse_score = 1.0 / (1.0 + 10.0 * mse)
        psnr_score = 1.0 if psnr == float("inf") else min(psnr / 40.0, 1.0)
        return float(0.5 * mse_score + 0.5 * psnr_score)

    def transmit(
        self,
        image: torch.Tensor | np.ndarray,
        sinr_db: float,
        compression_ratio: float = 1.0,
        task_weights: Any = None,
    ) -> SemanticTransmissionResult:
        weights = self._normalize_task_weights(task_weights)
        inputs = self._prepare_input(image).to(self.device)

        with torch.no_grad():
            reconstructed, details = self.model(
                inputs,
                sinr_db=sinr_db,
                compression_ratio=compression_ratio,
                return_details=True,
            )

        mse, psnr = self._compute_mse_and_psnr(inputs, reconstructed)
        semantic_quality = self._semantic_quality_score(mse=mse, psnr=psnr)
        transmission_cost = float(details["compression_ratio"])
        semantic_utility = float(weights.quality * semantic_quality - weights.cost * transmission_cost)

        return SemanticTransmissionResult(
            reconstructed=reconstructed.detach().cpu(),
            mse=mse,
            psnr=psnr,
            semantic_quality=semantic_quality,
            semantic_utility=semantic_utility,
            transmission_cost=transmission_cost,
            transmitted_symbols=int(details["transmitted_symbols"]),
            total_symbols=int(details["total_symbols"]),
            compression_ratio=float(details["compression_ratio"]),
            requested_compression_ratio=float(details["requested_compression_ratio"]),
            sinr_db=float(details["sinr_db"]),
            mask_mode=str(details["mask_mode"]),
        )
