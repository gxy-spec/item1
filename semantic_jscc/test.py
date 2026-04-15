from __future__ import annotations
import argparse
import datetime
import os
import sys

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

if __package__ is None and __name__ == "__main__":
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def parse_ratio_list(text: str) -> list[float]:
    values = []
    for chunk in text.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        values.append(float(chunk))
    if not values:
        raise ValueError("keep-ratios must contain at least one value")
    return values


def get_dataloaders(batch_size=128, num_workers=2):
    transform = transforms.Compose([transforms.ToTensor()])
    test_set = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return test_loader


def save_image_grid(x, title, path):
    grid = torchvision.utils.make_grid(x.cpu(), nrow=8, padding=2)
    np_grid = grid.permute(1, 2, 0).numpy()
    plt.figure(figsize=(10, 5))
    plt.imshow(np_grid)
    plt.title(title)
    plt.axis("off")
    plt.savefig(path, bbox_inches="tight")
    plt.close()


def load_checkpoint(model, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    return {
        "sinr_db": checkpoint.get("sinr_db", checkpoint.get("snr_db", 10.0)),
        "keep_ratios": checkpoint.get("keep_ratios", [1.0]),
        "latent_dim": checkpoint.get("latent_dim", None),
    }


def compute_psnr(images, recon):
    mse = torch.mean((images - recon) ** 2).item()
    if mse <= 1e-12:
        return float("inf")
    return float(10.0 * np.log10(1.0 / mse))


def visualize(model, test_loader, device, sinr_db, keep_ratios, output_dir="semantic_jscc/results"):
    model.eval()
    images, _ = next(iter(test_loader))
    images = images.to(device)[:16]

    os.makedirs(output_dir, exist_ok=True)
    save_image_grid(images, "Original CIFAR-10 Images", os.path.join(output_dir, "original.png"))

    for keep_ratio in keep_ratios:
        with torch.no_grad():
            recon = model(images, sinr_db=sinr_db, compression_ratio=keep_ratio)
        psnr = compute_psnr(images, recon)
        ratio_tag = str(keep_ratio).replace(".", "p")
        save_image_grid(
            recon,
            f"Reconstructed Images @ SINR {sinr_db}dB, keep_ratio={keep_ratio:.2f}, PSNR={psnr:.2f}",
            os.path.join(output_dir, f"reconstructed_keep_{ratio_tag}.png"),
        )
        print(f"keep_ratio={keep_ratio:.2f} psnr={psnr:.3f}")


def main():
    from semantic_jscc.models import DeepJSCCModel

    parser = argparse.ArgumentParser(description="DeepJSCC CIFAR-10 Test")
    parser.add_argument("--checkpoint", type=str, default="semantic_jscc/checkpoints/deepjscc_best.pth")
    parser.add_argument("--latent-dim", type=int, default=256)
    parser.add_argument(
        "--snr-db",
        "--sinr-db",
        dest="sinr_db",
        type=float,
        default=10.0,
        help="SINR value in dB for the channel model",
    )
    parser.add_argument(
        "--keep-ratios",
        type=str,
        default=None,
        help="Optional keep ratios for evaluation. Defaults to checkpoint keep ratios.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory. If not specified, creates timestamped directory under semantic_jscc/results/",
    )
    args = parser.parse_args()

    if args.output_dir is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = f"semantic_jscc/results/results_{timestamp}"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_loader = get_dataloaders(batch_size=128)
    model = DeepJSCCModel(latent_dim=args.latent_dim, sinr_db=args.sinr_db).to(device)
    checkpoint_meta = load_checkpoint(model, args.checkpoint, device)

    keep_ratios = parse_ratio_list(args.keep_ratios) if args.keep_ratios is not None else checkpoint_meta["keep_ratios"]

    print("Device:", device)
    print("Loaded checkpoint:", args.checkpoint)
    print("Testing with SINR =", args.sinr_db, "dB")
    print("Keep ratios:", keep_ratios)

    visualize(model, test_loader, device, args.sinr_db, keep_ratios, args.output_dir)


if __name__ == "__main__":
    main()
