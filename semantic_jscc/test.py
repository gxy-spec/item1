import os
import sys
import argparse
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

if __package__ is None and __name__ == '__main__':
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from semantic_jscc.models import DeepJSCCModel


def get_dataloaders(batch_size=128, num_workers=2):
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return test_loader


def save_image_grid(x, title, path):
    grid = torchvision.utils.make_grid(x.cpu(), nrow=8, padding=2)
    np_grid = grid.permute(1, 2, 0).numpy()
    plt.figure(figsize=(10, 5))
    plt.imshow(np_grid)
    plt.title(title)
    plt.axis('off')
    plt.savefig(path, bbox_inches='tight')
    plt.close()


def load_checkpoint(model, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state'])
    return checkpoint.get('snr_db', 10.0)


def visualize(model, test_loader, device, snr_db, output_dir='semantic_jscc/results'):
    model.eval()
    model.channel.snr_db = snr_db
    images, _ = next(iter(test_loader))
    images = images.to(device)[:16]
    with torch.no_grad():
        recon = model(images)

    os.makedirs(output_dir, exist_ok=True)
    save_image_grid(images, 'Original CIFAR-10 Images', os.path.join(output_dir, 'original.png'))
    save_image_grid(recon, f'Reconstructed Images after AWGN {snr_db}dB', os.path.join(output_dir, 'reconstructed.png'))
    print('Saved original and reconstructed images to', output_dir)


def main():
    parser = argparse.ArgumentParser(description='DeepJSCC CIFAR-10 Test')
    parser.add_argument('--checkpoint', type=str, default='semantic_jscc/checkpoints/deepjscc_best.pth')
    parser.add_argument('--latent-dim', type=int, default=256)
    parser.add_argument('--snr-db', type=float, default=10.0)
    parser.add_argument('--output-dir', type=str, default='semantic_jscc/results')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    test_loader = get_dataloaders(batch_size=128)
    model = DeepJSCCModel(latent_dim=args.latent_dim, snr_db=args.snr_db).to(device)
    load_checkpoint(model, args.checkpoint, device)

    print('Device:', device)
    print('Loaded checkpoint:', args.checkpoint)
    print('Testing with SNR =', args.snr_db, 'dB')

    visualize(model, test_loader, device, args.snr_db, args.output_dir)

    for snr in [0.0, 5.0, 10.0, 15.0]:
        model.channel.snr_db = snr
        images, _ = next(iter(test_loader))
        images = images.to(device)[:8]
        with torch.no_grad():
            recon = model(images)
        fig, axs = plt.subplots(2, 8, figsize=(16, 4))
        for idx in range(8):
            axs[0, idx].imshow(images[idx].cpu().permute(1, 2, 0).numpy())
            axs[0, idx].axis('off')
            axs[1, idx].imshow(recon[idx].cpu().permute(1, 2, 0).numpy())
            axs[1, idx].axis('off')
        fig.suptitle(f'DeepJSCC Reconstruction @ AWGN {snr} dB')
        fig.savefig(os.path.join(args.output_dir, f'recon_snr_{int(snr)}dB.png'), bbox_inches='tight')
        plt.close(fig)
        print('Saved SNR comparison image:', os.path.join(args.output_dir, f'recon_snr_{int(snr)}dB.png'))


if __name__ == '__main__':
    main()
