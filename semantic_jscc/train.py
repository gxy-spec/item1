import os
import sys
import argparse
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import torch
import torch.nn as nn
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
    train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, test_loader


def train_epoch(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for images, _ in train_loader:
        images = images.to(device)
        optimizer.zero_grad()
        recon = model(images)
        loss = criterion(recon, images)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
    return total_loss / len(train_loader.dataset)


def evaluate(model, test_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for images, _ in test_loader:
            images = images.to(device)
            recon = model(images)
            loss = criterion(recon, images)
            total_loss += loss.item() * images.size(0)
    return total_loss / len(test_loader.dataset)


def main():
    parser = argparse.ArgumentParser(description='DeepJSCC CIFAR-10 Training')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--latent-dim', type=int, default=256)
    parser.add_argument('--snr-db', type=float, default=10.0)
    parser.add_argument('--output-dir', type=str, default='semantic_jscc/checkpoints')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader, test_loader = get_dataloaders(batch_size=args.batch_size)
    model = DeepJSCCModel(latent_dim=args.latent_dim, snr_db=args.snr_db).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    os.makedirs(args.output_dir, exist_ok=True)
    print('Device:', device)
    print('Training DeepJSCC model with SNR =', args.snr_db, 'dB')

    best_loss = float('inf')
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = evaluate(model, test_loader, criterion, device)
        print(f'Epoch {epoch}/{args.epochs} | Train MSE: {train_loss:.6f} | Val MSE: {val_loss:.6f}')
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save({'model_state': model.state_dict(), 'snr_db': args.snr_db}, os.path.join(args.output_dir, 'deepjscc_best.pth'))
            print('Saved best checkpoint:', os.path.join(args.output_dir, 'deepjscc_best.pth'))


if __name__ == '__main__':
    main()
