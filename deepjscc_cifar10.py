import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


class DeepJSCCEncoder(nn.Module):
    """DeepJSCC编码器：将输入图像映射为连续语义码字。"""

    def __init__(self, latent_dim=256):
        super().__init__()
        # 32x32x3 -> 16x16x32 -> 8x8x64 -> 4x4x128 -> 4x4xlatent_channels
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(128, latent_dim // 16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(latent_dim // 16)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.tanh(self.bn4(self.conv4(x)))
        # tanh有助于将信号限制到[-1,1]范围，便于后续AWGN信道建模
        return x


class DeepJSCCDecoder(nn.Module):
    """DeepJSCC解码器：从语义码字恢复图像。"""

    def __init__(self, latent_dim=256):
        super().__init__()
        self.deconv1 = nn.ConvTranspose2d(latent_dim // 16, 128, kernel_size=3, stride=1, padding=1)
        self.deconv2 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.deconv3 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.deconv4 = nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(128)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(32)

    def forward(self, x):
        x = F.relu(self.bn1(self.deconv1(x)))
        x = F.relu(self.bn2(self.deconv2(x)))
        x = F.relu(self.bn3(self.deconv3(x)))
        x = torch.sigmoid(self.deconv4(x))
        # sigmoid输出0-1，直接对应输入图像像素区间
        return x


class AWGNChannel(nn.Module):
    """AWGN信道：在语义码字上加入高斯噪声，支持可调SNR。"""

    def __init__(self, snr_db=10.0):
        super().__init__()
        self.snr_db = snr_db

    def forward(self, x):
        if not self.training and self.snr_db == float('inf'):
            return x
        power = torch.mean(x.pow(2), dim=[1, 2, 3], keepdim=True)
        snr_linear = 10 ** (self.snr_db / 10.0)
        noise_variance = power / snr_linear
        noise = torch.randn_like(x) * torch.sqrt(noise_variance)
        return x + noise


class DeepJSCCModel(nn.Module):
    """完整DeepJSCC流程模型：编码器 -> 信道 -> 解码器。"""

    def __init__(self, latent_dim=256, snr_db=10.0):
        super().__init__()
        self.encoder = DeepJSCCEncoder(latent_dim=latent_dim)
        self.channel = AWGNChannel(snr_db=snr_db)
        self.decoder = DeepJSCCDecoder(latent_dim=latent_dim)

    def forward(self, x):
        z = self.encoder(x)
        z_channel = self.channel(z)
        x_hat = self.decoder(z_channel)
        return x_hat


def get_dataloaders(batch_size=128, num_workers=2):
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, test_loader


def save_image_grid(x, title, path):
    x = x.detach().cpu().numpy()
    grid = np.transpose(torchvision.utils.make_grid(torch.from_numpy(x), nrow=8, padding=2, normalize=False), (1, 2, 0))
    plt.figure(figsize=(10, 5))
    plt.imshow(grid)
    plt.title(title)
    plt.axis('off')
    plt.savefig(path, bbox_inches='tight')
    plt.close()


def train(model, train_loader, optimizer, criterion, device):
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


def plot_reconstruction(model, test_loader, device, snr_db, output_dir='results'):
    model.eval()
    model.channel.snr_db = snr_db
    images, _ = next(iter(test_loader))
    images = images.to(device)[:16]
    with torch.no_grad():
        recon = model(images)
    os.makedirs(output_dir, exist_ok=True)
    save_image_grid(images, 'Original CIFAR-10 Images', os.path.join(output_dir, 'original.png'))
    save_image_grid(recon, f'Reconstructed Images after AWGN {snr_db}dB', os.path.join(output_dir, 'reconstructed.png'))
    print(f'已保存原图与重建图到 {output_dir}/original.png 和 {output_dir}/reconstructed.png')


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    latent_dim = 256
    snr_db = 10.0
    epochs = 10
    batch_size = 128
    learning_rate = 1e-3

    train_loader, test_loader = get_dataloaders(batch_size=batch_size)
    model = DeepJSCCModel(latent_dim=latent_dim, snr_db=snr_db).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    print('设备:', device)
    print('DeepJSCC模型结构: Encoder->AWGNChannel->Decoder')
    print('训练SNR:', snr_db, 'dB')

    for epoch in range(1, epochs + 1):
        train_loss = train(model, train_loader, optimizer, criterion, device)
        test_loss = evaluate(model, test_loader, criterion, device)
        print(f'Epoch {epoch}/{epochs} | Train MSE: {train_loss:.6f} | Test MSE: {test_loss:.6f}')

    plot_reconstruction(model, test_loader, device, snr_db)

    # 演示不同SNR下的恢复效果：0dB、5dB、10dB、15dB
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
        os.makedirs('results', exist_ok=True)
        fig.savefig(f'results/recon_snr_{int(snr)}dB.png', bbox_inches='tight')
        plt.close(fig)
        print(f'已保存SN R={snr}dB对比图: results/recon_snr_{int(snr)}dB.png')


if __name__ == '__main__':
    main()
