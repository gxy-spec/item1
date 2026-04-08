import torch
import torch.nn as nn
import torch.nn.functional as F


class DeepJSCCEncoder(nn.Module):
    """DeepJSCC编码器：将输入图像映射为连续语义码字。"""

    def __init__(self, latent_dim=256):
        super().__init__()
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
        noise_var = power / snr_linear
        noise = torch.randn_like(x) * torch.sqrt(noise_var)
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
