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


class SINRChannel(nn.Module):
    """SINR信道：在语义码字上加入高斯噪声，支持动态SINR。"""

    def __init__(self, sinr_db=10.0):
        super().__init__()
        self.sinr_db = sinr_db

    def forward(self, x, sinr_db=None, signal_mask=None):
        sinr_db = self.sinr_db if sinr_db is None else sinr_db
        if not self.training and sinr_db == float("inf"):
            return x

        if signal_mask is None:
            power = torch.mean(x.pow(2), dim=[1, 2, 3], keepdim=True)
            effective_mask = None
        else:
            effective_mask = signal_mask.to(dtype=x.dtype)
            active_power = (x.pow(2) * effective_mask).sum(dim=[1, 2, 3], keepdim=True)
            active_count = effective_mask.sum(dim=[1, 2, 3], keepdim=True).clamp_min(1.0)
            power = active_power / active_count

        sinr_linear = 10 ** (sinr_db / 10.0)
        noise_var = power / max(sinr_linear, 1e-10)
        noise = torch.randn_like(x) * torch.sqrt(noise_var)
        if effective_mask is not None:
            noise = noise * effective_mask
        return x + noise


class DeepJSCCModel(nn.Module):
    """完整DeepJSCC流程模型：编码器 -> SINR信道 -> 解码器。"""

    def __init__(self, latent_dim=256, sinr_db=10.0, mask_mode="uniform"):
        super().__init__()
        if latent_dim % 16 != 0:
            raise ValueError("latent_dim must be divisible by 16")
        self.encoder = DeepJSCCEncoder(latent_dim=latent_dim)
        self.channel = SINRChannel(sinr_db=sinr_db)
        self.decoder = DeepJSCCDecoder(latent_dim=latent_dim)
        self.latent_dim = latent_dim
        self.mask_mode = str(mask_mode).lower()
        if self.mask_mode not in {"prefix", "uniform"}:
            raise ValueError("mask_mode must be 'prefix' or 'uniform'")

    def _build_channel_mask(self, z, compression_ratio):
        compression_ratio = float(max(0.0, min(1.0, compression_ratio)))
        total_channels = z.shape[1]
        active_channels = max(1, min(total_channels, int(round(total_channels * compression_ratio))))
        mask = torch.zeros_like(z)
        if self.mask_mode == "prefix" or active_channels == total_channels:
            selected_indices = torch.arange(active_channels, device=z.device)
        else:
            selected_indices = torch.linspace(
                0,
                total_channels - 1,
                steps=active_channels,
                device=z.device,
            ).round().long().unique(sorted=True)
            if selected_indices.numel() < active_channels:
                full_indices = torch.arange(total_channels, device=z.device)
                missing = active_channels - selected_indices.numel()
                remaining = full_indices[~torch.isin(full_indices, selected_indices)]
                selected_indices = torch.cat([selected_indices, remaining[:missing]], dim=0)
                selected_indices = selected_indices.sort().values
        mask[:, selected_indices, :, :] = 1.0
        return mask, active_channels

    def forward(self, x, sinr_db=None, compression_ratio=1.0, return_details=False):
        z = self.encoder(x)
        mask, active_channels = self._build_channel_mask(z, compression_ratio)
        z_masked = z * mask
        z_channel = self.channel(z_masked, sinr_db=sinr_db, signal_mask=mask)
        x_hat = self.decoder(z_channel)

        if not return_details:
            return x_hat

        total_symbols = z.shape[1] * z.shape[2] * z.shape[3]
        transmitted_symbols = active_channels * z.shape[2] * z.shape[3]
        details = {
            "latent_shape": tuple(z.shape),
            "active_channels": int(active_channels),
            "total_channels": int(z.shape[1]),
            "compression_ratio": float(transmitted_symbols / total_symbols),
            "requested_compression_ratio": float(max(0.0, min(1.0, compression_ratio))),
            "total_symbols": int(total_symbols),
            "transmitted_symbols": int(transmitted_symbols),
            "sinr_db": float(self.channel.sinr_db if sinr_db is None else sinr_db),
            "mask_mode": self.mask_mode,
        }
        return x_hat, details
