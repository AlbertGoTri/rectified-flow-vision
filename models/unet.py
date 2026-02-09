"""
Arquitectura UNet simplificada para los flow models.
Basada en la arquitectura usada en diffusion models pero adaptada para flow matching.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Optional


class SinusoidalPositionEmbeddings(nn.Module):
    """Embeddings posicionales sinusoidales para el tiempo."""
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
    
    def forward(self, time: torch.Tensor) -> torch.Tensor:
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class ResidualBlock(nn.Module):
    """Bloque residual con normalización de grupo y condicionamiento temporal."""
    
    def __init__(self, in_channels: int, out_channels: int, time_emb_dim: int, 
                 dropout: float = 0.1):
        super().__init__()
        
        self.norm1 = nn.GroupNorm(8, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels)
        )
        
        self.dropout = nn.Dropout(dropout)
        
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.shortcut = nn.Identity()
    
    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        
        # Añadir embedding temporal
        time_emb = self.time_mlp(time_emb)
        h = h + time_emb[:, :, None, None]
        
        h = self.conv2(self.dropout(F.silu(self.norm2(h))))
        
        return h + self.shortcut(x)


class AttentionBlock(nn.Module):
    """Bloque de self-attention."""
    
    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        
        self.norm = nn.GroupNorm(8, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        
        h = self.norm(x)
        qkv = self.qkv(h)
        q, k, v = qkv.chunk(3, dim=1)
        
        # Reshape para multi-head attention
        q = q.view(B, self.num_heads, C // self.num_heads, H * W)
        k = k.view(B, self.num_heads, C // self.num_heads, H * W)
        v = v.view(B, self.num_heads, C // self.num_heads, H * W)
        
        # Attention
        scale = (C // self.num_heads) ** -0.5
        attn = torch.einsum('bhcn,bhcm->bhnm', q, k) * scale
        attn = F.softmax(attn, dim=-1)
        
        h = torch.einsum('bhnm,bhcm->bhcn', attn, v)
        h = h.view(B, C, H, W)
        h = self.proj(h)
        
        return x + h


class Downsample(nn.Module):
    """Downsampling con convolución strided."""
    
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    """Upsampling con interpolación y convolución."""
    
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        return self.conv(x)


class UNet(nn.Module):
    """
    UNet simplificado para predicción del campo de velocidad en flow matching.
    
    Arquitectura simple y robusta:
    - Encoder: 3 niveles de resolución decreciente
    - Middle: bloques de procesamiento
    - Decoder: 3 niveles con skip connections
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        model_channels: int = 64,
        out_channels: int = 3,
        channel_mult: List[int] = [1, 2, 4],
        num_res_blocks: int = 2,
        attention_resolutions: List[int] = [16, 8],
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_levels = len(channel_mult)
        self.num_res_blocks = num_res_blocks
        
        time_emb_dim = model_channels * 4
        
        # Embedding temporal
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(model_channels),
            nn.Linear(model_channels, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )
        
        # Input
        self.input_conv = nn.Conv2d(in_channels, model_channels, 3, padding=1)
        
        # Calcular canales por nivel
        self.channels = [model_channels * m for m in channel_mult]
        
        # ===== ENCODER =====
        self.enc_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        
        in_ch = model_channels
        for level in range(self.num_levels):
            out_ch = self.channels[level]
            
            # Bloques para este nivel
            for i in range(num_res_blocks):
                self.enc_blocks.append(ResidualBlock(in_ch, out_ch, time_emb_dim, dropout))
                in_ch = out_ch
            
            # Downsample (excepto último nivel)
            if level < self.num_levels - 1:
                self.downsamples.append(nn.Conv2d(out_ch, out_ch, 3, stride=2, padding=1))
            else:
                self.downsamples.append(None)
        
        # ===== MIDDLE =====
        mid_ch = self.channels[-1]
        self.mid_block1 = ResidualBlock(mid_ch, mid_ch, time_emb_dim, dropout)
        self.mid_attn = AttentionBlock(mid_ch)
        self.mid_block2 = ResidualBlock(mid_ch, mid_ch, time_emb_dim, dropout)
        
        # ===== DECODER =====
        self.dec_blocks = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        
        in_ch = mid_ch
        for level in range(self.num_levels - 1, -1, -1):
            out_ch = self.channels[level]
            skip_ch = self.channels[level]  # Skip tiene los mismos canales
            
            # Primer bloque recibe skip connection
            self.dec_blocks.append(ResidualBlock(in_ch + skip_ch, out_ch, time_emb_dim, dropout))
            
            # Bloques adicionales
            for i in range(num_res_blocks - 1):
                self.dec_blocks.append(ResidualBlock(out_ch, out_ch, time_emb_dim, dropout))
            
            in_ch = out_ch
            
            # Upsample (excepto último nivel)
            if level > 0:
                self.upsamples.append(nn.Sequential(
                    nn.Upsample(scale_factor=2, mode='nearest'),
                    nn.Conv2d(out_ch, out_ch, 3, padding=1)
                ))
            else:
                self.upsamples.append(None)
        
        # Output
        self.output_conv = nn.Sequential(
            nn.GroupNorm(8, self.channels[0]),
            nn.SiLU(),
            nn.Conv2d(self.channels[0], out_channels, 3, padding=1)
        )
    
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # Time embedding
        t_emb = self.time_mlp(t)
        
        # Input
        h = self.input_conv(x)
        
        # Encoder - guardar para skip connections
        skips = []
        block_idx = 0
        
        for level in range(self.num_levels):
            for _ in range(self.num_res_blocks):
                h = self.enc_blocks[block_idx](h, t_emb)
                block_idx += 1
            
            skips.append(h)  # Guardar antes del downsample
            
            if self.downsamples[level] is not None:
                h = self.downsamples[level](h)
        
        # Middle
        h = self.mid_block1(h, t_emb)
        h = self.mid_attn(h)
        h = self.mid_block2(h, t_emb)
        
        # Decoder - usar skip connections en orden inverso
        block_idx = 0
        
        for level_idx, level in enumerate(range(self.num_levels - 1, -1, -1)):
            skip = skips.pop()
            
            # Concatenar skip connection
            h = torch.cat([h, skip], dim=1)
            h = self.dec_blocks[block_idx](h, t_emb)
            block_idx += 1
            
            # Bloques adicionales
            for _ in range(self.num_res_blocks - 1):
                h = self.dec_blocks[block_idx](h, t_emb)
                block_idx += 1
            
            # Upsample
            if self.upsamples[level_idx] is not None:
                h = self.upsamples[level_idx](h)
        
        return self.output_conv(h)


def count_parameters(model: nn.Module) -> int:
    """Cuenta el número de parámetros entrenables."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test del modelo
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model = UNet(
        in_channels=3,
        model_channels=64,
        out_channels=3,
        channel_mult=[1, 2, 4],
        num_res_blocks=2,
        attention_resolutions=[16, 8]
    ).to(device)
    
    print(f"Número de parámetros: {count_parameters(model):,}")
    
    # Test forward pass
    x = torch.randn(2, 3, 64, 64).to(device)
    t = torch.rand(2).to(device)
    
    with torch.no_grad():
        out = model(x, t)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
