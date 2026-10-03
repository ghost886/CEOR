from __future__ import annotations

import torch


def zero_pixel_baseline(pixel_values: torch.Tensor) -> torch.Tensor:
    """Matches IGOS++ style baseline: `image_tensor * 0` (no visual signal)."""
    return pixel_values * 0


def gaussian_blur_pixel_values(
    pixel_values: torch.Tensor,
    kernel_size: int = 15,
    sigma: float = 4.0,
) -> torch.Tensor:
    """Lightweight separable Gaussian blur on CHW tensors (single image batch)."""
    if pixel_values.dim() != 4:
        raise ValueError("Expected pixel_values [B, C, H, W]")
    b, c, h, w = pixel_values.shape
    x = pixel_values.reshape(b * c, 1, h, w)
    radius = kernel_size // 2
    coords = torch.arange(-radius, radius + 1, device=x.device, dtype=x.dtype)
    g1d = torch.exp(-(coords / sigma) ** 2)
    g1d = g1d / g1d.sum()
    kh, kw = g1d.numel(), g1d.numel()
    weight_h = g1d.view(1, 1, kh, 1).expand(c, 1, kh, 1)
    weight_w = g1d.view(1, 1, 1, kw).expand(c, 1, 1, kw)
    pad_h, pad_w = kh // 2, kw // 2
    y = torch.nn.functional.conv2d(x, weight_h, padding=(pad_h, 0), groups=c)
    y = torch.nn.functional.conv2d(y, weight_w, padding=(0, pad_w), groups=c)
    return y.view(b, c, h, w)
