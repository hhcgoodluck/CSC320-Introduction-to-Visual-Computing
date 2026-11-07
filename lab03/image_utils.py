"""Small shared image helpers used across CSC320 notebooks."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


IMAGE_DIR = Path("images")


def load_image(name, image_dir=IMAGE_DIR):
    """Load an RGB image from the course image directory."""
    path = Path(image_dir) / name
    if not path.exists():
        raise FileNotFoundError(f"Add a real photo at {path}")
    return np.asarray(Image.open(path).convert("RGB"))


def show(img, title=None, ax=None):
    """Display an image on a matplotlib axis."""
    ax = ax or plt.gca()
    ax.imshow(img, cmap="gray" if np.asarray(img).ndim == 2 else None)
    ax.set_axis_off()
    if title:
        ax.set_title(title)
    return ax


def as_float_image(image):
    """Convert an image array to float values in [0, 1]."""
    img = np.asarray(image)
    if np.issubdtype(img.dtype, np.integer):
        img = img.astype(float) / np.iinfo(img.dtype).max
    else:
        img = img.astype(float)
    return np.clip(img, 0, 1)


def rgb_to_gray(image):
    """Convert RGB to grayscale using luminance weights."""
    img = as_float_image(image)
    if img.ndim == 2:
        return img
    return 0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]


def to_uint8(image):
    """Convert a float image in [0, 1] to uint8."""
    return np.clip(np.asarray(image) * 255, 0, 255).astype(np.uint8)


def resize_image(image, scale=None, output_shape=None, method="bilinear"):
    """Resize an image with Pillow interpolation."""
    img = as_float_image(image)
    h, w = img.shape[:2]
    if output_shape is None:
        if scale is None:
            return img.copy()
        output_shape = (max(1, int(round(h * scale))), max(1, int(round(w * scale))))
    resampling = getattr(Image, "Resampling", Image)
    methods = {
        "nearest": resampling.NEAREST,
        "bilinear": resampling.BILINEAR,
        "bicubic": resampling.BICUBIC,
        "lanczos": resampling.LANCZOS,
    }
    pil = Image.fromarray(to_uint8(img))
    resized = pil.resize((output_shape[1], output_shape[0]), methods[method])
    return as_float_image(np.asarray(resized))


def convolve_image(image, kernel, padding="reflect"):
    """Convolve an image with a 2D kernel."""
    img = as_float_image(image)
    kernel = np.asarray(kernel, dtype=float)
    kh, kw = kernel.shape
    py, px = kh // 2, kw // 2
    pad_width = ((py, py), (px, px)) if img.ndim == 2 else ((py, py), (px, px), (0, 0))
    mode = "constant" if padding == "constant" else padding
    padded = np.pad(img, pad_width, mode=mode)
    out = np.zeros_like(img, dtype=float)
    flipped = kernel[::-1, ::-1]
    for y in range(kh):
        for x in range(kw):
            out += flipped[y, x] * padded[y : y + img.shape[0], x : x + img.shape[1]]
    return out


def show_image(img, title, ax, cmap=None, vmin=None, vmax=None):
    """Display an image on an axis and hide axis ticks."""
    ax.imshow(img, cmap=cmap or ("gray" if np.asarray(img).ndim == 2 else None), vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.set_axis_off()
    return ax
