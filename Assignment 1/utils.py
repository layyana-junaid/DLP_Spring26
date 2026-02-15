#Layyana Junaid 23k-0056
import numpy as np
from PIL import Image

def load_grayscale(path: str) -> np.ndarray:
    """Load image as grayscale float32 array in range [0,255]."""
    img = Image.open(path).convert("L")
    return np.array(img, dtype=np.float32)

def save_grayscale(arr: np.ndarray, path: str):
    """Save grayscale image array (float/any) safely."""
    out = np.clip(arr, 0, 255).astype(np.uint8)
    Image.fromarray(out, mode="L").save(path)

def pad_zeros(img: np.ndarray, pad_h: int, pad_w: int) -> np.ndarray:
    """Zero-pad image on all sides."""
    return np.pad(img, ((pad_h, pad_h), (pad_w, pad_w)), mode="constant", constant_values=0)

def conv2d(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    kh, kw = kernel.shape
    pad_h, pad_w = kh // 2, kw // 2

    k_flip = np.flipud(np.fliplr(kernel)).astype(np.float32)

    padded = pad_zeros(img, pad_h, pad_w)
    H, W = img.shape
    out = np.zeros((H, W), dtype=np.float32)

    for i in range(H):
        for j in range(W):
            region = padded[i:i+kh, j:j+kw]
            out[i, j] = np.sum(region * k_flip)

    return out

def gaussian_kernel(size: int = 7, sigma: float = 1.0) -> np.ndarray:
    if size % 2 != 1:
        raise ValueError("Kernel size must be odd.")
    ax = np.arange(-(size // 2), size // 2 + 1, dtype=np.float32)
    xx, yy = np.meshgrid(ax, ax)
    k = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    k /= np.sum(k)
    return k.astype(np.float32)
