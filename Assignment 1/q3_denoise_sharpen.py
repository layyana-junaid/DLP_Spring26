#Layyana Junaid 23k-0056
import numpy as np
from utils import load_grayscale, save_grayscale, conv2d, gaussian_kernel

def add_gaussian_noise(img: np.ndarray, mean: float = 0.0, std: float = 15.0, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    noise = rng.normal(loc=mean, scale=std, size=img.shape).astype(np.float32)
    return np.clip(img + noise, 0, 255)

def main():
    dog = load_grayscale("dog.jpg")

    dog_noisy = add_gaussian_noise(dog, mean=0.0, std=15.0, seed=42)
    save_grayscale(dog_noisy, "dog_noisy.png")

    gk = gaussian_kernel(size=7, sigma=1.0)
    dog_denoised = conv2d(dog_noisy, gk)
    save_grayscale(dog_denoised, "dog_denoised.png")

    sharpening_kernel = np.array([
        [1, 4,   6,   4, 1],
        [4, 16,  24, 16, 4],
        [6, 24, -476, 24, 6],
        [4, 16,  24, 16, 4],
        [1, 4,   6,   4, 1]
    ], dtype=np.float32) * (-1.0 / 256.0)

    dog_sharp = conv2d(dog_denoised, sharpening_kernel)
    save_grayscale(dog_sharp, "dog_sharpened.png")

    print("Saved:")
    print("- dog_noisy.png")
    print("- dog_denoised.png")
    print("- dog_sharpened.png")

if __name__ == "__main__":
    main()
