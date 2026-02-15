#Layyana Junaid 23k-0056

import numpy as np
from utils import load_grayscale, save_grayscale

def add_gaussian_noise(img: np.ndarray, mean: float = 0.0, std: float = 15.0, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    noise = rng.normal(loc=mean, scale=std, size=img.shape).astype(np.float32)
    noisy = img + noise
    return np.clip(noisy, 0, 255)

def main():
    dog = load_grayscale("dog.jpg")
    dog_noisy = add_gaussian_noise(dog, mean=0.0, std=15.0, seed=42)
    save_grayscale(dog_noisy, "dog_noisy.png")
    print("Saved: dog_noisy.png")

if __name__ == "__main__":
    main()
