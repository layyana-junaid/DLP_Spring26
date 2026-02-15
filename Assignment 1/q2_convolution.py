#Layyana Junaid 23k-0056
import numpy as np
from utils import load_grayscale, save_grayscale, conv2d

def main():
    dog = load_grayscale("dog.jpg")

    kernel = np.array([[ 1, 0,-1],
                       [ 2, 0,-2],
                       [ 1, 0,-1]], dtype=np.float32)

    conv = conv2d(dog, kernel)

    # Rescale for viewing (edges can be negative)
    conv_view = (conv - conv.min()) / (conv.max() - conv.min() + 1e-8) * 255.0
    save_grayscale(conv_view, "dog_convolved_edges.png")
    print("Saved: dog_convolved_edges.png")

if __name__ == "__main__":
    main()
