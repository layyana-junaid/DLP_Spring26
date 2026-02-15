#Layyana Junaid 23k-0056
import numpy as np
from utils import load_grayscale, save_grayscale

def template_match_convolution(image: np.ndarray, template: np.ndarray) -> np.ndarray:
    H, W = image.shape
    h, w = template.shape

    t0 = template - np.mean(template)
    t0_flip = np.flipud(np.fliplr(t0))

    outH, outW = H - h + 1, W - w + 1
    resp = np.zeros((outH, outW), dtype=np.float32)

    for i in range(outH):
        for j in range(outW):
            patch = image[i:i+h, j:j+w]
            p0 = patch - np.mean(patch)
            resp[i, j] = np.sum(p0 * t0_flip)
    return resp

def template_match_correlation(image: np.ndarray, template: np.ndarray) -> np.ndarray:
    H, W = image.shape
    h, w = template.shape

    t0 = template - np.mean(template)

    outH, outW = H - h + 1, W - w + 1
    resp = np.zeros((outH, outW), dtype=np.float32)

    for i in range(outH):
        for j in range(outW):
            patch = image[i:i+h, j:j+w]
            p0 = patch - np.mean(patch)
            resp[i, j] = np.sum(p0 * t0)
    return resp

def argmax_2d(a: np.ndarray):
    idx = np.argmax(a)
    return np.unravel_index(idx, a.shape)  # (row, col)

def draw_box_on_image_gray(image: np.ndarray, top: int, left: int, h: int, w: int, thickness: int = 3, value: int = 255) -> np.ndarray:
    out = image.copy()
    bottom = top + h
    right = left + w

    # top and bottom
    out[top:top+thickness, left:right] = value
    out[bottom-thickness:bottom, left:right] = value

    # left and right
    out[top:bottom, left:left+thickness] = value
    out[top:bottom, right-thickness:right] = value

    return out

def main():
    shelf = load_grayscale("shelf.jpg")
    template = load_grayscale("template.jpg")
    h, w = template.shape

    resp_conv = template_match_convolution(shelf, template)
    resp_corr = template_match_correlation(shelf, template)

    r_conv, c_conv = argmax_2d(resp_conv)
    r_corr, c_corr = argmax_2d(resp_corr)

    shelf_conv_box = draw_box_on_image_gray(shelf, r_conv, c_conv, h, w)
    shelf_corr_box = draw_box_on_image_gray(shelf, r_corr, c_corr, h, w)

    save_grayscale(shelf_conv_box, "shelf_match_convolution.png")
    save_grayscale(shelf_corr_box, "shelf_match_correlation.png")

    print("Best match (convolution) top-left:", (r_conv, c_conv))
    print("Best match (correlation) top-left:", (r_corr, c_corr))
    print("Saved:")
    print("- shelf_match_convolution.png")
    print("- shelf_match_correlation.png")

if __name__ == "__main__":
    main()
