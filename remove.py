import os
import datetime
import matplotlib.pyplot as plt
import numpy as np
from scipy.fftpack import dct, idct
from PIL import Image

def save_image(img, path, name=None, category="debug"):
    if name is None:
        name = datetime.datetime.now().strftime('%Y-%m-%dT%H-%M-%S.%f')

    dest = os.path.join(path, f"{category}{'_' if name is not None and len(name) > 0 else ''}{name}.png")
    plt.imsave(dest, img)

def gradient(A: np.ndarray) -> np.ndarray:
    rows, cols = A.shape
    
    grad_x = np.zeros_like(A)
    grad_x[:, 0: cols - 1] = np.diff(A, axis=1)

    grad_y = np.zeros_like(A)
    grad_y[0:rows - 1, :] = np.diff(A, axis=0)

    B = np.concatenate((grad_x[..., np.newaxis], grad_y[..., np.newaxis]), axis=-1)
    return B

def divergence(A):
    m, n, _ = A.shape
    B = np.zeros(shape=(m, n))

    T = A[:, :, 0]
    T1 = np.zeros(shape=(m, n))
    T1[:, 1:n] = T[:, 0:n-1]
    B = B + T - T1

    T = A[:, :, 1]
    T1 = np.zeros(shape=(m, n))
    T1[1:m, :] = T[0:m-1, :]
    B = B + T - T1
    return B

def laplacian(f, h: float = None):
    dims = 2
    grads = gradient(f)

    if h is not None:
        norm = np.linalg.norm(grads, axis=-1)
        mask = (norm < h)[..., np.newaxis].repeat(dims, axis=-1)
        grads[mask] = 0
    
    laplacian = divergence(grads)
    return laplacian

def dct2(block):
    return dct(dct(block.T, norm='ortho').T, norm='ortho')

def idct2(block):
    return idct(idct(block.T, norm='ortho').T, norm='ortho')

def min_max_scale(ar: np.array, new_min: float = 0, new_max: float = 1):
    ar -= ar.min()
    ar /= np.ptp(ar)  
    ar *= (new_max - new_min)
    ar += new_min
    return ar
def remove_reflection(image: np.ndarray, h: float, lmbd: float = 0, mu: float = 1, epsilon: float = 1e-8, output_dir: str = "output"):
    if not np.all((0 <= image) & (image <= 1)):
        raise ValueError("Input image doesn't have all values between 0 and 1.")
    if len(image.shape) != 3:
        raise ValueError("Input image must have 3 dimensions.")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    channels = image.shape[-1]

    laplacians = np.zeros(shape=image.shape)
    for c in range(channels):
        lapl1 = laplacian(image[..., c], h=h)
        lapl2 = laplacian(lapl1)
        laplacians[:, :, c] = lapl2

        save_image(lapl2, output_dir, name=f"{c}", category="laplacian")

    lapl_out = np.interp(laplacians, (laplacians.min(), laplacians.max()), (0, +1))
    save_image(lapl_out, output_dir, name="all", category="laplacian")

    rhs = laplacians + epsilon * image
    rhs_out = np.interp(rhs, (rhs.min(), rhs.max()), (0, 1))
    save_image(rhs_out, output_dir, name="", category="rhs")

    T = np.zeros(shape=rhs.shape)
    for c in range(channels):
        rhs_slice = rhs[..., c]

        M, N = rhs_slice.shape

        m = np.cos((np.pi * np.arange(M)) / M)
        n = np.cos((np.pi * np.arange(N)) / N)
        kappa = 2 * (np.add.outer(m, n) - 2)

        save_image(kappa, output_dir, name=f"{c}", category="kappa")

        const = mu * (kappa ** 2) - lmbd * kappa + epsilon

        save_image(const, output_dir, name=f"{c}", category="denominator")

        u = dct2(rhs_slice)
        u = u / const
        u = idct2(u)

        T[..., c] = u

        save_image(const, output_dir, name=f"{c}", category=f"T")

    min_max_scale(T)
    save_image(T, output_dir, name="", category="out")

    return T

if __name__ == "__main__":
    img = np.array(Image.open("input_image.png")) / 255.0 
    output_image = remove_reflection(img, h=0.1, output_dir="output")
    Image.fromarray((output_image * 255).astype(np.uint8)).save("output_image.png")
