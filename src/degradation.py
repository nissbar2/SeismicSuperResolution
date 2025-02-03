import numpy as np
import cv2
from scipy.ndimage import convolve
from matplotlib import pyplot as plt


def fourier_resize(img, scale=2):
    """
    Resize an image using the Fourier Transform.

    Parameters:
        img (np.ndarray): Input 2D image.
        scale (float): Scaling factor (e.g., 2 means downscaling by 2, 0.5 means upscaling by 2).

    Returns:
        np.ndarray: Resized image.
    """
    # Compute the 2D Fourier Transform
    img_fft = np.fft.fft2(img)
    img_fft_shifted = np.fft.fftshift(img_fft)

    # Get original dimensions
    h, w = img.shape

    # Compute new size
    new_h, new_w = int(h / scale), int(w / scale)

    # Crop the center for downscaling
    start_h, start_w = (h - new_h) // 2, (w - new_w) // 2
    img_fft_shifted_cropped = img_fft_shifted[start_h:start_h + new_h, start_w:start_w + new_w]

    # Shift back and perform inverse Fourier Transform
    img_resized = np.fft.ifft2(np.fft.ifftshift(img_fft_shifted_cropped)).real

    return img_resized

def gaussian_resize_2d(img, scale=2):
    img_resized = cv2.pyrDown(img)
    return img_resized

def gaussian_resize_3d(img, prev, post, scale=2):
    """
    Applies a 3D 3x3x3 Gaussian filter on the image using prev and post frames.

    Parameters:
        img (np.ndarray): Current image (2D).
        prev (np.ndarray): Previous image (2D).
        post (np.ndarray): Next image (2D).
        scale (int): Not used in filtering, but can be used for further resizing.

    Returns:
        np.ndarray: Filtered 2D image.
    """
    # Define a 3D Gaussian kernel (normalized)
    gaussian_kernel = np.array([
        [[1, 2, 1], [2, 4, 2], [1, 2, 1]],
        [[2, 4, 2], [4, 8, 4], [2, 4, 2]],
        [[1, 2, 1], [2, 4, 2], [1, 2, 1]]
    ], dtype=np.float32)
    gaussian_kernel /= gaussian_kernel.sum()  # Normalize kernel

    # Stack the 3 slices into a single 3D volume
    volume = np.stack([prev, img, post], axis=0)

    # Apply 3D convolution
    smoothed_volume = convolve(volume, gaussian_kernel, mode='reflect')

    # Extract the center slice (filtered `img`)
    filtered_img = smoothed_volume[1]

    return fourier_resize(filtered_img, scale)


##################################
def show_both(matrix1,matrix2, title="def"):
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)  # 1 row, 2 columns, first plot
    plt.imshow(matrix1, cmap='gray')
    plt.title("Matrix 1")
    plt.axis("off")  # Hide axes

    plt.subplot(1, 2, 2)  # 1 row, 2 columns, second plot
    plt.imshow(matrix2, cmap='gray')
    plt.title("Matrix 2")
    plt.axis("off")

    plt.show()
    # plt.savefig(f"{title}.png", dpi=300)

def show_one(matrix1, title="def"):
    plt.imshow(matrix1, cmap='gray')
    plt.title("Matrix 1")
    plt.axis("off")  # Hide axes
    plt.axis("off")
    plt.show()
    # plt.savefig(f"{title}.png", dpi=300)


if __name__ == '__main__':
    img_prev = np.load('../my_data/Isramco_N_C_1990_Z36N_AGC.npy')
    img_cur = np.load('../my_data/Isramco_N_C_1991_Z36N_AGC.npy')
    img_post = np.load('../my_data/Isramco_N_C_1992_Z36N_AGC.npy')
    img_resized = gaussian_resize_3d(img_prev,img_cur,img_post, scale=2)
    # print(img_resized.shape)
    show_one(img_resized, title="curandnone")

