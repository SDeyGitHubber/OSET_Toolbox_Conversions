import numpy as np
from scipy.interpolate import interp1d, interp2d
from scipy.ndimage import zoom

def linear_warp(x, L):
    """
    Linear time warping of vectors/matrices to arbitrary lengths.

    Args:
        x (np.ndarray): Input vector or matrix.
        L (int or tuple): Desired output length (int for vector, tuple for matrix).

    Returns:
        y (np.ndarray): Time-warped output.
    """
    if x.ndim == 1:  # ✅ Vector case
        M = len(x)
        tx = np.linspace(0, 1, M)       # Original time points
        ty = np.linspace(0, 1, L)        # New time points
        interpolator = interp1d(tx, x, kind='linear', fill_value="extrapolate")
        y = interpolator(ty)

    elif x.ndim == 2:  # ✅ Matrix case
        M1, M2 = x.shape
        num_rows, num_columns = L

        # Use ndimage zoom for faster interpolation
        row_scale = num_rows / M1
        col_scale = num_columns / M2

        y = zoom(x, (row_scale, col_scale), order=1)  # Bilinear interpolation

    else:
        raise ValueError("Input should be a vector or matrix")

    return y