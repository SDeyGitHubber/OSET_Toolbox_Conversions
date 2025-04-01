import numpy as np

def skew(data):
    """
    Calculate the skewness, mean, and standard deviation for each channel in the matrix.

    Args:
        data (np.ndarray): 2D matrix of shape (N channels x T samples).

    Returns:
        skw (np.ndarray): Skewness vector of shape (N,).
        m (np.ndarray): Mean vector of shape (N,).
        sd (np.ndarray): Standard deviation vector of shape (N,).
    """
    # ✅ Mean along the second dimension (columns)
    m = np.mean(data, axis=1)
    
    # ✅ Standard deviation along the second dimension
    sd = np.std(data, axis=1, ddof=0)  # MATLAB uses population std by default

    # ✅ Mean of cubed values
    m3 = np.mean(np.power(data, 3), axis=1)
    
    # ✅ Skewness calculation (handle zero std cases)
    with np.errstate(divide='ignore', invalid='ignore'):
        skw = np.divide((m3 - 3 * m * sd**2 - m**3), (sd**3))
        skw = np.nan_to_num(skw)  # Replace NaN with 0

    return skw, m, sd