import numpy as np

def polynomial_fit(x, fs, N, method='LS'):
    """
    Polynomial fit over a signal segment using least squares or pseudoinversion.

    Args:
        x (np.ndarray): Noisy input signal.
        fs (float): Sampling frequency.
        N (int): Polynomial order (>= 1).
        method (str): Fitting method: 'LS' (least squares) or 'PINV' (pseudoinverse).

    Returns:
        y (np.ndarray): Fitted mode.
        p (np.ndarray): Polynomial coefficients.
    """
    # ✅ Input validation
    if N < 0:
        raise ValueError("Polynomial order must be >= 0")
    if method not in ['LS', 'PINV']:
        raise ValueError("Unknown method. Use 'LS' or 'PINV'.")

    # ✅ Reshape the signal to row vector
    x = np.atleast_1d(x).flatten()
    L = len(x)

    # ✅ Time vector
    t = np.arange(L) / fs

    # ✅ Construct Vandermonde matrix
    T = np.vstack([t**i for i in range(N + 1)]).T

    # ✅ Compute TX = T.T @ x
    TX = T.T @ x

    # ✅ Polynomial fitting
    if method == 'LS':
        # Least squares solution
        p = np.linalg.lstsq(T, x, rcond=None)[0]
    elif method == 'PINV':
        # Pseudoinverse solution
        p = np.linalg.pinv(T.T @ T) @ TX

    # ✅ Fitted polynomial signal
    y = T @ p

    return y, p