import numpy as np

def ecg_gen_gmm(phi, theta0, alpha, b, theta):
    """
    Generate synthetic ECG using a Gaussian Mixture Model.

    Args:
        phi (np.ndarray): ECG phase signal derived from real ECG.
        theta0 (float): Desired phase shift.
        alpha (np.ndarray): Amplitudes of Gaussian kernels.
        b (np.ndarray): Widths (standard deviations) of the Gaussian kernels.
        theta (np.ndarray): Centers of the Gaussian kernels.

    Returns:
        np.ndarray: Synthetic ECG signal.
        np.ndarray: Shifted ECG phase.
    """
    N = len(phi)

    # ✅ Shift and wrap ECG phase
    phi = (phi + theta0 + np.pi) % (2 * np.pi) - np.pi

    # ✅ Phase differences for Gaussian kernels
    dtetai = (np.tile(phi, (len(theta), 1)).T - np.tile(theta, (N, 1)) + np.pi) % (2 * np.pi) - np.pi

    # ✅ Generate synthetic ECG signal using GMM
    ecg = np.sum(np.tile(alpha, (N, 1)) * np.exp(-dtetai**2 / (2 * np.tile(b, (N, 1))**2)), axis=1)

    return ecg, phi