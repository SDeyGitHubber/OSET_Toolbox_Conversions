import numpy as np

def ecg_gen_from_phase(params, phase):
    """
    Synthetic ECG generator using a Gaussian Mixture Model.

    Args:
        params (dict or list): ECG model parameters (alpha, b, theta) either as a dict or list.
        phase (np.ndarray): Cardiac phase signal.

    Returns:
        np.ndarray: Synthetic ECG time-series.
    """
    if isinstance(params, dict):
        alpha = np.array(params['alpha'])
        b = np.array(params['b'])
        theta = np.array(params['theta'])
    else:
        # Vector case: Extract parameters
        L = len(params) // 3
        alpha = np.array(params[:L])
        b = np.array(params[L:2 * L])
        theta = np.array(params[2 * L:])

    # Initialize the ECG signal
    x = np.zeros_like(phase)

    # Generate the ECG signal using Gaussian mixture model
    for j in range(len(alpha)):
        # Phase difference with wrapping
        dtheta = (phase - theta[j] + np.pi) % (2 * np.pi) - np.pi

        # Gaussian kernel contribution
        x += alpha[j] * np.exp(-dtheta**2 / (2 * b[j]**2))

    return x