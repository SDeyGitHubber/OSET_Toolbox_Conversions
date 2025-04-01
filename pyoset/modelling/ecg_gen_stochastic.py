import numpy as np

def ecg_gen_stochastic(N, fs, f, f_deviations, alpha, delta_alpha, b, delta_b, theta, delta_theta, theta0):
    """
    Generate synthetic ECG with beat-wise stochastic deviations.

    Args:
        N (int): Signal length (number of samples).
        fs (float): Sampling frequency (Hz).
        f (float): Average heart rate (Hz).
        f_deviations (float): Beat-wise heart rate deviation (percentage).
        alpha (np.ndarray): Amplitudes of Gaussian kernels.
        delta_alpha (float): Percentage amplitude deviation.
        b (np.ndarray): Widths of the Gaussian kernels.
        delta_b (float): Percentage width deviation.
        theta (np.ndarray): Phases of the Gaussian kernels.
        delta_theta (float): Percentage phase deviation.
        theta0 (float): Initial phase of the ECG.

    Returns:
        np.ndarray: Synthetic ECG signal.
        np.ndarray: Shifted ECG phase.
    """
    # ✅ Initialize parameters
    w = 2 * np.pi * f   # Angular frequency
    dt = 1 / fs         # Time step

    phi = np.zeros(N)    # Phase vector
    ecg = np.zeros(N)    # ECG signal

    phi[0] = theta0
    d_alpha = np.copy(alpha)
    d_theta = np.copy(theta)
    d_b = np.copy(b)
    n_gmm = len(alpha)

    for i in range(N - 1):
        dtetai = (phi[i] - d_theta + np.pi) % (2 * np.pi) - np.pi

        if i == 0:
            ecg[i] = np.sum(d_alpha * np.exp(-dtetai**2 / (2 * d_b**2)))

        ecg[i + 1] = ecg[i] - dt * np.sum(w * d_alpha / (d_b**2) * dtetai * np.exp(-dtetai**2 / (2 * d_b**2)))

        # ✅ Phase update
        phi[i + 1] = phi[i] + w * dt

        # ✅ Beat transition with stochastic deviations
        if phi[i + 1] > np.pi:
            phi[i + 1] -= 2 * np.pi
            d_alpha = alpha * (1 + (np.random.rand(n_gmm) - 0.5) * delta_alpha)
            d_theta = theta * (1 + (np.random.rand(n_gmm) - 0.5) * delta_theta)
            d_b = b * np.maximum(0, (1 + (np.random.rand(n_gmm) - 0.5) * delta_b))
            w = 2 * np.pi * f * np.maximum(0, (1 + (np.random.rand() - 0.5) * f_deviations))

    return ecg, phi