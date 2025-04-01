import numpy as np

def outlier_filter(x_raw, method='MEDIAN', half_wlen=5, percentile=95):
    """
    Outlier filter to detect and replace outliers with mean or median of a sliding window.
    
    Args:
        x_raw (np.ndarray): Input signal (num_channels x time).
        method (str): 'MEAN' or 'MEDIAN'.
        half_wlen (int): Half window length for sliding window.
        percentile (float): Percentile threshold for outlier detection.
        
    Returns:
        np.ndarray: Filtered signal.
    """
    if method not in ['MEAN', 'MEDIAN']:
        raise ValueError("Invalid method. Use 'MEAN' or 'MEDIAN'.")

    # First-order difference along time axis
    df = np.diff(x_raw, axis=1)
    diff_threshold = np.percentile(np.abs(df), percentile, axis=1)

    # Initialize filtered signal
    x_filtered = np.copy(x_raw)
    T = x_raw.shape[1]

    for t in range(T):
        # Define the window range
        start_idx = max(0, t - half_wlen)
        end_idx = min(T, t + half_wlen + 1)
        window = x_raw[:, start_idx:end_idx]

        # Apply mean or median filtering
        if method == 'MEAN':
            avg = np.mean(window, axis=1)
        else:
            avg = np.median(window, axis=1)

        # Detect and replace outliers
        er = np.abs(x_raw[:, t] - avg)
        replace = er >= diff_threshold[:, np.newaxis]
        x_filtered[replace, t] = avg[replace]

    return x_filtered