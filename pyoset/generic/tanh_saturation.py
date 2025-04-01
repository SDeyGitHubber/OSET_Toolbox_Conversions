import numpy as np

def tanh_saturation(x, param, mode='ksigma'):
    """
    Tanh saturation of outlier samples.

    Args:
        x (np.ndarray): Input data (channels x time).
        param (float or np.ndarray): Scaling factor or absolute thresholds.
        mode (str): 'ksigma' or 'absolute' mode (default: 'ksigma').

    Returns:
        y (np.ndarray): Saturated output.
    """
    
    # Validate mode
    if mode not in ['ksigma', 'absolute']:
        raise ValueError("Invalid mode. Use 'ksigma' or 'absolute'.")

    # Scale factor calculation
    if mode == 'ksigma':
        alpha = param * np.std(x, axis=1, keepdims=True)
    elif mode == 'absolute':
        if np.isscalar(param):
            alpha = np.full((x.shape[0], 1), param)
        elif isinstance(param, (list, np.ndarray)) and len(param) == x.shape[0]:
            alpha = np.array(param)[:, np.newaxis]
        else:
            raise ValueError("Invalid param dimensions.")

    # Saturation transformation
    y = alpha * np.tanh(x / alpha)

    return y