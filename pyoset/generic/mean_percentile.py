import numpy as np

def mean_percentile(data, lower_percentile, upper_percentile):
    """
    Calculate the mean of the data within the specified percentile range for each column.

    Args:
        data (np.ndarray): Matrix where each column represents a separate dataset.
        lower_percentile (float): Lower percentile (e.g., 2.5).
        upper_percentile (float): Upper percentile (e.g., 97.5).

    Returns:
        mean_values (np.ndarray): Row vector containing the mean within the percentile range for each column.
    """
    # ✅ Input validation
    if data.size == 0:
        raise ValueError("Input data cannot be empty")

    if lower_percentile < 0 or upper_percentile > 100 or lower_percentile >= upper_percentile:
        raise ValueError("Invalid percentile range. Ensure 0 <= lower_percentile < upper_percentile <= 100")

    # ✅ Initialize output array
    mean_values = np.zeros(data.shape[1])

    # ✅ Calculate mean for each column within percentile range
    for i in range(data.shape[1]):
        col_data = data[:, i]

        # Calculate percentile bounds
        lower, upper = np.percentile(col_data, [lower_percentile, upper_percentile])

        # Filter data within percentile range
        filtered_data = col_data[(col_data >= lower) & (col_data <= upper)]

        # Compute mean of filtered data
        mean_values[i] = np.mean(filtered_data)

    return mean_values