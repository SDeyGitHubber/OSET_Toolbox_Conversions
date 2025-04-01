import numpy as np

def robust_weighted_average(x):
    """
    Robust weighted averaging of biomedical signals.

    Args:
        x (np.ndarray): An (N x T) matrix containing N ensembles of a noisy event-related signal of length T.

    Returns:
        mn (np.ndarray): The robust weighted average over the N rows of x.
        vr_mn (np.ndarray): The variance of the average beat across the N rows of x.
        md (np.ndarray, optional): The robust weighted median over the N rows of x.
        vr_md (np.ndarray, optional): The variance of the median beat across the N rows of x.
    """
    num_beats, _ = x.shape

    if num_beats > 1:
        # ✅ Average beat (mean)
        mn0 = np.mean(x, axis=0)
        noise0 = x - mn0
        vr = np.var(noise0, axis=1, ddof=0)
        sm = np.sum(1 / vr)
        weight = 1 / (vr * sm)
        mn = np.dot(weight, x)
        
        noise = x - mn
        vr_mn = np.var(noise, axis=0, ddof=0)

        # ✅ Median calculation (optional)
        md, vr_md = None, None
        if x.shape[0] > 2:  # Check if median is required
            md0 = np.median(x, axis=0)
            noise0 = x - md0
            vr = np.var(noise0, axis=1, ddof=0)
            sm = np.sum(1 / vr)
            weight = 1 / (vr * sm)
            md = np.dot(weight, x)
            
            noise = x - md
            vr_md = np.var(noise, axis=0, ddof=0)

        return (mn, vr_mn, md, vr_md) if md is not None else (mn, vr_mn)

    else:
        # ✅ Single row case
        mn = x[0]
        vr_mn = np.zeros_like(x[0])
        md = x[0] if x.shape[0] > 2 else None
        vr_md = np.zeros_like(x[0]) if md is not None else None

        return (mn, vr_mn, md, vr_md) if md is not None else (mn, vr_mn)
