import numpy as np
from scipy.linalg import eig

def laplacian_eigenmap(C, kappa):
    """
    Laplacian eigen-map of positive semi-definite matrix pairs.

    Args:
        C (np.ndarray): Tensor of shape (N, N, K) with K semi-positive definite matrices.
        kappa (float): Median factor for distance normalization.

    Returns:
        V (np.ndarray): Eigenvectors of the Laplacian eigen-map.
        d (np.ndarray): Eigenvalues of the Laplacian eigen-map.
        delta (np.ndarray): Eigen-map distance matrix.
        similarity (np.ndarray): Similarity matrix.
        epsilon (float): kappa times the median of delta.
    """
    K = C.shape[2]
    delta = np.zeros((K, K))

    # ✅ Pairwise eigen-map distance calculation
    for i in range(K - 1):
        for j in range(i + 1, K):
            lambdas = eig(C[:, :, i], C[:, :, j])[0]
            delta[i, j] = np.sum(np.log(np.abs(lambdas))**2)
            delta[j, i] = delta[i, j]  # Symmetric matrix

    # ✅ Epsilon and similarity matrix
    mask = np.triu(np.ones((K, K), dtype=bool), 1)
    dd = delta[mask]
    epsilon = kappa * np.median(dd)
    similarity = np.exp(-delta / epsilon)

    # ✅ Laplacian eigen-map calculation
    S = np.diag(1 / np.sqrt(np.sum(similarity, axis=1)))
    L = S @ similarity @ S
    eig_vals, eig_vecs = np.linalg.eig(L)

    # ✅ Sort eigenvalues in descending order
    idx = np.argsort(-eig_vals)
    d = eig_vals[idx]
    V = eig_vecs[:, idx]

    return V, d, delta, similarity, epsilon