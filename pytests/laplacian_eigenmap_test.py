import unittest
import numpy as np
import matlab
from matlab import engine

from pyoset.generic.laplacian_eigenmap import laplacian_eigenmap as lemap_py


class TestLaplacianEigenmap(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Start MATLAB engine
        cls.eng = matlab.engine.start_matlab()
        cls.eng.addpath(cls.eng.genpath('OSET/matlab'), nargout=0)

    @classmethod
    def tearDownClass(cls):
        # Stop MATLAB engine
        cls.eng.quit()

    def test_laplacian_eigenmap(self):
        """Test MATLAB vs Python Laplacian Eigenmap."""

        # ✅ Test cases with random SPD matrices
        np.random.seed(42)
        K = 5
        N = 4
        kappa = 0.5

        C = np.random.randn(N, N, K)
        for i in range(K):
            C[:, :, i] = C[:, :, i] @ C[:, :, i].T  # Make SPD

        # 🛠️ Python result
        V_py, d_py, delta_py, similarity_py, epsilon_py = lemap_py(C, kappa)

        # 🛠️ MATLAB result
        C_matlab = matlab.double(C.tolist())
        V_matlab, d_matlab, delta_matlab, similarity_matlab, epsilon_matlab = self.eng.laplacian_eigenmap(C_matlab, kappa, nargout=5)

        # ✅ Convert MATLAB outputs to NumPy
        V_matlab = np.array(V_matlab)
        d_matlab = np.array(d_matlab).flatten()
        delta_matlab = np.array(delta_matlab)
        similarity_matlab = np.array(similarity_matlab)
        epsilon_matlab = float(epsilon_matlab)

        # ✅ Statistical checks
        diff_V = np.abs(V_py - V_matlab)
        diff_d = np.abs(d_py - d_matlab)

        print(f"✅ Max Diff (V): {np.max(diff_V):.6f}")
        print(f"✅ Mean Diff (V): {np.mean(diff_V):.6f}")
        print(f"✅ Max Diff (d): {np.max(diff_d):.6f}")
        print(f"✅ Mean Diff (d): {np.mean(diff_d):.6f}")

        # ✅ Assertions
        np.testing.assert_array_almost_equal(V_py, V_matlab, decimal=5, err_msg="Mismatch in eigenvectors")
        np.testing.assert_array_almost_equal(d_py, d_matlab, decimal=5, err_msg="Mismatch in eigenvalues")


if __name__ == '__main__':
    unittest.main()
