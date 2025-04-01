import unittest
import numpy as np
import matlab
from matlab import engine

from pyoset.generic.linear_warp import linear_warp as warp_py  # Import Python function


class TestLinearWarp(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Start MATLAB engine
        cls.eng = matlab.engine.start_matlab()
        cls.eng.addpath(cls.eng.genpath('OSET/matlab'), nargout=0)

    @classmethod
    def tearDownClass(cls):
        # Stop MATLAB engine
        cls.eng.quit()

    def test_linear_warp(self):
        """Test MATLAB vs Python linear_warp."""
        
        # ✅ Custom test cases
        test_cases = [
            (np.linspace(0, 10, 20), 50),    # Vector case: 20 → 50 elements
            (np.linspace(-5, 5, 10), 30),    # Vector case: 10 → 30 elements
            (np.random.randn(10, 20), (15, 30)),  # Matrix case: 10x20 → 15x30
            (np.random.randn(5, 10), (8, 12)),    # Matrix case: 5x10 → 8x12
            (np.ones((4, 4)) * 100, (10, 10))     # Matrix case: constant values
        ]

        # ✅ Comparison loop
        for x, L in test_cases:
            with self.subTest(x=x, L=L):
                # 🛠️ Python result
                y_py = warp_py(x, L)

                # 🛠️ MATLAB result
                x_matlab = matlab.double(x.tolist())
                L_matlab = matlab.double([L]) if isinstance(L, int) else matlab.double(L)

                y_matlab = self.eng.linear_warp(x_matlab, L_matlab, nargout=1)
                y_matlab = np.array(y_matlab).squeeze()  # 🔥 Ensure correct shape

                # ✅ Statistical checks
                diff = np.abs(y_py - y_matlab)
                max_diff = np.max(diff)
                mean_diff = np.mean(diff)
                std_diff = np.std(diff)

                print(f"✅ Shape: {x.shape if x.ndim == 2 else len(x)} → {L}")
                print(f"Max Difference: {max_diff:.6f}")
                print(f"Mean Difference: {mean_diff:.6f}")
                print(f"Std Dev Difference: {std_diff:.6f}\n")

                # ✅ Assertions
                np.testing.assert_array_almost_equal(y_py, y_matlab, decimal=5, err_msg="Mismatch in values")


if __name__ == '__main__':
    unittest.main()