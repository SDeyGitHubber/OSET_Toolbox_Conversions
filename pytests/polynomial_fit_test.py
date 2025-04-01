import unittest
import numpy as np
import matlab
from matlab import engine

from pyoset.generic.polynomial_fit import polynomial_fit as fit_py


class TestPolynomialFit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # ✅ Start MATLAB engine
        cls.eng = matlab.engine.start_matlab()
        cls.eng.addpath(cls.eng.genpath('OSET/matlab'), nargout=0)

    @classmethod
    def tearDownClass(cls):
        # ✅ Stop MATLAB engine
        cls.eng.quit()

    def test_polynomial_fit(self):
        """Test MATLAB vs Python polynomial fit."""

        # ✅ Test cases with custom signals
        test_cases = [
            (np.sin(np.linspace(0, 2 * np.pi, 100)), 100, 3, 'LS'),       # Sine wave, order 3, LS
            (np.random.randn(200), 200, 5, 'PINV'),                       # Random noise, order 5, PINV
            (np.ones(50) * 2, 50, 2, 'LS'),                               # Constant signal, order 2, LS
            (np.linspace(0, 10, 150)**2, 150, 4, 'PINV'),                 # Quadratic signal, order 4, PINV
        ]

        for x, fs, N, method in test_cases:
            with self.subTest(x=x, fs=fs, N=N, method=method):
                # ✅ Python result
                y_py, p_py = fit_py(x, fs, N, method)

                # ✅ Convert Python values to MATLAB-compatible types
                x_matlab = matlab.double(x.tolist())
                fs_matlab = float(fs)      # Cast to double
                N_matlab = float(N)        # Cast to double

                # ✅ MATLAB result
                try:
                    y_matlab, p_matlab = self.eng.polynomial_fit(x_matlab, fs_matlab, N_matlab, method, nargout=2)
                    y_matlab = np.array(y_matlab).flatten()
                    p_matlab = np.array(p_matlab).flatten()
                except Exception as e:
                    self.fail(f"MATLAB execution failed: {e}")

                # ✅ Ensure dimensions match
                assert y_py.shape == y_matlab.shape, "Dimension mismatch in fitted signal"
                assert p_py.shape == p_matlab.shape, "Dimension mismatch in coefficients"

                # ✅ Statistical checks
                diff_y = np.abs(y_py - y_matlab)
                diff_p = np.abs(p_py - p_matlab)

                print(f"✅ Method: {method}, Order: {N}")
                print(f"Max Diff (y): {np.max(diff_y):.6f}")
                print(f"Mean Diff (y): {np.mean(diff_y):.6f}")
                print(f"Max Diff (p): {np.max(diff_p):.6f}")
                print(f"Mean Diff (p): {np.mean(diff_p):.6f}\n")

                # ✅ Assertions
                np.testing.assert_array_almost_equal(y_py, y_matlab, decimal=5, err_msg="Mismatch in fitted signal")
                np.testing.assert_array_almost_equal(p_py, p_matlab, decimal=5, err_msg="Mismatch in coefficients")


if __name__ == '__main__':
    unittest.main()