import unittest
import numpy as np
import matlab
from matlab import engine
from pyoset.generic.outlier_filter import outlier_filter as py_filter

class TestOutlierFilter(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Start MATLAB engine
        cls.eng = matlab.engine.start_matlab()
        cls.eng.addpath(cls.eng.genpath('OSET/matlab'), nargout=0)

    @classmethod
    def tearDownClass(cls):
        # Stop MATLAB engine
        cls.eng.quit()

    def test_outlier_filter(self):
        """Test MATLAB vs Python outlier_filter."""

        # ✅ Test cases
        test_cases = [
            (np.random.randn(10, 50), 'MEAN', 5, 95),    # Random noise with MEAN method
            (np.random.randn(5, 100), 'MEDIAN', 10, 90), # Random noise with MEDIAN method
            (np.ones((10, 30)) * 50, 'MEAN', 3, 99),     # Constant signal, MEAN
            (np.linspace(0, 1, 200).reshape(10, 20), 'MEDIAN', 4, 97),  # Linear data, MEDIAN
        ]

        for x_raw, method, half_wlen, percentile in test_cases:
            with self.subTest(x_raw=x_raw, method=method, half_wlen=half_wlen, percentile=percentile):
                # 🛠️ Python result
                py_result = py_filter(x_raw, method, half_wlen, percentile)

                # 🛠️ MATLAB result
                x_matlab = matlab.double(x_raw.tolist())
                mat_result = self.eng.outlier_filter(x_matlab, method, half_wlen, percentile, nargout=1)
                mat_result = np.array(mat_result)

                # ✅ Statistical comparison
                diff = np.abs(py_result - mat_result)
                max_diff = np.max(diff)
                mean_diff = np.mean(diff)

                print(f"✅ Method: {method}, Half Window: {half_wlen}, Percentile: {percentile}")
                print(f"Max Difference: {max_diff:.8f}")
                print(f"Mean Difference: {mean_diff:.8f}\n")

                # ✅ Allowing small tolerance for floating-point mismatches
                np.testing.assert_allclose(py_result, mat_result, atol=1e-4, rtol=1e-4, err_msg="Mismatch in filtered output")

if __name__ == '__main__':
    unittest.main()