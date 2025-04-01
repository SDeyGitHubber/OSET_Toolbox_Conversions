import unittest
import numpy as np
import matlab
from matlab import engine
from pyoset.generic.mean_percentile import mean_percentile as py_mean

class TestMeanPercentile(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Start MATLAB engine
        cls.eng = matlab.engine.start_matlab()
        cls.eng.addpath(cls.eng.genpath('OSET/matlab'), nargout=0)

    @classmethod
    def tearDownClass(cls):
        # Stop MATLAB engine
        cls.eng.quit()

    def test_mean_percentile(self):
        """Test MATLAB vs Python mean_percentile."""
        
        # ✅ Test cases
        test_cases = [
            (np.random.randn(100, 5), 2.5, 97.5),        # Random data
            (np.random.randn(50, 10), 10, 90),           # Random data with wider range
            (np.ones((30, 5)) * 50, 20, 80),             # Constant values
            (np.linspace(0, 100, 200).reshape(100, 2), 5, 95),  # Linearly increasing data
        ]

        # ✅ Edge case: Empty data
        with self.assertRaises(ValueError):
            py_mean(np.array([]), 10, 90)

        # ✅ Edge case: Invalid percentiles
        with self.assertRaises(ValueError):
            py_mean(np.random.randn(10, 5), -10, 110)

        for data, lower, upper in test_cases:
            with self.subTest(data=data, lower=lower, upper=upper):
                # 🛠️ Python result
                py_result = py_mean(data, lower, upper)

                # 🛠️ MATLAB result
                mat_data = matlab.double(data.tolist())
                mat_result = self.eng.mean_percentile(mat_data, lower, upper, nargout=1)
                mat_result = np.array(mat_result).flatten()

                # ✅ Statistical comparison
                diff = np.abs(py_result - mat_result)
                max_diff = np.max(diff)
                mean_diff = np.mean(diff)

                print(f"✅ Lower: {lower}, Upper: {upper}")
                print(f"Max Difference: {max_diff:.8f}")
                print(f"Mean Difference: {mean_diff:.8f}\n")

                # ✅ Adjust tolerance to accommodate minor floating-point differences
                np.testing.assert_allclose(
                    py_result, mat_result, atol=1e-4, rtol=1e-4, 
                    err_msg="Mismatch in percentile mean"
                )

if __name__ == '__main__':
    unittest.main()
