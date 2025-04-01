import unittest
import numpy as np
import matlab
from matlab import engine
from pyoset.generic.skew import skew as py_skew

class TestSkewness(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Start MATLAB engine and add OSET path."""
        cls.eng = matlab.engine.start_matlab()
        cls.eng.addpath(cls.eng.genpath('OSET/matlab'), nargout=0)

    @classmethod
    def tearDownClass(cls):
        """Stop MATLAB engine."""
        cls.eng.quit()

    def test_skew_matlab_vs_python(self):
        """Test MATLAB vs Python skewness calculation."""

        # ✅ Test cases: [rows x columns]
        test_cases = [
            np.random.normal(0, 1, (10, 100)),    # Normally distributed data
            np.random.exponential(1.0, (5, 150)),  # Positively skewed
            -np.random.exponential(1.0, (5, 150)), # Negatively skewed
            np.ones((4, 200)),                    # Constant values (zero skew)
            np.random.uniform(-1, 1, (7, 50))      # Uniform distribution
        ]

        for i, data in enumerate(test_cases):
            with self.subTest(i=i):
                # 🛠️ Python skewness calculation
                py_skw, py_m, py_sd = py_skew(data)

                # 🛠️ MATLAB skewness calculation
                mat_data = matlab.double(data.tolist())
                mat_skw, mat_m, mat_sd = self.eng.skew(mat_data, nargout=3)

                # Convert MATLAB output to NumPy
                mat_skw = np.array(mat_skw).flatten()
                mat_m = np.array(mat_m).flatten()
                mat_sd = np.array(mat_sd).flatten()

                # ✅ Print differences for validation
                diff_skw = np.abs(py_skw - mat_skw)
                diff_m = np.abs(py_m - mat_m)
                diff_sd = np.abs(py_sd - mat_sd)

                print(f"\n✅ Test Case {i+1}")
                print(f"✅ Max Skew Difference: {np.max(diff_skw):.8f}")
                print(f"✅ Mean Skew Difference: {np.mean(diff_skw):.8f}")
                print(f"✅ Max Mean Difference: {np.max(diff_m):.8f}")
                print(f"✅ Max SD Difference: {np.max(diff_sd):.8f}\n")

                # ✅ Assert similarity with tolerance
                np.testing.assert_allclose(py_skw, mat_skw, atol=1e-4, rtol=1e-4, err_msg="Mismatch in skewness")
                np.testing.assert_allclose(py_m, mat_m, atol=1e-4, rtol=1e-4, err_msg="Mismatch in mean")
                np.testing.assert_allclose(py_sd, mat_sd, atol=1e-4, rtol=1e-4, err_msg="Mismatch in standard deviation")

if __name__ == '__main__':
    unittest.main()