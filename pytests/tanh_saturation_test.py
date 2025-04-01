import unittest
import numpy as np
import matlab
from matlab import engine

from pyoset.generic.tanh_saturation import tanh_saturation as tanh_py  # Import Python function


class TestTanhSaturation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Start MATLAB engine
        cls.eng = matlab.engine.start_matlab()
        cls.eng.addpath(cls.eng.genpath('OSET/matlab'), nargout=0)

    @classmethod
    def tearDownClass(cls):
        # Stop MATLAB engine
        cls.eng.quit()

    def test_tanh_saturation(self):
        """Test MATLAB vs Python tanh_saturation."""
        
        # ✅ Custom test cases
        test_cases = [
            np.random.randn(8, 100),   # 8 channels, 100 samples
            np.random.randn(4, 50) * 10,   # Scaled values
            np.random.randn(6, 120) + 5,   # Offset values
            np.ones((3, 80)) * 100,        # Large constant values
            np.zeros((2, 150)),            # All zero input
        ]

        # ✅ Parameters
        params = [
            (2.0, 'ksigma'),          # ksigma mode
            (5.0, 'ksigma'),          # Larger scaling factor
            (1.5, 'absolute'),        # Absolute threshold scalar
            ([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 'absolute'),  # Channel-specific
            (0.5, 'ksigma')           # Smaller scaling factor
        ]

        # ✅ Comparison loop
        for x in test_cases:
            for param, mode in params:
                with self.subTest(x=x, param=param, mode=mode):
                    # Python result
                    y_py = tanh_py(x, param, mode)

                    # MATLAB result
                    x_matlab = matlab.double(x.tolist())
                    param_matlab = matlab.double([param]) if np.isscalar(param) else matlab.double(param)
                    y_matlab = self.eng.tanh_saturation(x_matlab, param_matlab, mode, nargout=1)
                    y_matlab = np.array(y_matlab)

                    # ✅ Statistical checks
                    diff = np.abs(y_py - y_matlab)
                    max_diff = np.max(diff)
                    mean_diff = np.mean(diff)
                    std_diff = np.std(diff)

                    print(f"✅ Mode: {mode}, Param: {param}")
                    print(f"Max Difference: {max_diff:.6f}")
                    print(f"Mean Difference: {mean_diff:.6f}")
                    print(f"Std Dev Difference: {std_diff:.6f}\n")

                    # ✅ Assertions
                    np.testing.assert_array_almost_equal(y_py, y_matlab, decimal=6, err_msg="Mismatch in values")


if __name__ == '__main__':
    unittest.main()