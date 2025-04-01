import sys
import os
import unittest
import numpy as np
from pyoset.ecg.robust_weighted_average import robust_weighted_average as rwa_py
import matlab
from matlab import engine


class TestRobustWeightedAverage(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Start MATLAB engine
        cls.eng = matlab.engine.start_matlab()
        cls.eng.addpath(cls.eng.genpath('OSET/matlab'), nargout=0)

    @classmethod
    def tearDownClass(cls):
        # Stop MATLAB engine
        cls.eng.quit()

    def test_robust_weighted_average(self):
        # Define test cases
        test_cases = [
            np.random.randn(10, 50),   # 10 beats, 50 samples each
            np.random.randn(5, 100),   # 5 beats, 100 samples each
            np.random.randn(20, 30),   # 20 beats, 30 samples each
            np.random.randn(1, 50),    # Single beat
            np.random.randn(50, 10)    # 50 beats, 10 samples each
        ]

        for x in test_cases:
            with self.subTest(x=x):
                # 🛠️ Run Python implementation
                mn_py, vr_mn_py, *opt_py = rwa_py(x)

                # 🔥 Ensure MATLAB receives the same matrix orientation
                x_matlab = matlab.double(x.tolist())  # No transpose
                result = self.eng.robust_weighted_average(x_matlab, nargout=4)

                # Extract MATLAB outputs
                mn_matlab = np.array(result[0]).flatten()
                vr_mn_matlab = np.array(result[1]).flatten()

                # Optional median and its variance
                md_matlab = np.array(result[2]).flatten() if len(result) > 2 else None
                vr_md_matlab = np.array(result[3]).flatten() if len(result) > 3 else None

                # Compare means and variances
                np.testing.assert_array_almost_equal(mn_py, mn_matlab, decimal=6, err_msg="Mean mismatch")
                np.testing.assert_array_almost_equal(vr_mn_py, vr_mn_matlab, decimal=6, err_msg="Variance mismatch")

                if md_matlab is not None and len(opt_py) > 0:
                    np.testing.assert_array_almost_equal(opt_py[0], md_matlab, decimal=6, err_msg="Median mismatch")

                if vr_md_matlab is not None and len(opt_py) > 1:
                    np.testing.assert_array_almost_equal(opt_py[1], vr_md_matlab, decimal=6, err_msg="Variance of median mismatch")


if __name__ == '__main__':
    unittest.main() 