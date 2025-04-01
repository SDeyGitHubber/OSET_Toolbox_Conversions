import unittest
import numpy as np
import matlab
import matlab.engine
from pyoset.ecg.ecg_polarity import ecg_polarity  # Python implementation

class TestECGPolarity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Start MATLAB engine and add OSET path before running tests."""
        cls.eng = matlab.engine.start_matlab()
        cls.eng.addpath(cls.eng.genpath('OSET/matlab'), nargout=0)

    @classmethod
    def tearDownClass(cls):
        """Close MATLAB engine after tests."""
        cls.eng.quit()

    def test_ecg_polarity(self):
        """Compare MATLAB and Python implementations of ecg_polarity."""
        test_cases = [
            np.random.randn(3, 1000),   # 3 leads, 1000 samples each
            np.random.randn(1, 500),    # 1 lead, 500 samples
            np.random.randn(5, 200),    # 5 leads, 200 samples
            np.random.randn(10, 1500),  # 10 leads, 1500 samples
        ]
        
        fs = 500  # Sampling frequency in Hz

        for ecg in test_cases:
            with self.subTest(ecg=ecg):
                # 🛠️ Run Python implementation
                polarity_py = ecg_polarity(ecg, fs)

                # 🔥 Convert to MATLAB format
                ecg_matlab = matlab.double(ecg.tolist())  
                fs_matlab = float(fs)  # Ensure MATLAB receives float type
                polarity_matlab = np.array(self.eng.ecg_polarity(ecg_matlab, fs_matlab)).flatten().astype(bool)  # Flatten output

                # ✅ Compare results
                np.testing.assert_array_equal(polarity_py, polarity_matlab, err_msg="Polarity mismatch!")

if __name__ == '__main__':
    unittest.main()