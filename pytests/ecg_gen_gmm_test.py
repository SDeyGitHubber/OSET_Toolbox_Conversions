import unittest
import numpy as np
import matlab
from matlab import engine
from pyoset.modelling.ecg_gen_gmm import ecg_gen_gmm as py_ecg

class TestECGGenGMM(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Start MATLAB engine
        cls.eng = matlab.engine.start_matlab()
        cls.eng.addpath(cls.eng.genpath('OSET/matlab'), nargout=0)

    @classmethod
    def tearDownClass(cls):
        # Stop MATLAB engine
        cls.eng.quit()

    def test_ecg_gen_gmm(self):
        """Test MATLAB vs Python GMM-based ECG generator."""

        # ✅ Test cases
        test_cases = [
            # (phi, theta0, alpha, b, theta)
            (np.linspace(-np.pi, np.pi, 100), 0.2, [1.0, 0.8, 0.6], [0.2, 0.3, 0.25], [0, np.pi/2, np.pi]),
            (np.linspace(-np.pi, np.pi, 200), 0.1, [0.9, 0.7, 0.5], [0.15, 0.2, 0.18], [np.pi/4, np.pi/2, 3*np.pi/4]),
            (np.linspace(-np.pi, np.pi, 150), -0.1, [0.5, 0.4, 0.3], [0.1, 0.15, 0.2], [0, np.pi/4, np.pi/2]),
        ]

        for phi, theta0, alpha, b, theta in test_cases:
            with self.subTest(phi=phi, theta0=theta0, alpha=alpha, b=b, theta=theta):
                # 🛠️ Python result
                py_ecg_signal, py_phi = py_ecg(phi, theta0, np.array(alpha), np.array(b), np.array(theta))

                # 🛠️ MATLAB result
                mat_phi = matlab.double(phi.tolist())
                mat_theta0 = float(theta0)
                mat_alpha = matlab.double(alpha)
                mat_b = matlab.double(b)
                mat_theta = matlab.double(theta)

                mat_result = self.eng.ecg_gen_gmm(mat_phi, mat_theta0, mat_alpha, mat_b, mat_theta, nargout=2)
                mat_ecg = np.array(mat_result[0]).flatten()
                mat_phi = np.array(mat_result[1]).flatten()

                # ✅ Statistical comparison
                diff_ecg = np.abs(py_ecg_signal - mat_ecg)
                diff_phi = np.abs(py_phi - mat_phi)

                max_diff_ecg = np.max(diff_ecg)
                max_diff_phi = np.max(diff_phi)

                print(f"✅ Max Difference (ECG): {max_diff_ecg:.8f}")
                print(f"✅ Max Difference (Phi): {max_diff_phi:.8f}\n")

                # ✅ Assert similarity with tolerance
                np.testing.assert_allclose(py_ecg_signal, mat_ecg, atol=1e-4, rtol=1e-4, err_msg="Mismatch in ECG output")
                np.testing.assert_allclose(py_phi, mat_phi, atol=1e-4, rtol=1e-4, err_msg="Mismatch in Phi output")

if __name__ == '__main__':
    unittest.main()