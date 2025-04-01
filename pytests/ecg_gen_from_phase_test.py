import unittest
import numpy as np
import matlab
from matlab import engine
from pyoset.modelling.ecg_gen_from_phase import ecg_gen_from_phase as py_ecg

class TestECGGenFromPhase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Start MATLAB engine
        cls.eng = matlab.engine.start_matlab()
        cls.eng.addpath(cls.eng.genpath('OSET/matlab'), nargout=0)

    @classmethod
    def tearDownClass(cls):
        # Stop MATLAB engine
        cls.eng.quit()

    def test_ecg_gen_from_phase(self):
        """Test MATLAB vs Python ECG generator."""

        # ✅ Test cases
        test_cases = [
            # (params, phase)
            ({'alpha': [1.0, 0.8, 0.6], 'b': [0.2, 0.3, 0.25], 'theta': [0, np.pi/2, np.pi]}, np.linspace(0, 2*np.pi, 100)),
            ([1.0, 0.8, 0.6, 0.2, 0.3, 0.25, 0, np.pi/2, np.pi], np.linspace(0, 2*np.pi, 200)),
            ([0.5, 0.4, 0.3, 0.1, 0.15, 0.2, 0, np.pi/4, np.pi/2], np.linspace(0, 2*np.pi, 150)),
        ]

        for params, phase in test_cases:
            with self.subTest(params=params, phase=phase):
                # 🛠️ Python result
                py_result = py_ecg(params, phase)

                # 🛠️ MATLAB result
                if isinstance(params, dict):
                    mat_params = {k: matlab.double(v) for k, v in params.items()}
                else:
                    mat_params = matlab.double(params)

                mat_phase = matlab.double(phase.tolist())
                mat_result = self.eng.ecg_gen_from_phase(mat_params, mat_phase, nargout=1)
                mat_result = np.array(mat_result).flatten()

                # ✅ Statistical comparison
                diff = np.abs(py_result - mat_result)
                max_diff = np.max(diff)
                mean_diff = np.mean(diff)

                print(f"✅ Max Difference: {max_diff:.8f}")
                print(f"✅ Mean Difference: {mean_diff:.8f}\n")

                # ✅ Assert similarity with tolerance
                np.testing.assert_allclose(py_result, mat_result, atol=1e-4, rtol=1e-4, err_msg="Mismatch in ECG output")

if __name__ == '__main__':
    unittest.main()