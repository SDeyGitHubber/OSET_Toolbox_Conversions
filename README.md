# Signal Processing & Model Conversion Project

This repository is focused on converting and optimizing MATLAB-based signal processing tools (EEG seizure detection, ECG feature extraction, signal quality index assessment, among others.) to Python. The proejct includes conversion of code with exat same functionalities from MATLAB to Python, with unit tests and notebooks wherever necessary.

## Important Folders

* The pyoset folder consists of the Python codes corresponding to their MATLAB counterparts in OSET/matlab/tools
* pytests folder consists of unit tests that compare the MATLAB and Python implementations
* notebook folder consists of notebooks for necessary visualisations and analysis for some implementations
  
## Requirements

Before you start working with this project, make sure to have Python 3.8 or higher installed on your machine.

### Install Dependencies

To install all required libraries, use the `requirements.txt` file provided, and then prun the next command to try the unit test as shown here.

```bash
pip install -r requirements.txt

python -m unittest <unit_test_file_name>
