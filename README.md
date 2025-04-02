# Signal Processing & Model Conversion Project

I have sourced the existing repository from (https://github.com/alphanumericslab/OSET) to demonstrate some code conversions, with unit tests that I have implemented( and have not yet been converted to Python). This repository is sourced fromThis repository is focused on converting and optimizing MATLAB-based signal processing tools (EEG seizure detection, ECG feature extraction, signal quality index assessment, among others.) to Python. The project includes conversion of code with exat same functionalities from MATLAB to Python, with unit tests and notebooks wherever necessary.

## Important Folders

* The pyoset folder consists of the Python codes which I implemented corresponding to their MATLAB counterparts in OSET/matlab/tools
* pytests folder consists of unit tests which I implemented  that compare the MATLAB and Python implementations
* notebook folder consists of notebooks which I implemented for necessary visualisations and analysis for some implementations
  
## Requirements

Before you start working with this project, make sure to have Python 3.8 or higher installed on your machine.

### Install Dependencies

To install all required libraries, use the `requirements.txt` file provided.
```bash
pip install -r requirements.txt
```
Run the next command to try the unit test as shown here.
```bash
python -m unittest <unit_test_file_name>
```
