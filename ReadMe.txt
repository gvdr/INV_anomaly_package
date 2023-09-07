Development repository for outlier detection of ION


To use, clone the repo. Make a virtual environment as follows: (works unter python 3.10.) 

in the root folder of the repo (INV_...)

python3.10 -m venv "venv_p3.10"
source venv_p3.10/bin/activate
pip install -r requirements.txt

(For a complete list of required packages (from pip freeze), see requirements.txt)

Then, install the outlier detection package by going to the outlierdetection folder and running

poetry install

There should now be a link to outlierdetection/src in the site-packages folder of the virtual environment venv_p3.10 

The package should be importable now as, e.g., 

import outlierdetection.univariate as UOD



automatic tests are available: in folder outlierdetection, run
pytest tests/
pytest tests/ --cov=outlierdetection --cov-report term-missing

Some notebooks testing the package are in the folder development_notebooks
