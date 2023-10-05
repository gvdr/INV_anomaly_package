Development repository for outlier detection of ION


To use, clone the repo. Make a virtual environment as follows: (works unter python 3.10.) 

in the root folder of the repo (INV_...)

python3.10 -m venv "venv_p3.10"
source venv_p3.10/bin/activate
pip install -r requirements.txt

(For a complete list of required packages (from pip freeze), see requirements.txt)

Then, install the outlier detection package by running in the root folder

pip install outlierdetection --no-index --find-links outlierdetection/dist/

for the latest version, or for a specific version e.g.

pip install outlierdetection==0.1.0 --no-index --find-links outlierdetection/dist/


If new versions have been released and the wheels have been pulled to outlierdetection/dist/, upgrade via 

pip install --upgrade outlierdetection --no-index --find-links outlierdetection/dist/



Alternatively, the package can be installed via
poetry install
in the outlierdetection folder. Then however, there is a link to the outlierdetection/src folder in site-packages instead of a physical copy of the package files. 


The package should be importable now as, e.g., 
import outlierdetection.univariate as UOD


The currently installed version is accessible via
pip show outlierdetection


automatic tests are available: in folder outlierdetection, run
pytest tests/
pytest tests/ --cov=outlierdetection --cov-report term-missing


Comments:
- There is currently a bug (September 11) with sphinx 7.2.5, loading numpy.typing.ufunc. Downgrade to 7.2.4 fixes it. 
