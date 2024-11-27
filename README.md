# mgo-with-automatic-analytic-continuation
This little repository demonstrates the new algorithm for metaplectic geometrical optics, which uses complex ray tracing to do automatic analytic continuation (AAC) needed for the steepest descent integrals when evaluating the inverse metaplectic transform.

## Getting Started
This project uses Python and iPython notebooks. The necessary python packages are listed in the `requirements.txt` file. To reproduce the main results, I recommend using the Anaconda-distribution of python and creating a virtual environment by running the following commands in a terminal (e.g. from Visual Studio Code):
```
conda create -n "mgo-aac" python=3.11
conda activate mgo-aac
conda install pip
pip install -r requirements.txt
```

## Reproduce the Main Results
The main result for the Airy problem is seen in the Jupyter Notebook:
1. `mgo_aac_airy.ipynb`
