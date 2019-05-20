# GLYFE

[![DOI](https://zenodo.org/badge/184261006.svg)](https://zenodo.org/badge/latestdoi/184261006)

GLYFE is a glucose predictive models benchmark. <!--It has been described in the paper "GLYFE: Benchmark of Personalized Glucose
Predictive Models in Type 1 Diabetes", published in IEEE Transactions on Biomedical Engineering.-->

## Getting Started

These instructions will help you get the data needed to run the benchmark as well as to develop new glucose predictive models.

### Prerequisites

To simulate the data need to run the benchmark, you will need a [MATLAB](https://fr.mathworks.com/products/matlab.html)(the R2018b version has been used here) and a [T1DMS licence](https://tegvirginia.com/software/t1dms/) (v3.2.1).

To run the benchmark, you will need the following ```Python 3.6``` libraries
```
Keras 2.2.4
numpy 1.15.4
pandas 0.23.4
pip 9.0.1
scikit-learn 0.20.1
scipy 1.1.0
setuptools 28.8.0
statsmodels 0.9.0
tensorboard 1.12.0
tensorflow-gpu 1.12.0
```

### Data Simulation

* Copy the ```GLYFE.scn``` scenario file into the  into the %matlab_t1dms_scenario_folder%. The files describes the scenario the virtual patients will follow during the simulation.
* Copy the %mat2csv_file% into the %matlab_t1dms_folder%
* Set random seed in Matlab console
* Launch T1DMS GUI:
** Select scenario file, 
** Add the 10 adults, adolescents, and children
** Select IV sensor and pumps
** Set seed to 1
** Launch Simulation
* Convert the simulated .mat files into CSV using the function %mat2csv_file%
``` test test ```
* Make sure the simulated data are right by check the CHA52 checksum, which should be equal to
``` checksum ```

## How to use the benchmark

### Run the benchmark on an existing model

### Run the benchmark on a new model


## TODO

* exclude basal insulin from data in dat2csv in Matlab
