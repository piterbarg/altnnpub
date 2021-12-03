# Alternatives to Deep Neural Networks for Function Approximations in Finance
## Code companion repo 

### Overview

This is a repository of Python code to go with our paper whose details could be found below 

We provide our implementations of the generalized stochastic sampling (gSS) and functional Tensor Train (fTT) algorithms from the paper, and related routines. This is a somewhat simplified version of the code that produced the test results that we reported.  Simplifications were made to improve clarity and increase general didactic value, at a (small) expense of cutting out some of the secondary tricks and variations. 

The code is released under the [MIT License](LICENSE)

### Installing the code

You do not _have to_ install this package once you have downloaded it -- see the next section on how to use it without any installation. But if you want to call our routines from a different project or directory, execute the following (note you need to run this from  `altnnpub` directory, assuming this is the root of the project directory -- the directory where this file that you are reading is located)
```
altnnpub>pip install -e .
```

Then you can call various methods from your code like this
```
from nnu import gss_kernels
kernel = gss_kernels.global_kernel_dict(1.0)['invquad']
...
```

to uninstall the package, run (from anywhere)
```
blah>pip uninstall altnnpub
```

### Running the code

The main entry point to the code is [main.py](./nnu/main.py) in `./nnu` folder.  Assuming the project directory is called `altnnpub`, the code is run via Python module syntax
```
altnnpub>python -m nnu.main
```
Various options such as which functions to fit, which models to use, and so on can be set in `main.py`

Results are reported in the terminal and are also stored in `./results` directory

All of our (non-test) Python code is in `./nnu` directory

### Jupyter notebooks
We provide a number of notebooks that demonstrate, at varying levels of detail, how to build and use certain models

* [ftt_als_01.ipynb](ftt_als_01.ipynb): Functional Tensor Train (fTT) approximation using the Alternating Least Squares (ALS) algorithm
* [functional_2D_low_rank_01.ipynb](functional_2D_low_rank_01.ipynb): Low-rank functional approximation of 2D functions done manually. This is an illustrative example of ALS applied to calculate successive  rank-1 approximations, as described in the paper
* [gss_example_keras_direct_01.ipynb](gss_example_keras_direct_01.ipynb): Create and test the generalized Stochastic Sampling (gSS) model. In this notebook do it "by hand", ie using granular interfaces such as the `Keras` functional interface. Here we create a `hidim` version of the model with the `Adam` optimizer for the frequency bounds (aka scales) and linear regression for the outer (linear) weights
* [gss_example_model_factory_01.ipynb](gss_example_model_factory_01.ipynb): Create and test the generalized Stochastic Sampling (gSS) model. This notebook uses `gss_model_factory` and other higher-level interfaces that the main entry point (`./nnu/main.py`) eventually calls. We create a `onedim` version of the model with a one-dim optimizer for the frequency bounds (aka scales) and linear regression for the outer (linear) weights 

### Test suite
Unit tests are collected in `./test` directory and provide useful examples of how different parts of the code can be used. The test suite can be run in the standard Python way using `pytest`, e.g. from the comamnd line at the project root directory:
```
altnnpub>pytest
```
Pytest is installed with `pip install pytest` command

Individual tests can be run using a `pytest -k test_blah` type command, which could be useful for debugging. This is all very well explained in pytest [documentation](https://docs.pytest.org/en/6.2.x/contents.html)

Tests are there predominantly to show how to call certain functions. They mostly test that the code simply runs rather than testing numbers, etc. except for tests in [test_gss_report_generator.py](./test/test_gss_report_generator.py) where actual fitting results are compared to the expected ones.  Tests  produce various output that could be interesting to see -- option `pytest -s` will print out whatever the tests are printing out

### Requirements

The code has been tested with Python 3.7 and 3.8. See [requirements.txt](requirements.txt) for required packages

### Contacting us
Our contact details are in the SSRN link below

### Details of the paper

>Antonov, Alexandre and Piterbarg, Vladimir, Alternatives to Deep Neural Networks for Function Approximations in Finance (November 7, 2021). Available at SSRN: https://ssrn.com/abstract=3958331 or http://dx.doi.org/10.2139/ssrn.3958331


