Code simulating quantum graphs. 
================================

This code can be used to simulate various types of quantum graphs, from the classic ones, to open quantum graphs or non-abelian quantum graphs. 

For more on the mathematics behind it, see the related publications: 
 - _Non-abelian quantum graphs_, A.A., In preparation
 - _Lasing on networks_, A.A. et al. , In preparation

## Installation

To install it, you need to have python3 as well as:
- python3
- numpy/scipy/matplotlib
- networkx
- multiprocessing
- tqdm

These can nbe installed using anaconda or pip. 

Then to install the package, just run
```bash
python setup.py install
```

if you are using an anaconda environment, otherwise
```bash
python3 setup.py install --user
```

to install it to you local folders.


## Abelian quantum graphs

For the abelian symmetry group U(1), use the main branch 'master'.


There are several test jupyter-notebooks based, on a simple small-world graph:

 1. `test_NAQ_U1_closed.ipynb`
  > classic closed quantum graphs, with real wavenumber
 2. `test_NAQ_U1_open.ipynb`
  > open quantum graphs, with complex wavenumbers
 3. `test_NAQ_U1_open_pump.ipynb` 
  > lasing quantum graph, with complex wavenumbers and a pump (requires to run step 2 to get the modes)
 4. `test_NAQ_U1_transport.ipynb
  > transport quantum graphs, with real wavenumber and an input/output at the node level
  
The test 2 and 3 are using a slightly more advanced graph generation, with more flexibility for scripting, etc... clean example for that will come out shortly. 
  
## Non-abelian quantum graphs

For non-abelian quantum graphs, use the brach 'dev-so3', where only the grop SO(3) has been implemented. 
This branch is work in progress, so don't try to look at it yet!



