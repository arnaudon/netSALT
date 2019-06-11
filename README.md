Code simulating quantum graphs. 
================================

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


There are several test jupyter-notebooks:

 1. _test_NAQ_U1_closed.ipynb_
  > classic closed quantum graphs, with real wavenumber
 2. _test_NAQ_U1_open.ipynb_ 
  > open quantum graphs, with complex wavenumbers
 3. _test_NAQ_U1_open_pump.ipynb_ 
  > lasing quantum graph, with complex wavenumbers and a pump (requires to run step 2 to get the modes)
 4. _test_NAQ_U1_transport.ipynb_
  > transport quantum graphs, with real wavenumber and an input/output at the node level
  
  
## Non-abelian quantum graphs

For non-abelian quantum graphs, use the brach 'dev-so3', where only the grop SO(3) has been implemented. 
This branch is work in progress, so don't try to look at it yet!



