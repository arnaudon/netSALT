Working code for non-abelian quantum graph. 

For the abelian symmetry group U(1), we recover the standard quantum graphs. 
Test file: test_NAQ_U1_closed.ipynb

Three types of open quantum graphs are implemented:
1) open graphs, test file: test_NAQ_U1_open.ipynb
2) lasing graphs, test file: test_NAQ_U1_open_pump.ipynb
2) transport graphs, test file: test_NAQ_U1_transport.ipynb 


To install it, you need to have:
- python3
- numpy/scipy/matplotlib
- networkx
- multiprocessing

then type in the main code folder

python3 setup.py install

this will intall it on you computer. 
If you are using anaconda, you're all good, if you have sudo privileges, too, otherwise use

python3 setup.py install --user

to install it to the local folders.

