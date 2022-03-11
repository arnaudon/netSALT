..  -*- coding: utf-8 -*-

.. _contents:

*netSALT*: simulating lasing networks
========================================

*netSALT* is a python package to solve the SALT equation on networkx. It builds on the theory of quantum graphs to include nonlinear modal interactions.


Installation
************

To install the dev version from `GitHub <https://github.com/imperialcollegelondon/hcga/>`_ with the commands::

       $ git clone git@github.com:ImperialCollegeLondon/netSALT.git
       $ cd netSALT 
       $ pip install .

There is no PyPi release version yet, but stay tuned for more!


Usage
*****

See the folder `example` with the sequence of scripts, from 0 to 0.

Citing
******

To cite *netSALT*, please use [paper]_. 

Credits
*******

The code is still a preliminary version, and written by us.

Original authors:
*****************

- Alexis Arnaudon, GitHub: `arnaudon <https://github.com/arnaudon>`_

Contributors:
*************

- Dhrub Saxena, Github: 


Bibliography
************

.. [paper] A. Arnaudon, D. Saxena, R. Sapienza, etc...
                “Lasing on networks”, In preparation, 2020

Code documentation
******************

Documentation of the code.

.. toctree::
    :maxdepth: 3

    modes
    physics
    quantum_graph
    algorithm
    pump
    plotting
    io
    utils

Indices and tables
******************

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
