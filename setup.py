#!/usr/bin/env python

import imp
import sys

from setuptools import setup, find_packages

if sys.version_info < (2, 7):
    sys.exit("Sorry, Python < 2.7 is not supported")

VERSION = imp.load_source("", "naq_graphs/version.py").__version__

setup(
    name="naq-graphs",
    author="Alexis Arnaudon",
    author_email="alexis.arnaudon@epfl.ch",
    version=VERSION,
    description="",
    install_requires=[
        'scipy>=1.2.0',
        'networkx',
        'matplotlib',
    ],
    packages=find_packages(),
)
