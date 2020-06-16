#!/usr/bin/env python

import imp
import sys

from setuptools import setup, find_packages

if sys.version_info < (2, 7):
    sys.exit("Sorry, Python < 2.7 is not supported")

VERSION = "0.1.0"

setup(
    name="naq-graphs",
    author="Alexis Arnaudon",
    author_email="alexis.arnaudon@epfl.ch",
    version=VERSION,
    description="",
    install_requires=[
        "scipy>=1.2.0",
        "networkx",
        "matplotlib",
        "scikit-image",
        "tables",
    ],
    packages=find_packages(),
)
