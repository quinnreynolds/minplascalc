#!/usr/bin/env python3

from setuptools import setup, find_packages
from codecs import open
from os import path
import minplascalc

here = path.abspath(path.dirname(__file__))

with open(path.join(here, 'README.rst'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='minplascalc',
    version=minplascalc.__version__,
    description='A simple set of tools for doing calculations of thermal plasma '
                'compositions relevant to metallurgical problems using Python 3',
    long_description=long_description,
    url='https://github.com/quinnreynolds/minplascalc',
    author='Quinn Reynolds',
    author_email='quinnr@mintek.co.za',
    license='GPLv3',
    classifiers=[
        'Development Status :: 3 - Alpha',

        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Chemistry',
        'Topic :: Scientific/Engineering :: Physics',

        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',

        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ],
    keywords='plasma gibbs composition',
    packages=find_packages(exclude=['contrib', 'docs', 'tests']),
    install_requires=['numpy', 'scipy>=1.4'],
    extras_require={
        'dev': ['matplotlib'],
        'test': ['coverage', 'pytest', 'jupyter'],
    },
    package_data={
        'minplascalc': ['data/*/*.json'],
    },
)
