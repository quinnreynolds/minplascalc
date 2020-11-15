#!/usr/bin/env python3

from setuptools import setup, find_packages
#import codecs
import os

here = os.path.abspath(os.path.dirname(__file__))

def read(rel_path):
    with open(os.path.join(here, rel_path), 'r') as fp:
        return fp.read()

def get_version(rel_path):
    for line in read(rel_path).splitlines():
        if line.startswith('__version__'):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    else:
        raise RuntimeError('Unable to find version string.')

with open(os.path.join(here, 'README.rst'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='minplascalc',
    version=get_version('minplascalc/__init__.py'),
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
    install_requires=['numpy','scipy>=1.4'],
    extras_require={
        'dev': ['matplotlib'],
        'test': ['coverage','pytest','jupyter'],
    },
    package_data={
        'minplascalc': ['data/*/*.json'],
    },
)
