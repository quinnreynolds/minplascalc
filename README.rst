README
======

.. image:: https://travis-ci.org/kittychunk/minplascalc.svg?branch=develop
    :target: https://travis-ci.org/kittychunk/minplascalc

.. image:: https://readthedocs.org/projects/minplascalc/badge/?version=latest
    :target: https://minplascalc.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status

A simple set of tools in Python 3 for doing calculations of thermal plasma
compositions relevant to metallurgical problems.

*Quinn Reynolds, MINTEK Pyrometallurgy Division, 2018-present*

What is this repository for?
----------------------------

-  You're here because you want to calculate plasma compositions,
   thermodynamics, and physical and radiative properties. This package will
   do most of those things for LTE plasmas, to varying degrees of fidelity. 
-  Version 0.x alpha

How do I get set up?
--------------------

-  Getting started: Clone the repo. You'll get the package and some 
   test drivers.
-  Configuration: You should be OK with
   ``pip install -r requirements.txt``
-  Dependencies: You will need an implementation of Python 3, and
   relatively recent versions of numpy and scipy. You need pytest to run
   the tests.
-  How to run tests: simply run ``pytest`` in the root directory.

Contribution guidelines
-----------------------

-  Writing tests: TBC
-  Code review: TBC
-  Other guidelines: TBC

Who do I talk to?
-----------------

-  quinnr@mintek.co.za

References
----------

-  MI Boulos, P Fauchais, and E Pfender. Thermal Plasmas: Fundamentals and 
   Applications Volume 1, *Plenum Press*, New York NY, 1994
-  S Chapman and TG Cowling. The Mathematical Theory of Non-Uniform Gases 
   3\ :sup:`ed`\, *Cambridge University Press*, Cambridge, United Kingdom, 
   1970
-  JC Stewart and KD Pyatt Jr. Lowering of Ionization Potentials in Plasmas, 
   *The Astrophysical Journal*, 144, 1966, p 1203
-  GJ Dunn and TW Eagar. Calculation of Electrical and Thermal 
   Conductivities of Metallurgical Plasmas, 
   *Welding Research Council*, Bulletin 357, 1990
-  RS Devoto. Transport Properties of Ionized Monatomic Gases, 
   *The Physics of Fluids*, 9(6), 1966, p 1230
-  A Kramida, Y Ralchenko, J Reader, and NIST ASD Team. NIST Atomic Spectra 
   Database (ver. 5.3) [Online], 
   *National Institute of Standards and Technology*, Gaithersburg MD, 
   http://physics.nist.gov/asd
-  Linstrom, P.J. and Mallard, W.G. (ed). NIST Chemistry WebBook, NIST 
   Standard Reference Database Number 69. 
   *National Institute of Standards and Technology*, Gaithersburg MD, 
   http://webbook.nist.gov/chemistry/
