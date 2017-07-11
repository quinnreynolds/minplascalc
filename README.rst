README
======

A simple set of tools for doing calculations of thermal plasma
compositions relevant to metallurgical problems using Python 3.

*Quinn Reynolds, MINTEK Pyrometallurgy Division, July 2017*

What is this repository for?
----------------------------

-  You're here because you want to calculate LTE plasma compositions,
   thermodynamics, and physical and radiative properties. Some day this
   tool might be able to do all that, but for now you can use scraped
   content from the very nifty NIST internet databases, convert it to a
   pretty storage format, and run plasma composition and thermodynamic
   calculations using it.
-  Version 0.x alpha

How do I get set up?
--------------------

-  Getting started: Clone the repo. You'll get the module
   (minplascalc.py) and some test drivers.
-  Configuration: You should be OK with
   ``pip install -r requirements.txt``
-  Dependencies: You will need an implementation of Python 3, and
   relatively recent versions of numpy and scipy. You need pytest to run
   the tests.
-  How to run tests: simply run ``pytest`` in the root directory.

Contribution guidelines
-----------------------

-  Writing tests: It is pitch black. You are likely to be eaten by a
   grue.
-  Code review: It is pitch black. You are likely to be eaten by a grue.
-  Other guidelines: It is pitch black. You are likely to be eaten by a
   grue.

Who do I talk to?
-----------------

-  Admin, Science, Chief Pizza Officer: quinnr@mintek.co.za
-  Support, Metallurgy, Chief Caffeine Officer: markuse@mintek.co.za
-  Python, Code, Chief Beer Officer: carl.sandrock@gmail.com

References
----------

-  Thermal Plasmas: Fundamentals and Applications Volume 1. Boulos, M.I.,  
   Fauchais, P., and Pfender, E., *Plenum Press*, New York NY, 1994
-  The Mathematical Theory of Non-Uniform Gases 3\ :sup:`ed`\. Chapman, S. and 
   Cowling, T.G., *Cambridge University Press*, Cambridge, United Kingdom, 1970
-  Lowering of Ionization Potentials in Plasmas. Stewart J.C. and 
   Pyatt jr, K.D., *The Astrophysical Journal*, 144, 1966, p 1203
-  Calculation of Electrical and Thermal Conductivities of Metallurgical 
   Plasmas. Dunn, G.J. and Eagar, T.W., *Welding Research Council*, Bulletin 
   357, 1990
-  Transport Properties of Ionized Monatomic Gases. Devoto, R.S., *The Physics*
   *of Fluids*, 9(6), 1966, p 1230
-  NIST Atomic Spectra Database (ver. 5.3), [Online]. Kramida, A., Ralchenko, 
   Yu., Reader, J., and NIST ASD Team, *National Institute of Standards and* 
   *Technology*, Gaithersburg MD, http://physics.nist.gov/asd
-  NIST Chemistry WebBook, NIST Standard Reference Database Number 69. 
   Linstrom P.J. and Mallard W.G. (ed), *National Institute of Standards* 
   *and Technology*, Gaithersburg MD, http://webbook.nist.gov/chemistry/

