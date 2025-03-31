r"""
Tutorial 07: Calculating transport and radiation properties of an :math:`O_2` plasma.
=====================================================================================

The most common use of minplascalc is the calculation of thermophysical
properties of plasmas in LTE. In this example we'll look at the transport and
radiation properties viscosity :math:`\mu`, electrical conductivity
:math:`\sigma`, thermal conductivity :math:`\kappa`, and total volumetric
emission coefficient :math:`\epsilon_{tot}`.

Again, we will use here the relatively simple case of a pure oxygen plasma.
As in :ref:`sphx_glr_auto_examples_plot_tutorial_05_oygen_plasma_lte_composition.py`,
a Mixture object must be created by the user to specify the plasma species
present and the relative proportions of elements.
We'll use a system identical to the previous example.
"""  # noqa: D205, E501

# %%
# Import the required libraries.
# ------------------------------
#
# We start by importing the modules we need:
#
# - matplotlib for drawing graphs,
# - numpy for array functions,
# - and of course minplascalc.

import matplotlib.pyplot as plt
import numpy as np

import minplascalc as mpc

# %%
# Create mixture object for the species we're interested in.
# ----------------------------------------------------------
#
# Next, we create a minplascalc LTE mixture object. Here we use a helper
# function in minplascalc which creates the object directly from a list of
# the species names.

oxygen_mixture = mpc.mixture.lte_from_names(
    ["O2", "O2+", "O", "O-", "O+", "O++"], [1, 0, 0, 0, 0, 0], 1000, 101325
)
# %%
# Set a range of temperatures to calculate the equilibrium compositions at.
# -------------------------------------------------------------------------
#
# Next, set a range of temperatures to calculate the equilibrium compositions
# at - in this case we're going from 1000 to 25000 K in 100 K steps.
# Also initialise a list to store the property values at each temperature.

temperatures = np.linspace(1000, 25000, 100)
viscosity = []
electrical_conductivity = []
thermal_conductivity = []
total_emission_coefficient = []

# %%
# Perform the composition calculations.
# --------------------------------------
#
# Now we can perform the property calculations.
# We loop over all the temperatures setting the mixture object's temperature
# attribute to the appropriate value, and calculating the plasma density by
# calling the LTE object's `calculate_viscosity()`,
# `calculate_electrical_conductivity()`, `calculate_thermal_conductivity()`
# and `calculate_total_emission_coefficient()` functions.
# Internally, these make calls to  `calculate_composition()` to obtain the
# composition of the plasma before the calculation of the properties.
#
# Note that execution of this calculation is fairly compute intensive and the
# following code snippet may take several tens of seconds to
# complete.

for T in temperatures:
    oxygen_mixture.T = T
    viscosity.append(oxygen_mixture.calculate_viscosity())
    electrical_conductivity.append(
        oxygen_mixture.calculate_electrical_conductivity()
    )
    total_emission_coefficient.append(
        oxygen_mixture.calculate_total_emission_coefficient()
    )
    thermal_conductivity.append(
        oxygen_mixture.calculate_thermal_conductivity()
    )

# %%
# Plot the results.
# -----------------
#
# Now we can visualise the properties by plotting them against temperature, to
# see how they vary.

fig, axs = plt.subplots(2, 2, figsize=(10, 7), sharex=True)

ax = axs[0, 0]
ax.set_title(r"$\mathregular{O_2}$ plasma viscosity")
ax.set_ylabel("$\\mathregular{\\mu [Pa.s]}$")
ax.plot(temperatures, viscosity, "k", label="minplascalc")
ax.legend()

ax = axs[0, 1]
ax.set_title(r"$\mathregular{O_2}$ plasma thermal conductivity")
ax.set_ylabel("$\\mathregular{\\kappa [W/(m.K)]}$")
ax.plot(temperatures, thermal_conductivity, "k", label="minplascalc")
ax.legend()

ax = axs[1, 0]
ax.set_title(r"$\mathregular{O_2}$ plasma electrical conductivity")
ax.set_xlabel("T [K]")
ax.set_ylabel("$\\mathregular{\\sigma [S/m]}$")
ax.plot(temperatures, electrical_conductivity, "k", label="minplascalc")
ax.legend()

ax = axs[1, 1]
ax.set_title(r"$\mathregular{O_2}$ plasma emission coefficient")
ax.set_xlabel("T [K]")
ax.set_ylabel("$\\mathregular{\\epsilon_{tot} [W/(m^3.sr)]}$")
ax.plot(temperatures, total_emission_coefficient, "k", label="minplascalc")
ax.legend()

plt.tight_layout()

# %%
# Conclusion
# ----------
#
# The results obtained using minplascalc are reasonably comparable to other
# data for oxygen plasmas in literature, for example [Boulos2023]_.
# There are some differences, particularly in the thermal conductivity curve,
# and these are likely to result from the use of different collision integral
# calculations and other base data.

# %%
