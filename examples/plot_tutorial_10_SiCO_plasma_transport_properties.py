r"""
Tutorial 10: Calculating transport and radiation properties of an SiO-CO plasma.
================================================================================

The most common use of minplascalc is expected to be the calculation of thermophysical properties
of plasmas in LTE as a function of elemental composition, temperature, and pressure.
For the more complex SiO-CO plasma, mixtures must be created as described in
:ref:`sphx_glr_auto_examples_plot_tutorial_08_SiCO_plasma_LTE_composition.py`
to specify the plasma species present and the relative proportions of elements.

For this tutorial we'll again look at three different SiO-CO mixtures ranging from 10% SiO to 90% SiO
by mole to show how the properties are affected by different mixture ratios.
"""  # noqa: D205

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
# Next, we create some minplascalc LTE mixture objects as before.

species = [
    "O2",
    "O2+",
    "O",
    "O+",
    "O++",
    "CO",
    "CO+",
    "C",
    "C+",
    "C++",
    "SiO",
    "SiO+",
    "Si",
    "Si+",
    "Si++",
]
x0s = [[0, 0, 0, 0, 0, 1 - sio, 0, 0, 0, 0, sio, 0, 0, 0, 0] for sio in [0.1, 0.5, 0.9]]
sico_mixtures = [mpc.mixture.lte_from_names(species, x0, 1000, 101325) for x0 in x0s]

# %%
# Set a range of temperatures to calculate the equilibrium compositions at.
# -------------------------------------------------------------------------
#
# Next, set a range of temperatures to calculate the equilibrium compositions at - in this case
# we're going from 1000 to 25000 K in 100 K steps.
# Also initialise a list to store the property values at each temperature.

temperatures = np.linspace(1000, 25000, 100)
viscosity, electrical_conductivity, thermal_conductivity, total_emission_coefficient = (
    [[] for _ in range(3)],
    [[] for _ in range(3)],
    [[] for _ in range(3)],
    [[] for _ in range(3)],
)

# %%
# Perform the composition calculations.
# --------------------------------------
#
# Now we can perform the property calculations.
# We loop over all the temperatures setting the mixture object's temperature attribute
# to the appropriate value, and calculating the plasma density by calling the LTE
# object's `calculate_viscosity()`, `calculate_electrical_conductivity()`,
# `calculate_thermal_conductivity()` and `calculate_total_emission_coefficient()` functions.
# Internally, these make calls to  `calculate_composition()` to obtain the composition
# of the plasma before the calculation of the properties.
#
# Note that execution of this calculation is fairly compute intensive and the following code
# snippet may take several tens of selectrical_conductivitys to complete.

for i, sico_mixture in enumerate(sico_mixtures):
    for T in temperatures:
        sico_mixture.T = T

        viscosity[i].append(sico_mixture.calculate_viscosity())
        electrical_conductivity[i].append(
            sico_mixture.calculate_electrical_conductivity()
        )
        total_emission_coefficient[i].append(
            sico_mixture.calculate_total_emission_coefficient()
        )
        thermal_conductivity[i].append(sico_mixture.calculate_thermal_conductivity())

# %%
# Plot the results.
# -----------------
#
# Now we can visualise the properties by plotting them against temperature, to see how they vary.

fig, axs = plt.subplots(2, 2, figsize=(10, 7), sharex=True)

labels = ["10% SiO", "50% SiO", "90% SiO"]

ax = axs[0, 0]
ax.set_title(r"$\mathregular{O_2}$ plasma viscosity")
ax.set_ylabel("$\\mathregular{\\mu [Pa.s]}$")
for visc, label in zip(viscosity, labels):
    ax.plot(temperatures, visc, label=label)

ax = axs[0, 1]
ax.set_title(r"$\mathregular{O_2}$ plasma thermal conductivity")
ax.set_ylabel("$\\mathregular{\\kappa [W/(m.K)]}$")
for thermal_cond, label in zip(thermal_conductivity, labels):
    ax.plot(temperatures, thermal_cond, label=label)

ax = axs[1, 0]
ax.set_title(r"$\mathregular{O_2}$ plasma electrical conductivity")
ax.set_xlabel("T [K]")
ax.set_ylabel("$\\mathregular{\\sigma [S/m]}$")
for elec_cond, label in zip(electrical_conductivity, labels):
    ax.plot(temperatures, elec_cond, label=label)

ax = axs[1, 1]
ax.set_title(r"$\mathregular{O_2}$ plasma emission coefficient")
ax.set_xlabel("T [K]")
ax.set_ylabel("$\\mathregular{\\epsilon_{tot} [W/(m^3.sr)]}$")
for emiss, label in zip(total_emission_coefficient, labels):
    ax.plot(temperatures, emiss, label=label)

ax.legend()

plt.tight_layout()

# %%
# Conclusion
# ----------
#
# The impact of changing the elemental composition of the plasma is again quite considerable
# and non-linear, especially for thermal conductivity and emission coefficient.
# The general trend as SiO content in the plasma increases is toward slightly lower values of
# :math:`\mu` and :math:`\kappa`, and much higher values of :math:`\epsilon_{tot}`, with
# :math:`\sigma` varying relatively little except at high temperatures.

# %%
