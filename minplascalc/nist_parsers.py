"""Downloads the couple (J, Level[eV]) from the [NIST_ASD]_ lebels data.

J is the total angular momentum number.
Level is the energy level in electron volts.


Inspired from https://github.com/exoclime/HELIOS-K/blob/master/nist_ELevels2.py
"""

import requests  # type: ignore  # To download the data from the NIST Atomic Spectra Database.

from minplascalc.units import Units

u = Units()


def get_nist_energy_levels(
    atom_symbol: str, ionization_state: str
) -> list[tuple[float, float]]:
    r"""Get the angular momentum and energy levels for a given atom and ionization state.

    Parameters
    ----------
    atom_symbol : str
        Atomic symbol, e.g. 'O' for oxygen.
    ionization_state : str
        Ionization state, as a roman string, e.g. 'I' for neutral, 'II' for singly ionized, etc.

    Returns
    -------
    list[tuple[float, float]]:
        A list of tuples, each containing the angular momentum number J and the energy level,
        in :math:`\text{J}`.

    Raises
    ------
    ValueError
        If the request to the NIST Atomic Spectra Database fails, or if the input is incorrect.
    """
    # URL for the NIST Atomic Spectra Database.
    # Button checked on https://physics.nist.gov/PhysRefData/ASD/levels_form.html:
    # - Level Units: eV,
    # - Format output: tab-delimeted,
    # - Level information: Level + J.
    nist_url = f"https://physics.nist.gov/cgi-bin/ASD/energy1.pl?de=0&spectrum={atom_symbol}+{ionization_state}&units=1&upper_limit=&parity_limit=both&conf_limit=All&conf_limit_begin=&conf_limit_end=&term_limit=All&term_limit_begin=&term_limit_end=&J_limit=&format=3&output=0&page_size=15&multiplet_ordered=0&level_out=on&j_out=on&temp=&submit=Retrieve+Data"

    # Download the data from the NIST Atomic Spectra Database.
    with requests.Session() as s:
        download = s.get(nist_url)
        data = download.content.decode("utf-8")

        # If the data contains the string "<FONT COLOR=red>", it probably means that there is a problem
        # with the input.
        check = data.find("<FONT COLOR=red>")
        if check > -1:
            raise ValueError(
                f"Please check the atomic number ({atom_symbol}) and ionization state ({ionization_state}).\n"
                "`atom_symbol` should be a string representing the atomic symbol (e.g. 'O' for oxygen).\n"
                "`ionization_state` should be a string with a roman number representing the ionization state"
                " (e.g. 'I' for neutral)."
            )

        # Remove all double quotes from the data.
        clean_data = data.replace('"', "")

    # header = clean_data.split("\n")[0]

    energy_levels: list[tuple[float, float]] = []

    for line in clean_data.split("\n")[1:]:
        if not line:  # Skip empty lines.
            continue

        J, prefix, energy_level, suffix, _ = line.split("\t")

        if J == "---":  # Ionisation state (probably).
            continue

        if "/" in J:
            numerator, denominator = J.split("/")  # e.g. 3/2
            J = float(numerator) / float(denominator)

        if not energy_level:
            # Skip empty lines.
            continue

        if J == "":
            print("Warning: J is empty. Skipping this line.")
            continue

        energy_levels.append((float(J), float(energy_level) * u.eV_to_J))

    return energy_levels


def get_nist_spectral_lines(
    atom_symbol: str, ionization_state: str
) -> list[tuple[float, float, float]]:
    r"""Get the spectral lines for a given atom and ionization state.

    Lines included are the observed one, as well as those of Ritz.

    Parameters
    ----------
    atom_symbol : str
        Atomic symbol, e.g. 'O' for oxygen.
    ionization_state : str
        Ionization state, as a roman string, e.g. 'I' for neutral, 'II' for singly ionized, etc.

    Returns
    -------
    list[tuple[float, float, float]]:
        A list of tuples, each containing:

        - the wavelength :math:`\lambda`, in :math:`\text{m}`,
        - the transition strength :math:`g_k A_{ki}`, in :math:`\text{Hz}`, and
        - the line energy :math:`E_k`, in :math:`J`.

    Raises
    ------
    ValueError
        If the request to the NIST Atomic Spectra Database fails, or if the input is incorrect.

    Notes
    -----
    See https://physics.nist.gov/PhysRefData/ASD/Html/lineshelp.html#LINES_INCLUDE_RITZ
    """
    # URL for the NIST Atomic Spectra Database.
    # Button checked on https://physics.nist.gov/PhysRefData/ASD/lines_form.html
    # - Format output: tab-delimeted,
    # - Energy level units: eV,
    # - Lines: Only with transition probabilities,
    # - Remove JavaScript,
    # - Wavelength data: Observed + Ritz,
    # - Wavelength in: Vacuum (< 200 nm)   Air (200 - 2,000 nm)   Vacuum (> 2,000 nm),
    # - Transition strength: gk Aki,
    # - Transition Type: 	Allowed (E1),
    # - Level information: Energies.
    nist_url = f"https://physics.nist.gov/cgi-bin/ASD/lines1.pl?spectra={atom_symbol}+{ionization_state}&output_type=0&low_w=&upp_w=&unit=1&de=0&plot_out=0&I_scale_type=1&format=3&line_out=1&remove_js=on&en_unit=1&output=0&page_size=15&show_obs_wl=1&show_calc_wl=1&order_out=0&max_low_enrg=&show_av=2&max_upp_enrg=&tsb_value=0&min_str=&A_out=1&max_str=&allowed_out=1&min_accur=&min_intens=&enrg_out=on&submit=Retrieve+Data"

    # Download the data from the NIST Atomic Spectra Database.
    with requests.Session() as s:
        download = s.get(nist_url)
        data = download.content.decode("utf-8")

        # If the data contains the string "<FONT COLOR=red>", it probably means that there is a problem
        # with the input.
        check = data.find("<FONT COLOR=red>")
        if check > -1:
            raise ValueError(
                f"Please check the atomic number ({atom_symbol}) and ionization state ({ionization_state}).\n"
                "`atom_symbol` should be a string representing the atomic symbol (e.g. 'O' for oxygen).\n"
                "`ionization_state` should be a string with a roman number representing the ionization state"
                " (e.g. 'I' for neutral)."
            )

        # Remove all double quotes from the data.
        clean_data = data.replace('"', "")

    spectral_lines: list[tuple[float, float, float]] = []

    for line in clean_data.split("\n")[1:]:
        if line.startswith("obs_wl"):
            continue
        if not line:  # Skip empty lines.
            continue

        # Split the line into its components.
        obs_wl_vac_nm, ritz_wl_vac_nam, gA_Hz, Acc, Ei_eV, Ek_eV, Type, _ = line.split(
            "\t"
        )

        if not obs_wl_vac_nm:
            # If the observed wavelength is not available, use the Ritz wavelength instead.
            obs_wl_vac_nm = ritz_wl_vac_nam

        # Remove all +x?[]() characters.
        Ek_eV = Ek_eV.replace("[", "").replace("]", "")
        Ek_eV = Ek_eV.replace("(", "").replace(")", "")
        Ek_eV = Ek_eV.replace("?", "").replace("+", "").replace("x", "")

        spectral_lines.append(
            (
                float(obs_wl_vac_nm) * 1e-9,
                float(gA_Hz),
                float(Ek_eV)
                * u.eV_to_J,  # Taking Ek for emission line. (Ei for absorption line.)
            )
        )

    return spectral_lines


if __name__ == "__main__":
    Z = "O"
    I = "II"
    energy_levels = get_nist_energy_levels(Z, I)
    spectral_lines = get_nist_spectral_lines(Z, I)

    # Save them in numpy format.
    import numpy as np

    np.save(f"energy_levels_{Z}_{I}.npy", energy_levels)
    np.save(f"spectral_lines_{Z}_{I}.npy", spectral_lines)

    # Open these files with
    # energy_levels = np.load(f"energy_levels_{Z}_{I}.npy")
    # spectral_lines = np.load(f"spectral_lines_{Z}_{I}.npy")
