import pickle

import numpy as np

from minplascalc.nist_parsers import get_nist_energy_levels, get_nist_spectral_lines
from minplascalc.utils import get_path_to_data


def test_compare_energy_levels():
    with open(get_path_to_data("demo", "nist", "nist_Oplus_levels"), "rb") as f:
        elevels = pickle.load(f)
    energy_levels = np.load(get_path_to_data("demo", "nist", "energy_levels_O_II.npy"))

    assert len(elevels) == len(energy_levels)

    assert np.isclose(elevels, energy_levels).all()


def test_compare_spectral_lines():
    with open(get_path_to_data("demo", "nist", "nist_Oplus_emissionlines"), "rb") as f:
        elines = pickle.load(f)
    spectral_lines = np.load(
        get_path_to_data("demo", "nist", "spectral_lines_O_II.npy")
    )

    assert len(elines) == len(spectral_lines)

    assert np.isclose(elines, spectral_lines).all()


def test_get_nist_energy_levels():
    Z = "O"
    I = "II"
    energy_levels_downloaded = get_nist_energy_levels(Z, I)

    energy_levels = np.load(get_path_to_data("demo", "nist", "energy_levels_O_II.npy"))

    assert len(energy_levels_downloaded) == len(energy_levels)
    assert np.isclose(energy_levels_downloaded, energy_levels).all()


def test_get_nist_spectral_lines():
    Z = "O"
    I = "II"
    spectral_lines_downloaded = get_nist_spectral_lines(Z, I)

    spectral_lines = np.load(
        get_path_to_data("demo", "nist", "spectral_lines_O_II.npy")
    )

    assert len(spectral_lines_downloaded) == len(spectral_lines)
    assert np.isclose(spectral_lines_downloaded, spectral_lines).all()
