import pytest
import minplascalc as mpc


def test_build_monatomic_species_json():
    # Demo of how to build an object for a monatomic species from a JSON file
    result = mpc.species_from_name('O+')

    assert len(result.energylevels) == 275
    assert result.sources is not None


def test_build_diatomic_species_json():
    # Demo of how to build an object for a diaatomic species from a JSON file
    result = mpc.species_from_name('CO')
    
    assert result.chargenumber == 0
    assert result.sources is not None
