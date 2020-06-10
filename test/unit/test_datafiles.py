import pytest
import minplascalc as mpc


def test_build_monatomic_species_json():
    result = mpc.species.from_name('O+')

    assert len(result.energylevels) == 275
    assert result.sources is not None


def test_build_diatomic_species_json():
    result = mpc.species.from_name('CO')
    
    assert result.chargenumber == 0
    assert result.sources is not None
