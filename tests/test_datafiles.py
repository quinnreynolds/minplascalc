import minplascalc as mpc


def test_build_monatomic_species_json():
    result = mpc.species.from_name("O+")

    assert result.stoichiometry == {"O": 1}
    assert result.charge_number == 1
    assert result.sources is not None


def test_build_diatomic_species_json():
    result = mpc.species.from_name("CO")

    assert result.stoichiometry == {"C": 1, "O": 1}
    assert result.charge_number == 0
    assert result.sources is not None
