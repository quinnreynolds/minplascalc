import pytest

import minplascalc as mpc


@pytest.mark.parametrize(
    "test_input, expected",
    [
        ("           2 |            0.0000000            |", [2, 0.0]),
        ("           1 |            0.0196224            |", [1, 0.0196224]),
        ("           0 |            0.0281416            |", [0, 0.0281416]),
        ("   2 |          [16.82239]             |", [2, 16.82239]),
        ("  5/2 |            30.425722+x            |", [2.5, 30.425722]),
        ("  7/2 |            32.92787?              |", [3.5, 32.92787]),
        (
            "  1/2 |            33.11632?              |\n"
            "3/2 |            33.140757              |",
            [0.5, 33.11632, 1.5, 33.140757],
        ),
    ],
)
def test_nist_str(test_input, expected):
    assert expected == mpc.parsers.nist_string(test_input)


@pytest.mark.parametrize(
    "test_input, expected",
    [
        (
            [
                "           2 |            0.0000000            |",
                "           1 |            0.0196224            |",
                "           0 |            0.0281416            |",
            ],
            [[2, 0.0], [1, 0.0196224], [0, 0.0281416]],
        ),
        (
            [
                "   2 |          [16.82239]             |",
                "  5/2 |            30.425722+x            |",
                "  7/2 |            32.92787?              |",
            ],
            [[2, 16.82239], [2.5, 30.425722], [3.5, 32.92787]],
        ),
    ],
)
def test_nist_energy_levels(test_input, expected):
    assert expected == mpc.parsers.nist_energy_levels(test_input)


def test_nist_energy_levels_error():
    with pytest.raises(ValueError):
        # Missing a coma to separate the two lines.
        mpc.parsers.nist_energy_levels(
            [
                "  1/2 |            33.11632?              |\n"
                "3/2 |            33.140757              |"
            ],
        )
