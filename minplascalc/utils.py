"""Library useful functions to retrieve data."""

from pathlib import Path

ROOT_FOLDER_PATH = Path(__file__).parent


def get_root() -> Path:
    """Return the full path of the minplascalc folder.

    Used not to worry about the project architecture.

    Returns
    -------
    Path
        the abspath to root folder (should end with 'minplascalc')

    Examples
    --------
    >>> from minplascalc.utils import get_root
    >>> path = get_root() / "data"
    """
    return ROOT_FOLDER_PATH


def get_path_to_data(*paths: str, force_return: bool = False) -> Path:
    """Return the absolute path to the data folder, or file inside.

    Parameters
    ----------
    *paths : str
        You can add a path to precise the folder inside.
    force_return : bool, optional
        If True, return path even if does not exists, by default False.

    Returns
    -------
    Path
        The abspath to the data (or file).

    Examples
    --------
    >>> from minplascalc.utils import get_path_to_data
    >>> path = get_path_to_data()
    >>> path = get_path_to_data("demo", "nist", "nist_Oplus_emission_lines")

    Raises
    ------
    FileNotFoundError
        If the file or folder is not found.
    """
    path_to_data_folder = get_root().joinpath("data", *paths)

    if not (path_to_data_folder.exists() or force_return):
        raise FileNotFoundError

    return path_to_data_folder.resolve()
