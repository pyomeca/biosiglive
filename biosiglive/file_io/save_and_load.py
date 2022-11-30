"""
This file is part of biosiglive. it contains the functions to save and read data.
"""
import pickle
import numpy as np
from pathlib import Path


def save(data_dict, data_path):
    """
    This function adds data to a pickle file. It not open the file, but appends the data to the file.

    Parameters
    ----------
    data_dict : dict
        The data to be added to the file.
    data_path : str
        The path to the file. The file must exist.
    """
    if Path(data_path).suffix != ".bio":
        if Path(data_path).suffix == "":
            data_path += ".bio"
        else:
            raise ValueError("The file must be a .bio file.")
    with open(data_path, "ab") as outf:
        pickle.dump(data_dict, outf, pickle.HIGHEST_PROTOCOL)


# TODO add dict merger
def load(filename, number_of_line=None):
    """
    This function reads data from a pickle file.
    Parameters
    ----------
    filename : str
        The path to the file.
    number_of_line : int
        The number of lines to read. If None, all lines are read.

    Returns
    -------
    data : dict
        The data read from the file.
    """
    if Path(filename).suffix != ".bio":
        raise ValueError("The file must be a .bio file.")
    data = {}
    limit = 2 if not number_of_line else number_of_line
    with open(filename, "rb") as file:
        count = 0
        while count < limit:
            try:
                data_tmp = pickle.load(file)
                for key in data_tmp.keys():
                    if key in data.keys():
                        if isinstance(data[key], list) is True:
                            data[key].append(data_tmp[key])
                        else:
                            data[key] = np.append(data[key], data_tmp[key], axis=len(data[key].shape) - 1)
                    else:
                        if isinstance(data_tmp[key], (int, float, str, dict)) is True:
                            data[key] = [data_tmp[key]]
                        elif isinstance(data_tmp[key], list) is True:
                            data[key] = [data_tmp[key]]
                        else:
                            data[key] = data_tmp[key]
                if number_of_line:
                    count += 1
                else:
                    count = 1
            except EOFError:
                break
    return data
