import os.path

import pickle
from pathlib import Path
from ..streaming.utils import dic_merger
import pickletools
import gzip
from multiprocessing import Pool


def _load_and_compress(origin_file, target_file=None):
    data = load(origin_file)
    if not target_file:
        target_file = origin_file.split(".")[:-1][0] + ".bio.gzip"
    if Path(target_file).suffix != ".gzip":
        target_file = target_file + ".gzip"
    f = gzip.open(target_file, "wb")
    data_pick = pickletools.optimize(pickle.dumps(data, pickle.HIGHEST_PROTOCOL))
    f.write(data_pick)
    f.close()


def compress(origin_file, target_file=None, multi_proc=False):
    if target_file and len(target_file) != len(origin_file):
        raise ValueError("The number of target files must be equal to the number of origin files.")
    if multi_proc:
        pool = Pool(processes=os.cpu_count())
        pool.map(_load_and_compress, origin_file)
    else:
        for i in range(len(origin_file)):
            target_file_tmp = target_file[i] if target_file else None
            _load_and_compress(origin_file[i], target_file_tmp)


def _safe_rename_file(file_name) -> str:
    """This function checks if the file extension is an integer.

    Parameters
    ----------
    file_name : str
        The name to the file.

    Returns
    -------
    str
        new name of the file.
    """
    i = 0
    while file_name[-(i + 1)].isdigit():
        i += 1
    old_value = int(file_name[-i:]) if i > 0 else None
    if old_value:
        if file_name[-(i + 1)] == "_" or file_name[-(i + 1)] == "-":
            i += 1
        return file_name[:-i] + "_" + str(old_value + 1)
    else:
        return file_name + "_1"


def save(data_dict, data_path, add_data=False, safe=True, compress=False):
    """This function adds data to a pickle file. It not open the file, but appends the data to the end, so it's fast.

    Parameters
    ----------
    data_dict : dict
        The data to be added to the file.
    data_path : str
        The path to the file. The file must exist.
    add_data : bool, optional
        If True, the data are added to the file. If False, the file is overwritten. The default is False.
    safe : bool, optional
        If True, the data are saved in a new file. The default is False.
    compress : bool, optional
        If True, the data are compressed. The default is False.
    """
    data_path_object = Path(data_path)
    if data_path_object.suffix != ".bio":
        if data_path_object.suffix == "":
            data_path += ".bio"
        else:
            raise ValueError("The file must be a .bio file.")
    if not add_data:
        if safe and os.path.isfile(data_path):
            file_name = _safe_rename_file(data_path)
            data_path = data_path_object.parent / (file_name + data_path_object.suffix)
            print(
                f"The file {data_path_object.name} already exists. The data will be saved in {data_path.name}."
                f" To avoid this message, remove the safe option."
            )
        if not safe and os.path.isfile(data_path):
            os.remove(data_path)

    if compress:
        if add_data:
            raise ValueError("The data cannot be added to a compressed file.")
        data_path = data_path + ".gzip"
        f = gzip.open(data_path, "wb")
        data_pick = pickletools.optimize(pickle.dumps(data_dict, pickle.HIGHEST_PROTOCOL))
        f.write(data_pick)
        f.close()
    else:
        with open(data_path, "ab") as outf:
            pickle.dump(data_dict, outf, pickle.HIGHEST_PROTOCOL)


def load_old(filename, number_of_line=None, merge=True):
    """This function reads data from a pickle file to concatenate them into one dictionary.

    Parameters
    ----------
    filename : str
        The path to the file.
    number_of_line : int
        The number of lines to read. If None, all lines are read.
    merge : bool
        If True, the data are merged into one dictionary. If False, the data are returned as a list of dictionaries.

    Returns
    -------
    data : dict
        The data read from the file.

    """
    with_gzip = False

    if Path(filename).suffix == ".bio":
        with_gzip = False
    elif Path(filename).suffix == ".gzip":
        with_gzip = True
    # else:
    #     raise ValueError("The file must be a .bio or a .bio.gzip file.")
    data = None if merge else []
    limit = 2 if not number_of_line else number_of_line
    if with_gzip:
        with gzip.open(filename, "rb") as file:
            count = 0
            while count < limit:
                try:
                    data_tmp = pickle.load(file)
                    if not merge:
                        data.append(data_tmp)
                    else:
                        if not data:
                            data = data_tmp
                        else:
                            data = dic_merger(data, data_tmp)
                    if number_of_line:
                        count += 1
                    else:
                        count = 1
                except EOFError:
                    break
    else:
        with open(filename, "rb") as file:
            count = 0
            while count < limit:
                try:
                    data_tmp = pickle.load(file)
                    if not merge:
                        data.append(data_tmp)
                    else:
                        if not data:
                            data = data_tmp
                        else:
                            data = dic_merger(data, data_tmp)
                    if number_of_line:
                        count += 1
                    else:
                        count = 1
                except EOFError:
                    break
    return data


def load(filename, number_of_line=None, merge=True):
    """This function reads data from a pickle file to concatenate them into one dictionary.

    Parameters
    ----------
    filename : str
        The path to the file.
    number_of_line : int
        The number of lines to read. If None, all lines are read.
    merge : bool
        If True, the data are merged into one dictionary. If False, the data are returned as a list of dictionaries.

    Returns
    -------
    data : dict
        The data read from the file.

    """
    if not Path(filename).is_file():
        raise FileNotFoundError(f"The file {filename} does not exist.")

    with_gzip = filename.endswith(".gzip")
    data = None if merge else []
    limit = number_of_line if number_of_line else float("inf")

    try:
        return _read_all_lines(filename, limit, data, with_gzip, merge)
    except Exception as e:
        raise RuntimeError(f"An error occurred while loading the file: {e}")


def _read_all_lines(filename, limit=float("inf"), data=(), with_gzip=False, merge=True):
    file_open = gzip.open if with_gzip else open
    with file_open(filename, "rb") as file:
        count = 0
        while count < limit:
            try:
                data_tmp = pickle.load(file)
                if not merge:
                    data.append(data_tmp)
                else:
                    data = dic_merger(data, data_tmp) if data else data_tmp
                count += 1
            except Exception as e:
                if str(e) == "Ran out of input":
                    # print(f"{count} lines read from the file")
                    pass
                else:
                    print(f"An error occurred while reading the file: {e} at line {count}")
                break

    return data
