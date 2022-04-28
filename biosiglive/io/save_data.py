import pickle
import numpy as np


def add_data_to_pickle(data_dict, data_path):
    with open(data_path, "ab") as outf:
        pickle.dump(data_dict, outf, pickle.HIGHEST_PROTOCOL)


def read_data(filename, number_of_line=None):
    data = {}
    limit = 2 if not number_of_line else number_of_line
    with open(filename, "rb") as file:
        count = 0
        while count < limit:
            try:
                data_tmp = pickle.load(file)
                for key in data_tmp.keys():
                    try:
                        if isinstance(data[key], list) is True:
                            data[key].append(data_tmp[key])
                        else:
                            data[key] = np.append(data[key], data_tmp[key], axis=len(data[key].shape) - 1)
                    except:
                        if isinstance(data_tmp[key], (int, float, str, dict)) is True:
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