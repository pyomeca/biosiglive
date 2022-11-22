import scipy.io as sio
from biosiglive.io.save_data import read_data
import numpy as np


def get_offline_data(data_type: list, path: str = None):
    if path:
        data = get_data_from_file(data_type, path)
    else:
        data = get_random_data(data_type)
    pass


def get_random_data(data_type: list):
    data_dic = {}
    for data in data_type:
        data_dic[data]
    pass


def get_data_from_file(data_type: list, file_path: str):
    try:
        data = sio.loadmat(file_path)
    except TypeError:
        data = read_data(file_path)

    return data


def adjust_dim_data(data: np.ndarray, data_type: list, data_shape: list):
    return data


def dic_merger(dic_to_merge, new_dic=None):
    if not new_dic:
        new_dic = dic_to_merge
    else:
        for key in dic_to_merge.keys():
            if isinstance(dic_to_merge[key], dict):
                new_dic[key] = dic_merger(dic_to_merge[key], new_dic[key])
            elif isinstance(dic_to_merge[key], list):
                new_dic[key] = dic_to_merge[key] + new_dic[key]
            elif isinstance(dic_to_merge[key], np.ndarray):
                new_dic[key] = np.append(dic_to_merge[key], new_dic[key],  axis=0)
            elif isinstance(dic_to_merge[key], int):
                if isinstance(new_dic[key], int):
                    new_dic[key] = [new_dic[key]]
                new_dic[key] = [new_dic[key]] + [dic_to_merge[key]]
    return new_dic
