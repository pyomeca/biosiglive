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