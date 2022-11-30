import os

from biosiglive import save, load
import numpy as np

if __name__ == "__main__":
    data = {
        "data_np": np.random.rand(2, 20),
        "data_list": [5, 8],
        "data_dict": {"data_np": np.random.rand(2, 20)},
        "data_int": 1,
    }
    i = 0
    while i != 50:
        save(data, "test")
        i += 1
    data_loaded = load("test.bio")
    os.remove("test.bio")
