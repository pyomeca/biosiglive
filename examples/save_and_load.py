"""
This example shows how to save data in a pickle file (with a *.bio extension) using the pre-implemented function. The special feature of this function is that the data is added to the file without reading the whole file, so it is a fast way to save data to a binary file inside a loop.
So it's a quick way to save data to a binary file inside a loop. You might want to do this for the online data stream in case of a malfunction and to not fill a table with too much data.
The file can then be read using the load function which concatenates all the data in the file and returns a dictionary containing all the data.
Keep in mind that the load function has to read each row one after the other, so if you have a lot of rows, it may take a while.
The load function takes the number of lines you want to read as an argument in case it opens too slowly.
"""

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
