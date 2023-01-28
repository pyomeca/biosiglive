"""
This example shows how to save data in a pickle file (with a *.bio extension) using the preimplemented function. The particularity of this function is
that the data are add to the file without reading all the file, so it is a fast way to save data in a binary file inside a loop.
You might want to do that for online data streaming in case of disfunction and to not fill an array with to mch data.
The file can then read using the load function wich will concatenate all the data in the file to return a dictionary with all the data.
Keep in mind that the load function need to read each line one after the other so if you have a lot of line it migth take a while.
The load function take the nuber of line that you want to read as argument in case of to slow oppening.
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
