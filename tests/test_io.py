from biosiglive import load, save
import numpy as np
import pytest
import os


@pytest.mark.parametrize("rt", [True, False])
def test_save_and_load(rt):
    np.random.seed(50)
    data = {
        "data_np": np.random.rand(2, 20),
        "data_list": [5, 8],
        "data_dict": {"data_np": np.random.rand(2, 20)},
        "data_int": 1,
    }
    shape = 50 if rt else 1
    shape_np = 50 * 20 if rt else 20
    i = 0
    while i != 50:
        if rt:
            save(data, "test")
        i += 1
    if not rt:
        save(data, "test")
    data_loaded = load("test.bio")
    os.remove("test.bio")
    np.testing.assert_almost_equal(len(list(data_loaded.keys())), len(list(data.keys())))
    np.testing.assert_almost_equal(data_loaded["data_np"].shape, (2, shape_np))
    np.testing.assert_almost_equal(len(data_loaded["data_int"]), shape)
    np.testing.assert_almost_equal(len(data_loaded["data_list"]), shape)
    np.testing.assert_almost_equal(len(data_loaded["data_dict"]), shape)
