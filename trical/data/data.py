from ..classes import Empty
import pickle
import numpy as np


def save_dict(data_dict, filename):
    f = open(filename, "wb")
    pickle.dump(data_dict, f)
    f.close()
    pass


def load_dict(filename):
    f = open(filename, "rb")
    data_dict = pickle.load(f)
    f.close()
    return data_dict


def save_object(object, filename):
    save_dict(object.__dict__, filename)
    pass


def load_object(filename):
    return Empty(**load_dict(filename))
