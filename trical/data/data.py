from ..classes import Empty
import json


def save_data(data_dict, filename):
    f = open(filename, "w")
    json.dump(data_dict, f)
    f.close()
    pass


def load_data(filename):
    f = open(filename, "r")
    data_dict = json.load(f)
    f.close()
    return data_dict


def save_object(object, filename):
    save_data(object.__dict__, filename)
    pass


def load_object(filename):
    return Empty(load_data(filename))
