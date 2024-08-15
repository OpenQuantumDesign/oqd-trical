"""
Module containing functions for saving and loading data for TrICal.
"""
from ..classes import Base
import pickle
import numpy as np


def save_dict(data_dict, filename):
    """
    Saves a dictionary into a file.

    :param data_dict: Dictionary of interest
    :type data_dict: :obj:`dict`
    :param filename: Path of file to save the dictionary in.
    :type filename: :obj:`str`
    """
    f = open(filename, "wb")
    pickle.dump(data_dict, f)
    f.close()
    pass


def load_dict(filename):
    """
    Loads a dictionary from a file.

    :param filename: Path of file containing the dictionary to be loaded.
    :type filename: :obj:`str`
    :returns: Loaded dictionary.
    :rtype: :obj:`dict`
    """
    f = open(filename, "rb")
    data_dict = pickle.load(f)
    f.close()
    return data_dict


def save_object(object, filename):
    """
    Saves the attributes of an object, as a dictionary, into a file.

    :param data_dict: Dictionary of interest
    :type data_dict: :obj:`dict`
    :param filename: Path of file to save the attriutes of the object in.
    :type filename: :obj:`object`
    """
    save_dict(object.__dict__, filename)
    pass


def load_object(filename):
    """
    Loads an instance of the Empty class with the attributes from a file.

    :param filename: Path of file containing the attributes to be loaded.
    :type filename: :obj:`str`
    :returns: Loaded instance of the Empty class.
    :rtype: :obj:`trical.classes.empty.Empty`
    """
    return Base(**load_dict(filename))
