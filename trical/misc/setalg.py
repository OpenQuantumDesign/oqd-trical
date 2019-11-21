"""
Relevant set algebra functions for TrICal
"""
import numpy as np


def intersection(a1, a2):
    """
    Takes the intersection between two arrays

    Args:
        a1 (1-D array): First array
        a2 (1-D array): Second array

    Returns:
        1-D array: Intersection between the two given arrays
    """
    return [i for i in a1 if i in a2]
