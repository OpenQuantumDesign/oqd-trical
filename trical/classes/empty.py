"""
Module defining the Empty class.
"""

class Empty(object):
    """
    An empty class that can be loaded with attributes.
    """
    def __init__(self, **kwargs):
        super(Empty, self).__init__()
        self.__dict__.update(kwargs)
        pass

    def update(self, **kwargs):
        """
        Updates the attributes of the empty class.
        """
        self.__dict__.update(kwargs)
        pass

    pass
