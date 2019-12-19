class Empty(object):
    def __init__(self, **kwargs):
        super(Empty, self).__init__()
        self.__dict__.update(kwargs)
        pass

    def update(self, **kwargs):
        self.__dict__.update(kwargs)
        pass

    pass
