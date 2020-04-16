from .base import Base

class Particle(Base):
    def __init__(self, symbol, m, q, **kwargs):
        self.symbol = symbol
        self.m = m
        self.q = q
        self.__dict__.update(**kwargs)
        super(Element, self).__init__()
        pass
