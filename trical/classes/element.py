from .base import Base

class Element(Base):
    def __init__(self, symbol, m, q, Omega_bar, A):
        self.symbol = symbol
        self.m = m
        self.q = q
        self.Omega_bar = Omega_bar
        self.A = A
        super(Element, self).__init__()
        pass
