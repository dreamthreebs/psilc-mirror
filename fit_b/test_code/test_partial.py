import numpy as np
from functools import partial
from iminuit import Minuit


class Person:
    def __init__(self, a):
        self.a = a

    def mul(self, x, y):
        return x * y

    def main(self):
        params = (0,0)
        obj = Minuit(self.mul, name=("x", "y"), *params)


obj = Person(a=3)
obj.main()
