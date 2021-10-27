from collections import UserList
from abc import ABC

class BaseET:
    # all the ET goodness goes in here, simpler because it's a singleton
    # optimize the b***z out of this

    def __init__(self, i=-1):
        self.i = i
        pass

    def __mul__(self, other):
        return ETS([self, other])

    def __str__(self):
        return f"[{self.i}]"

class ET(BaseET):
    pass

    def tx(self, eta):
        pass

    def rx(self, eta):
        pass

class ETS(UserList):
    # listy superpowers
    # this is essentially just a container for a list of ET instances

    def __init__(self, arg):
        super().__init__()
        if isinstance(arg, list):
            if not all([isinstance(a, ET) for a in arg]):
                raise ValueError('bad arg')
            self.data = arg

    def __str__(self):
        return " * ".join([str(e) for e in self.data])


    def __mul__(self, other):
        if isinstance(other, ET):
            return ETS([*self.data, other])
        elif instance(other, ETS):
            return ETS([*self.data, *other.data])

    def __rmul__(self, other):
        return ETS([other, self.data])

b = ET(1) * ET(2)
print(len(b))
print(b)

b = b * ET(3)
print(b)
print(len(b))

print(b[1])