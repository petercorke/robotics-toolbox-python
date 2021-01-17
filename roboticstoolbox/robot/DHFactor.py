from roboticstoolbox import ETS
import re
import sympy
import numpy as np
from spatialmath import base

pi2 = base.pi() / 2
deg = base.pi() / sympy.Integer('180')

class DHFactor(ETS):

    def __init__(self, s):
        et_re = re.compile(r"([RT][xyz])\(([^)]*)\)")

        super().__init__()
        # self.data = []

        for axis, eta in et_re.findall(s):
            print(axis,eta)
            if eta[0] == 'q':
                eta = None
                unit = None
            else:
                # eta can be given as a variable or a number
                try:
                    # first attempt to create symbolic number
                    eta = sympy.Number(eta)
                except:
                    # failing that, a symbolic variable
                    eta = sympy.symbols(eta)
                if axis[0] == 'R':
                    # convert to degrees, assumed unit in input string
                    eta = sympy.simplify(eta * deg)

            if axis == 'Rx':
                e = ETS.rx(eta)
            elif axis == 'Ry':
                e = ETS.ry(eta)
            elif axis == 'Rz':
                e = ETS.rz(eta)
            elif axis == 'Tx':
                e = ETS.tx(eta)
            elif axis == 'Ty':
                e = ETS.ty(eta)
            elif axis == 'Tz':
                e = ETS.tz(eta)

            self.data.append(e.data[0])

    def simplify(self):

        self.merge()
        self.swap(convention)
        self.merge()
        self.float_right()
        self.merge()
        self.substitute_to_z()
        self.merge()
        for i in range(nloops):
            nchanges = 0

            nchanges += self.merge()
            nchanges += self.swap(convention)
            nchanges += self.merge()
            nchanges += self.eliminate_y()
            nchanges += self.merge()

            if nchanges == 0:
                # try this then
                nchanges += self.substitute_to_z2()
            if nchanges == 0:
                break
        else:
            # too many iterations
            print('too many iterations')

    def substitute_to_z(self):
        # substitute all non Z joint transforms according to rules
        nchanges = 0
        out = ETS()
        for e in self:
            if e.isjoint:
                out *= e
            else:
                # do a substitution
                if e.axis == 'Rx':
                    new = ETS.ry(pi2) * ETS.rz(e.eta) * ETS.ry(-pi2)
                elif e.axis == 'Ry':
                    new = ETS.rx(-pi2) * ETS.rz(e.eta) * ETS.rx(pi2)
                elif e.axis == 'tx':
                    new = ETS.ry(pi2) * ETS.tz(e.eta) * ETS.ry(-pi2)
                elif e.axis == 'ty':
                    new = ETS.rx(-pi2) * ETS.tz(e.eta) * ETS.rx(pi2)
                else:
                    out *= e
                    continue
                out *= new
                nchanges += 1

        self.data = out.data
        return nchanges

    def merge(self):

        def canmerge(prev, this):
            return prev.axis == this.axis and not (prev.isjoint and this.isjoint)

        out = ETS()
        while len(self.data) > 0:
            this = self.pop(0)

            if len(self.data) > 0:
                next = self[0]

                if canmerge(this, next):
                    new = DHFactor.add(this, next)
                    out *= new
                    self.pop(0)  # remove next from the queue
                    print(f"Merging {this * next} to {new}")
                else:
                    out *= this

        self.data = out.data
        # remove zeros

    @staticmethod
    def add(this, that):
        if this.isjoint and not that.isjoint:
            out = ETS(this)
            if out.eta is None:
                out.eta = that.eta
            else:
                out.eta += that.eta
        elif not this.isjoint and that.isjoint:
            out = ETS(that)
            if out.eta is None:
                out.eta = this.eta
            else:
                out.eta += this.eta
        else:
            raise ValueError('both ET cannot be joint variables')
        return out

    def eliminate_y(self):

        nchanges = 0
        out = ETS()
        jointyet = False

        def eliminate(prev, this):
            if this.isjoint or prev.isjoint:
                return None

            new = None
            if prev.axis == 'Rx' and this.axis == 'ty':  # RX.TY -> TZ.RX
                new = ETS.tx(prev.eta) * prev
            elif prev.axis == 'Rx' and this.axis == 'tz':  # RX.TZ -> TY.RX
                new = ETS.ty(-prev.eta) * prev
            elif prev.axis == 'Ry' and this.axis == 'tz':  # RY.TX-> TZ.RY
                new = ETS.tz(-prev.eta) * prev
            elif prev.axis == 'Ry' and this.axis == 'tz':  # RY.TZ-> TX.RY
                new = ETS.tx(prev.eta) * prev

            elif prev.axis == 'ty' and this.axis == 'Rx':  # TY.RX -> RX.TZ
                new = this * ETS.tz(-this.eta)
            elif prev.axis == 'tx' and this.axis == 'Rz':  # TX.RZ -> RZ.TY
                new = this * ETS.tz(this.eta)
            elif prev.axis == 'Ry' and this.axis == 'Rx':  # RY(Q).RX -> RX.RZ(-Q)
                new = this * ETS.Rz(-prev.eta)
            elif prev.axis == 'Rx' and this.axis == 'Ry':  # RX.RY -> RZ.RX
                new = ETS.Rz(this.eta) * prev
            elif prev.axis == 'Rz' and this.axis == 'Rx':  # RZ.RX -> RX.RY
                new = this * ETS.Ry(this.eta)
            return new

        for i in range(len(self)):
            this = self[i]
            jointyet = this.isjoint
            if i == 0 or not jointyet:
                continue

            prev = self[i-1]

            new = eliminate(prev, this)
            if new is not None:
                self[i-1:i] = new
                nchanges += 1

        return nchanges

    def __str__(self, q=None):
        """
        Pretty prints the ETS

        :param q: control how joint variables are displayed
        :type q: str
        :return: Pretty printed ETS
        :rtype: str

        ``q`` controls how the joint variables are displayed:

        - None, format depends on number of joint variables
            - one, display joint variable as q
            - more, display joint variables as q0, q1, ...
            - if a joint index was provided, use this value
        - "", display all joint variables as empty parentheses ``()``
        - "θ", display all joint variables as ``(θ)``
        - format string with passed joint variables ``(j, j+1)``, so "θ{0}"
          would display joint variables as θ0, θ1, ... while "θ{1}" would
          display joint variables as θ1, θ2, ...  ``j`` is either the joint
          index, if provided, otherwise a sequential value.

        Example:

        .. runblock:: pycon

            >>> from roboticstoolbox import ETS
            >>> e = ETS.rz() * ETS.tx(1) * ETS.rz()
            >>> print(e[:2])
            >>> print(e)
            >>> print(e.__str__(""))
            >>> print(e.__str__("θ{0}"))  # numbering from 0
            >>> print(e.__str__("θ{1}"))  # numbering from 1
            >>> # explicit joint indices
            >>> e = ETS.rz(j=3) * ETS.tx(1) * ETS.rz(j=4)
            >>> print(e)
            >>> print(e.__str__("θ{0}"))

        .. note:: Angular parameters are converted to degrees, except if they
            are symbolic.

        Example:

        .. runblock:: pycon

            >>> from roboticstoolbox import ETS
            >>> from spatialmath.base import symbol
            >>> theta, d = symbol('theta, d')
            >>> e = ETS.rx(theta) * ETS.tx(2) * ETS.rx(45, 'deg') * \
            >>>     ETS.ry(0.2) * ETS.ty(d)
            >>> str(e)

        :SymPy: supported
        """

        # override this class from ETS
        es = []
        j = 0
        c = 0

        if q is None:
            if len(self.joints()) > 1:
                q = "q{0}"
            else:
                q = "q"

        # For et in the object, display it, data comes from properties
        # which come from the named tuple
        for et in self:

            s = f"{et.axis}("
            if et.isjoint:
                if q is not None:
                    if et.jindex is None:
                        _j = j
                    else:
                        _j = et.jindex
                    qvar = q.format(_j, _j+1) # lgtm [py/str-format/surplus-argument]  # noqa
                else:
                    qvar = ""
                if et.isflip:
                    s += f"-{qvar}"
                else:
                    s += f"{qvar}"
                j += 1

            if et.eta is not None:
                if et.isrevolute:
                    if base.issymbol(et.eta):
                        if s[-1] == "(":
                            s += f"{et.eta}"
                        else:
                            # adding to a previous value
                            if str(et.eta).startswith('-'):
                                s += f"{et.eta}"
                            else:
                                s += f"+{et.eta}"
                    else:
                        s += f"{et.eta * 180 / np.pi:.4g}°"

                elif et.isprismatic:
                    s += f"{et.eta}"

                elif et.isconstant:
                    s += f"C{c}"
                    c += 1
            s += ")"

            es.append(s)

        return " * ".join(es)
            
if __name__ == "__main__":  # pragram: no cover
    s = 'Rz(45) Tz(L1) Rz(q1) Ry(q2) Ty(L2) Tz(L3) Ry(q3) Tx(L4) Ty(L5) Tz(L6) Rz(q4) Ry(q5) Rz(q6)'

    ets = DHFactor(s)
    print(ets)
    ets.substitute_to_z()
    print(ets)
    ets.merge()
    print(ets)
