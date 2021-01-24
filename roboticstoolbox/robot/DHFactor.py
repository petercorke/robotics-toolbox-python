from roboticstoolbox import ETS
import re
import sympy
import numpy as np
from spatialmath import base

pi2 = base.pi() / 2
deg = base.pi() / sympy.Integer('180')

# PROGRESS
# subs2z does a bad thing in first phase, 2 subs it shouldnt make
class DHFactor(ETS):

    def __init__(self, axis=None, eta=None, **kwargs):

        super().__init__(axis=axis, eta=eta, **kwargs)

    @classmethod
    def parse(cls, s):
        et_re = re.compile(r"([RT][xyz])\(([^)]*)\)")

        # self.data = []

        jointnum = 0
        ets = DHFactor()

        for axis, eta in et_re.findall(s):
            if eta[0] == 'q':
                eta = None
                unit = None
                j = jointnum
                jointnum += 1
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
                    eta = eta * deg
                j = None

            if axis == 'Rx':
                e = DHFactor.rx(eta, j=j)
            elif axis == 'Ry':
                e = DHFactor.ry(eta, j=j)
            elif axis == 'Rz':
                e = DHFactor.rz(eta, j=j)
            elif axis == 'Tx':
                e = DHFactor.tx(eta, j=j)
            elif axis == 'Ty':
                e = DHFactor.ty(eta, j=j)
            elif axis == 'Tz':
                e = DHFactor.tz(eta, j=j)

            ets *= e
        
        return cls(ets)

    # ---------------------------------------------------------------------- #

    def simplify(self, mdh=False):

        self.merge()
        self.swap(mdh)
        self.merge()
        self.float_right()
        self.merge()
        # self.substitute_to_z()
        self.merge()
        print('inital merge and swap\n  ', self)
        print('main loop')
        for i in range(10):
            nchanges = 0

            nchanges += self.merge()
            nchanges += self.swap(mdh)
            nchanges += self.merge()
            print(ets)
            nchanges += self.eliminate_y()
            nchanges += self.merge()
            print(ets)

            if nchanges == 0:
                # try this then
                nchanges += self.substitute_to_z2()
            if nchanges == 0:
                break
        else:
            # too many iterations
            print('too many iterations')

    # ---------------------------------------------------------------------- #

    def float_right(self):
        """
        Attempt to 'float' translational terms as far to the right as
	 * possible and across joint boundaries.
        """
        
        nchanges = 0

        for i in range(len(self)):

            this = self[i]
            if this.isjoint or this.isrevolute:
                continue
            
            crossed = False
            for j in range(i+1, len(self)):
                next = self[j]

                if next.isprismatic:
                    continue
                if next.isrevolute and next.axis[1] == this.axis[1]:
                    crossed = True
                    continue
                break

            if crossed:
                del self[i]
                self.insert(j-1, this)
                nchanges += 1
                print('floated')

        return nchanges
    # ---------------------------------------------------------------------- #
    def swap(self, mdh=False):

        #  we want to sort terms into the order:
        # 	RZ
        # 	TX
        # 	TZ
        # 	RX

        def do_swap(this, next):
            if mdh:
                # modified DH
                return this.axis == 'Rx' and next.axis == 'tx' \
                    or this.axis == 'Ry' and next.axis == 'ty' \
                    or this.axis == 'Rz' and next.axis == 'tz' \
                    or this.axis == 'tz' and next.axis == 'tx'
            else:
                # standard DH

                # push constant translations through rotational joints
				# of the same type

                if this.axis == 'tz' and next.axis == 'tx':
                    return True

                if next.isjoint and (
                        this.axis == 'tx' and next.axis == 'Rx'
                        or this.axis == 'ty' and next.axis == 'Ry'
                        or this.axis == 'tz' and next.axis == 'Rz'):
                    return True
                elif not this.isjoint and (
                        this.axis == 'Rx' and next.axis == 'tx'
                        or this.axis == 'Ry' and next.axis == 'ty'):
                    return True

                if not this.isjoint and not next.isjoint and \
                        this.axis == 'tz' and next.axis == 'Rz':
                    return True

                #  move Ty terms to the right
                if this.axis == 'ty' and next.axis == 'tz' \
                        or this.axis == 'ty' and next.axis == 'tx':
                    return True

            return False

        total_changes = 0
        while True:
            nchanges = 0
            for i in range(len(self) - 1):
                this = self[i]
                next = self[i+1]

                if do_swap(this, next):
                    del self[i]
                    self.insert(i+1, this)
                    nchanges += 1
                    print(f"swapping {this} <--> {next}")
            if nchanges == 0:
                total_changes += nchanges
                break

        return total_changes

    # ---------------------------------------------------------------------- #


    def substitute_to_z(self):
        # substitute all non Z joint transforms according to rules
        nchanges = 0
        out = DHFactor()

        def subs_z(this):
            if this.axis == 'Rx':
                return DHFactor.ry(pi2) \
                     * DHFactor.rz(this.eta) \
                     * DHFactor.ry(-pi2)
            elif this.axis == 'Ry':
                return DHFactor.rx(-pi2) \
                    * DHFactor.rz(this.eta) \
                    * DHFactor.rx(pi2)
            elif this.axis == 'tx':
                return DHFactor.ry(pi2) \
                     * DHFactor.tz(this.eta) \
                     * DHFactor.ry(-pi2)
            elif this.axis == 'ty':
                return DHFactor.rx(-pi2) \
                     * DHFactor.tz(this.eta) \
                     * DHFactor.rx(pi2)

        for e in self:
            if not e.isjoint:
                out *= e
            else:
                # do a substitution
                new = DHFactor.subs_z(e)
                if new is None:
                    out *= e
                    continue
                else:
                    out *= new
                    nchanges += 1
                    print(f"subs2z: {e} := {new}")

        self.data = out.data
        return nchanges

    def substitute_to_z2(self):
        # substitute all non Z joint transforms according to rules
        nchanges = 0
        out = DHFactor()
        jointyet = False

        def subs_z(prev, this):
            if this.axis == 'Ry':
                return DHFactor.rz(pi2) \
                     * DHFactor.rx(this.eta) \
                     * DHFactor.rz(-pi2)
            elif this.axis == 'ty':
                if prev.axis == 'Rz':
                    return DHFactor.rz(pi2) \
                         * DHFactor.ty(this.eta) \
                         * DHFactor.r(-pi2)
                else:
                    return DHFactor.rx(-pi2) \
                         * DHFactor.ty(this.eta) \
                         * DHFactor.rx(pi2)

        for i in range(len(self)):
            this = self[i]
            if this.isjoint:
                jointyet = True
                continue
            
            if i == 0 or not jointyet:
                continue

            prev = self[i-1]

            new = subs_z(prev, this)

            if new is None:
                out *= this
                continue
            else:
                out *= new
                nchanges += 1
                print(f"subs2z2: {this} := {new}")

        self.data = out.data
        return nchanges
    # ---------------------------------------------------------------------- #

    def merge(self):

        def can_merge(prev, this):
            return prev.axis == this.axis and not (prev.isjoint and this.isjoint)

        out = DHFactor()
        nchanges = 0
        while len(self.data) > 0:
            this = self.pop(0)

            if len(self.data) > 0:
                next = self[0]

                if can_merge(this, next):
                    new = DHFactor.add(this, next)
                    out *= new
                    self.pop(0)  # remove next from the queue
                    print(f"Merging {this * next} to {new}")
                else:
                    out *= this
            else:
                out *= this

        self.data = out.data
        return nchanges
        # remove zeros

    @staticmethod
    def add(this, that):
        if this.isjoint and not that.isjoint:
            out = DHFactor(this)
            if out.eta is None:
                out.eta = that.eta
            else:
                out.eta += that.eta
        elif not this.isjoint and that.isjoint:
            out = DHFactor(that)
            if out.eta is None:
                out.eta = this.eta
            else:
                out.eta += this.eta
        elif not this.isjoint and not that.isjoint:
            out = DHFactor(this)
            if out.eta is None:
                out.eta = that.eta
            else:
                out.eta += that.eta
        else:
            raise ValueError('both ET cannot be joint variables')
        return out

    # ---------------------------------------------------------------------- #

    def eliminate_y(self):

        nchanges = 0
        out = DHFactor()
        jointyet = False

        def subs_y(prev, this):
            if this.isjoint or prev.isjoint:
                return None

            new = None
            if prev.axis == 'Rx' and this.axis == 'ty':  # RX.TY -> TZ.RX
                new = DHFactor.tx(prev.eta) * prev
            elif prev.axis == 'Rx' and this.axis == 'tz':  # RX.TZ -> TY.RX
                new = DHFactor.ty(-prev.eta) * prev
            elif prev.axis == 'Ry' and this.axis == 'tz':  # RY.TX-> TZ.RY
                new = DHFactor.tz(-prev.eta) * prev
            elif prev.axis == 'Ry' and this.axis == 'tz':  # RY.TZ-> TX.RY
                new = DHFactor.tx(prev.eta) * prev

            elif prev.axis == 'ty' and this.axis == 'Rx':  # TY.RX -> RX.TZ
                new = this * DHFactor.tz(-this.eta)
            elif prev.axis == 'tx' and this.axis == 'Rz':  # TX.RZ -> RZ.TY
                new = this * DHFactor.tz(this.eta)
            elif prev.axis == 'Ry' and this.axis == 'Rx':  # RY(Q).RX -> RX.RZ(-Q)
                new = this * DHFactor.Rz(-prev.eta)
            elif prev.axis == 'Rx' and this.axis == 'Ry':  # RX.RY -> RZ.RX
                new = DHFactor.Rz(this.eta) * prev
            elif prev.axis == 'Rz' and this.axis == 'Rx':  # RZ.RX -> RX.RY
                new = this * DHFactor.Ry(this.eta)
            return new

        for i in range(len(self)):
            this = self[i]
            jointyet = this.isjoint
            if i == 0 or not jointyet:  # leave initial const xform
                continue

            prev = self[i-1]

            new = subs_y(prev, this)
            if new is not None:
                self[i-1:i+1] = new
                nchanges += 1
                print(f"eliminate Y: {this} := {new}")


        return nchanges

    # ---------------------------------------------------------------------- #

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
        q = "q{0}"
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
    s = 'Tz(L1) Rz(q1) Ry(q2) Ty(L2) Tz(L3) Ry(q3) Tx(L4) Ty(L5) Tz(L6) Rz(q4) Ry(q5) Rz(q6)'

    ets = DHFactor.parse(s)
    print(ets)
    # ets.substitute_to_z()
    # print(ets)
    # ets.merge()
    # print(ets)

    ets.swap()
    ets.simplify()
    print(ets)
    # ets.merge()
    print(ets)

