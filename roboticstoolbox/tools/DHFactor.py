"""
@author Samuel Drew
"""


class Element:     # pragma nocover

    TX = 0
    TY = 1
    TZ = 2
    RX = 3
    RY = 4
    RZ = 5
    DH_STANDARD = 6
    DH_MODIFIED = 7

    # an array of counters for the application of each rule
    # just for debugging.
    rules = [0] * 20

    # mapping from type to string
    typeName = ["TX", "TY", "TZ", "RX", "RY", "RZ", "DH", "DHm"]

    # order of elementary transform for each DH convention
    # in each tuple, the first element is the transform type,
    # the second is true if it can be a joint variable.
    dhStandard = (RZ, 1), (TX, 0), (TZ, 1), (RX, 0)
    dhModified = (RX, 0), (TX, 0), (RZ, 1), (TZ, 1)

    def __init__(
            self,
            elementIn=None,
            stringIn=None,
            eltype=None,
            constant=None,
            sign=0):

        self.var = None         # eg. q1, for joint var types
        self.symconst = None    # eg. L1, for lengths

        # DH parameters, only set if type is DH_STANDARD/MODIFIED
        self.theta = None
        self.alpha = None
        self.A = None
        self.D = None
        self.prismatic = None
        self.offset = None

        if stringIn:
            if elementIn or eltype or constant:
                raise ValueError(
                    "if parsing a string, string must be the only input")
            i = None
            sType = stringIn[0:2]     # Tx, Rx etc
            sRest = stringIn[2:]       # the argument including brackets

            if not (sRest[-1] == ")" and sRest[0] == "("):
                raise ValueError("brackets")

            match = False
            for i in range(6):
                check = self.typeName[i].lower()
                if sType.lower() == check:
                    match = True
                    self.eltype = i
            if not match:
                raise ValueError("bad transform name: " + sType)

            sRest = sRest[1:-1]     # get the argument from between brackets

            # handle an optional minus sign
            negative = ""

            if sRest[0] == '-':
                negative = "-"
                sRest = sRest[1]

            if sRest[0] == "q":
                self.var = negative + sRest
            elif sRest[0] == 'L':
                self.symconst = negative + sRest
            else:
                try:
                    constant = float(sRest)
                    if negative == "-":
                        constant = -constant
                        # negative = ""
                except Exception:
                    raise ValueError("bad argument in term " + stringIn)

        elif elementIn:
            if not isinstance(elementIn, Element):
                raise TypeError("elementIn must be an existing Element object")
            self.eltype = elementIn.eltype
            if elementIn.var:
                self.var = elementIn.var
            if elementIn.symconst:
                self.symconst = elementIn.symconst
            self.constant = elementIn.constant
            # if sign < 0:
            #     self.negate()

        # one of TX, TY ... RZ, DH_STANDARD/MODIFIED
        if eltype:
            self.eltype = eltype
            if constant:
                self.constant = constant    # eg. 90, for angles
            if sign < 0:
                self.negate()

    @staticmethod
    def showRuleUsage():
        for i in range(20):
            if Element.rules[i] > 0:
                print("Rule " + str(i) + ": " + str(Element.rules[i]))

    def istrans(self):
        return (self.eltype == self.TX) or (self.eltype == self.TY) \
            or (self.eltype == self.TZ)

    def isrot(self):
        return (self.eltype == self.RX) or (self.eltype == self.RY) or \
            (self.eltype == self.RZ)

    def isjoint(self):
        return self.var is not None

    def axis(self):
        if self.eltype == self.RX or self.eltype == self.TX:
            return 0
        elif self.eltype == self.RY or self.eltype == self.TY:
            return 1
        elif self.eltype == self.RZ or self.eltype == self.TZ:
            return 2
        else:
            raise ValueError("bad transform type")

    def symAdd(self, s1, s2):

        if s1 is None and s2 is None:
            return None
        elif s1 and s2 is None:
            return s1
        elif s1 is None and s2:
            return s2
        else:
            if s2[0] == "-":
                return s1 + s2
            else:
                return s1 + "+" + s2

    def add(self, e):
        if self.eltype != Element.DH_STANDARD and \
                self.eltype != Element.DH_MODIFIED:
            raise ValueError("wrong element type " + str(self))
        print("  adding: " + str(self) + " += " + str(e))
        if e.eltype == self.RZ:
            if e.isjoint():
                self.prismatic = 0
                self.var = e.var
                self.offset = e.constant
                self.theta = 0.0
            else:
                self.theta = e.constant
        elif e.eltype == self.TX:
            self.A = e.symconst
        elif e.eltype == self.TZ:
            if e.isjoint():
                self.prismatic = 1
                self.var = e.var
                self.D = None
            else:
                self.D = e.symconst
        elif e.eltype == self.RX:
            self.alpha = e.constant
        else:
            raise ValueError("Can't factorise " + str(e))

    # test if this particular element could be part of a DH term
    # eg. Rz(q1) can be, Rx(q1) cannot.
    def factorMatch(self, dhWhich, i, verbose):

        dhFactors = None
        match = False

        if dhWhich == self.DH_STANDARD:
            dhFactors = self.dhStandard
        elif dhWhich == self.DH_MODIFIED:
            dhFactors = self.dhModified
        else:
            raise ValueError("bad DH type")

        match = (self.eltype == dhFactors[i][0]) and not (
            (dhFactors[i][1] == 0) and self.isjoint())

        if verbose > 0:
            print(" matching " + str(self) + " (i=" + str(i) + ") " +
                  " to " + self.typeName[dhFactors[i][0]] + "<" +
                  str(dhFactors[i][1]) + ">" + " -> " + str(match))
        return match

    def merge(self, e):
        assert type(e) == Element, "merge(Element e)"
        """
        don't merge if dissimilar transform or
        both are joint variables
        """
        if e.eltype != self.eltype or e.isjoint() and self.isjoint():
            return self

        sum = Element(self)

        sum.var = self.symAdd(self.var, e.var)
        sum.symconst = self.symAdd(self.symconst, e.symconst)
        sum.constant = self.constant + e.constant

        if not sum.isjoint() and sum.symconst is None and sum.constant == 0:
            print("Eliminate: " + self + " " + e)
            return None
        else:
            print("Merge: " + self + " " + e + " := " + sum)
            return sum

    def swap(self, next, dhWhich):
        assert type(next) == Element, "type(next) == Element"

        # don't swap if both are joint variables
        if self.isjoint() and next.isjoint():
            return False

        if dhWhich == Element.DH_STANDARD:
            # order = [2, 0, 3, 4, 0, 1]
            if self.eltype == Element.TZ and next.eltype == Element.TX or \
                    self.eltype == Element.TX and next.eltype == Element.RX \
                    and next.isjoint() or \
                    self.eltype == Element.TY and next.eltype == Element.RY \
                    and next.isjoint() or \
                    self.eltype == Element.TZ and next.eltype == Element.RZ \
                    and next.isjoint() or \
                    not self.isjoint() and self.eltype == Element.RX and \
                    next.eltype == Element.TX or \
                    not self.isjoint() and self.eltype == Element.RY and \
                    next.eltype == Element.TY or \
                    not self.isjoint() and not next.isjoint() and \
                    self.eltype == Element.TZ and \
                    next.eltype == Element.RZ or \
                    self.eltype == Element.TY and \
                    next.eltype == Element.TZ or \
                    self.eltype == Element.TY and next.eltype == Element.TX:
                print("Swap: " + self + " <-> " + next)
                return True
        elif dhWhich == Element.DH_MODIFIED:
            if self.eltype == Element.RX and next.eltype == Element.TX or \
                    self.eltype == Element.RY and \
                    next.eltype == Element.TY or \
                    self.eltype == Element.RZ and \
                    next.eltype == Element.TZ or \
                    self.eltype == Element.TZ and next.eltype == Element.TX:
                print("Swap: " + self + " <-> " + next)
                return True
        else:
            raise ValueError("bad DH type")
        return False

    # negate the arguments of the element
    def negate(self):

        self.constant = -self.constant

        if self.symconst:
            s = list(self.symconst)
            # if no leading sign character insert one (so we can flip it)
            if s[0] != "+" and s[0] != "-":
                s.insert(0, "+")
            for i in range(len(s)):
                if s[i] == "+":
                    s[i] = "-"
                elif s[i] == "-":
                    s[i] = "+"
                if s[0] == "+":
                    s.pop(0)
        s = "".join(s)    # lgtm [py/unused-local-variable]

    '''
    Return a string representation of the parameters (argument)
    of the element, which can be a number, symbolic constant,
    or a joint variable.
    '''
    def argString(self):
        s = ""

        if self.eltype == Element.RX or Element.RY or Element.RZ or \
                Element.TX or Element.TY or Element.TZ:
            if self.var:
                s = self.var
            if self.symconst:
                if self.var:
                    if self.symconst[0] != "-":
                        s = s + "+"
                s = s + self.symconst
            # constants always displayed with a sign character
            if self.constant != 0.0:
                if self.constant >= 0.0:
                    s = s + "+" + '{0:.3f}'.format(self.constant)
                else:
                    s = s + '{0:.3f}'.format(self.constant)
        elif self.eltype == Element.DH_STANDARD or Element.DH_MODIFIED:
            # theta, d, a, alpha
            # theta
            if self.prismatic == 0:
                # revolute joint
                s = s + self.var
                if self.offset >= 0:
                    s = s + "+" + '{0:.3f}'.format(self.offset)
                elif self.offset < 0:
                    s = s + '{0:.3f}'.format(self.offset)
            else:
                # prismatic joint
                s = s + '{0:.3f}'.format(self.theta)
            s = s + ", "

            # d
            if self.prismatic > 0:
                s = s + self.var
            else:
                s = s + self.D if self.D else s + "0"
            s = s + ", "

            # a
            s = s + self.A if self.A else s + "0"
            s = s + ", "

            # alpha
            s = s + '{0:.3f}'.format(self.alpha)
        else:
            raise ValueError("bad Element type")
        return s

    def toString(self):
        s = Element.typeName[self.eltype] + "("
        s = s + self.argString()
        s = s + ")"
        return s
