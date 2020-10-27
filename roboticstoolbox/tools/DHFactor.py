"""
@author Samuel Drew
"""


class Element:

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
                raise ValueError("if parsing a string, string must be the only input")
            i = None
            sType = stringIn[0:2]     # Tx, Rx etc
            sRest = stringIn[2:]       # the argument including brackets

            if not (sRest[-1] == ")" and sRest[0] == "("):
                raise ValueError("brackets")

            match = False
            for i in range(6):
                check = self.typeName[i].lower()
                if sType.lower == check:
                    match = True
            if not match:
                raise ValueError("bad transform name: " + sType)
            self.eltype = i

            sRest = sRest[1:-2]     # get the argument from between brackets

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
                        negative = ""
                except:
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

        # transform parameters, only one of these is set
        if constant:
            self.constant = constant    # eg. 90, for angles

    @staticmethod
    def showRuleUsage():
        for i in range(20):
            if Element.rules[i] > 0:
                print("Rule " + str(i) + ": " + str(Element.rules[i]))

    def istrans(self):
        return (self.eltype == self.TX) or (self.eltype == self.TY) or (self.eltype == self.TZ)

    def isrot(self):
        return (self.eltype == self.RX) or (self.eltype == self.RY) or (self.eltype == self.RZ)

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
        #TODO method for adding symbols
        print("symAdd not yet implemented")


    def add(self, e):
        if self.eltype != self.DH_Standard and self.eltype != self.DH_MODIFIED:
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

        match =	(self.eltype == dhFactors[i][0]) and not ((dhFactors[i][1] == 0) and self.isjoint())

        if verbose > 0:
            print(" matching " + str(self) + " (i=" + str(i) + ") " +
                  " to " + self.typeName[dhFactors[i][0]] + "<" +
                  str(dhFactors[i][1]) + ">" + " -> " + str(match))
        return match


    def merge(self, e):

        """
        don't merge if dissimilar transform or
        both are joint variables
        """
