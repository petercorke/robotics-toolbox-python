#!/usr/bin/env python3
import re
import sys
from collections import UserList

import sympy
import copy


class Element:
    # Transform types
    TX, TY, TZ = 0, 1, 2
    RX, RY, RZ = 3, 4, 5
    DH_STANDARD = 6
    DH_MODIFIED = 7

    TYPE_NAMES = ["Tx", "Ty", "Tz", "Rx", "Ry", "Rz", "DH", "DHm"]
    TYPE_NAMES_LC = ["tx", "ty", "tz", "rx", "ry", "rz", "dh", "dhm"]

    # Order of elementary transforms for DH conventions: (Type, IsJointPossible)
    DH_STANDARD_ORDER = [(RZ, True), (TZ, True), (TX, False), (RX, False)]
    DH_MODIFIED_ORDER = [(RX, False), (TX, False), (RZ, True), (TZ, True)]

    import sympy

    re_element = re.compile(r"([RT][xyz])\(([^)]+)\)")

    def __init__(self, *args):
        if len(args) == 1 and isinstance(args[0], str):
            # Rx(...)
            m = Element.re_element.match(args[0])
            if m is None:
                raise ValueError(f"bad string value: {el_value}")
            el_type = m.group(1)
            el_value = m.group(2)
        elif len(args) == 2:
            el_type = args[0]
            el_value = args[1]

        if isinstance(el_type, str):
            try:
                self.type = Element.TYPE_NAMES_LC.index(el_type.lower())
            except ValueError:
                raise ValueError(f"Invalid transform type: {el_type}")
        elif isinstance(el_type, int):
            self.type = el_type

        self.val = sympy.sympify(el_value)
        self._joint = False
        self._check_q(self)

    @staticmethod
    def _check_q(el):
        for var in el.val.free_symbols:
            if str(var).startswith("q"):
                el._joint = True

    def __add__(self, other):
        if self.type != other.type:
            raise ValueError("can only add Elements of the same type")
        new = copy.copy(self)
        new.val += other.val
        self._check_q(new)
        return new

    # def __init__(self, type_val, constant=0):
    #     if not isinstance(type_val, int) or not (0 <= type_val < len(self.TYPE_NAMES)):
    #         raise ValueError(f"Invalid transform type: {type_val}")
    #     self.type = type_val
    #     self.var = None  # e.g., 'q1'
    #     # self.symconst = None  # e.g., 'L1'
    #     if isinstance(constant, int):
    #         self.constant = sympy.Integer(constant)
    #     elif isinstance(constant, str):
    #         self.constant = sympy.symbol(constant)
    #     elif isinstance(constant, sympy.core.Expr):
    #         self.constant = constant
    #     else:
    #         raise ValueError("constant must be int or str")

    #     # DH specific parameters
    #     self.theta = 0.0
    #     self.alpha = 0.0
    #     self.A = None
    #     self.D = None
    #     self.prismatic = 0
    #     self.offset = 0.0

    # # Element(curr, Element.TZ, prev.constant)

    def clone(self, type_val, constant=0):
        new_el = Element(type_val, constant)
        new_el.val = self.val
        # new_el.symconst = self.symconst
        return new_el

    def __repr__(self):
        return f"{self.TYPE_NAMES[self.type]}({self.val})"
        # s += str(self.val)
        # # if self.symconst:
        # #     s += f"{self.symconst}"

        # return s + ")"

    @classmethod
    def from_string(cls, s):
        """Parses a string like 'Rx(90)' or 'Tz(q1)'."""
        match = re.match(r"([RT][xyz])\(([^)]+)\)", s, re.IGNORECASE)
        if not match:
            raise ValueError(f"Invalid transform format: {s}")

        s_type = match.group(1).lower()
        s_arg = match.group(2)

        # Determine type index
        try:
            type_idx = [name.lower() for name in cls.TYPE_NAMES].index(s_type)
        except ValueError:
            raise ValueError(f"Bad transform name: {s_type}")

        el = cls(type_idx)

        if s_arg.startswith("q") or s_arg.startswith("-q"):
            # we have a joint coordinate
            el.var = s_arg
        else:
            try:
                val = int(s_arg)
                el.constant = sympy.Integer(val)
            except ValueError:
                # not an int
                el.constant = sympy.symbols(s_arg)

        # # Handle sign
        # negative = False
        # if s_arg.startswith("-"):
        #     negative = True
        #     s_arg = s_arg[1:]

        # if s_arg.startswith("q"):
        #     el.var = ("-" if negative else "") + s_arg
        # elif s_arg.startswith("L"):
        #     el.symconst = ("-" if negative else "") + s_arg
        # else:
        #     try:
        #         el.constant = float(s_arg)
        #         if negative:
        #             el.constant = -el.constant
        #     except ValueError:
        #         raise ValueError(f"Bad argument in term: {s}")
        return el

    @property
    def is_trans(self):
        return self.type in (self.TX, self.TY, self.TZ)

    @property
    def is_rot(self):
        return self.type in (self.RX, self.RY, self.RZ)

    @property
    def is_joint(self):
        # return self._joint
        for var in self.val.free_symbols:
            if str(var).startswith("q"):
                return True
        else:
            return False

    @property
    def axis(self):
        if self.type in (self.TX, self.RX):
            return 0
        if self.type in (self.TY, self.RY):
            return 1
        if self.type in (self.TZ, self.RZ):
            return 2
        raise ValueError("Bad transform type for axis check")

    def negate(self):
        self.constant = -self.constant
        if self.symconst:
            # Flip all signs in the symbolic constant string
            s = self.symconst
            if not s.startswith(("-", "+")):
                s = "+" + s

            new_s = ""
            for char in s:
                if char == "+":
                    new_s += "-"
                elif char == "-":
                    new_s += "+"
                else:
                    new_s += char

            if new_s.startswith("+"):
                new_s = new_s[1:]
            self.symconst = new_s

    def sym_add(self, s1, s2):
        if s1 is None and s2 is None:
            return None
        if s1 is not None and s2 is None:
            return s1
        if s1 is None and s2 is not None:
            return s2
        if s2.startswith("-"):
            return f"{s1}{s2}"
        return f"{s1}+{s2}"

    def merge(self, other):
        """Attempts to merge two adjacent transforms."""
        if self.type != other.type or (self.is_joint and other.is_joint):
            return None

        return self + other
        # new_el = Element(self.type)
        # new_el.var = self.sym_add(self.var, other.var)
        # new_el.constant = self.constant + other.constant

        # if not new_el.is_joint and new_el.constant == 0:
        #     return None  # Null transform
        # return new_el

    def arg_string(self):
        # single ET
        return str(self.val).replace(" ", "")
        # if self.var is not None:
        #     sum = sympy.symbols(self.var)
        # else:
        #     sum = sympy.Integer(0)
        # if self.constant:
        #     sum += self.constant
        # if s and not self.symconst.startswith("-"):
        #     s += "+"
        # s += self.symconst
        # if sum != 0:
        #     s += str(sum).replace(" ", "")
        #     f"+{self.constant:g}" if self.constant > 0 else f"{self.constant:g}"
        # )
        # return str(sum).replace(" ", "") if sum else "0"

    def __str__(self):
        return f"{self.TYPE_NAMES[self.type]}({self.arg_string()})"


class DH(Element):
    def __init__(self, modified=False):
        # DH specific parameters
        self.theta = 0.0
        self.alpha = 0.0
        self.A = 0
        self.D = 0
        self.prismatic = 0
        self.offset = 0.0

    def add_to_dh(self, e):
        """Accumulates primitive transforms into a DH element."""
        # print(f"Adding {e} to DH element")

        if e.type == self.RZ:
            if e.is_joint:
                # q, c = e.val.as_coeff_Add()
                self.prismatic = 0
                # self.val = e.val
                # self.offset = c
                self.theta = e.val
            else:
                self.theta = e.val
        elif e.type == self.TX:
            self.A = e.val
        elif e.type == self.TZ:
            if e.is_joint:
                self.prismatic = 1
                self.var = e.val
                self.D = e.val
            else:
                self.D = e.val
        elif e.type == self.RX:
            self.alpha = e.val

    def __str__(self):
        # theta, d, a, alpha
        parts = []
        # if self.prismatic == 0:
        #     # revolute joint
        #     s_theta = self.val if self.val else "0"
        #     # s_theta += str(self.offset)
        #     # if self.offset > 0:
        #     #     s_theta += str(self.offset)
        #     # elif self.offset < 0:
        #     #     s_theta += f"{self.offset:g}"
        #     parts.append(f"{self.theta}")
        # else:
        #     # prismatic joijnt
        #     parts.append(f"{self.theta:g}")

        parts.append(str(self.theta).replace(" ", ""))
        parts.append(str(self.D).replace(" ", ""))
        parts.append(str(self.A).replace(" ", ""))
        parts.append(str(self.alpha).replace(" ", ""))
        return "DH[" +  ", ".join([str(part) for part in parts]) + "]"


class DHFactor(UserList):
    def __init__(self, src_string=None):
        if src_string:
            self.parse_string(src_string)

        self.A = 0
        self.D = 0
        self.alpha = 0
        self.theta = 0
        self.prismatic = 0

    def set(self, value):
        self.data = value

    def print(self, label):
        print(f"{label+':':<6s} {self}")

    def parse_string(self, buffer):
        # Tokenize e.g. "Rz(q1)Tx(L1)"
        tokens = re.findall(r"([RT][xyz]\([^)]+\))", buffer, re.IGNORECASE)
        self.data = [Element(t) for t in tokens]

        # print(f"INIT: {buffer}")
        self.print("INIT")
        self.simplify()
        self.print("SIMPLIFIED")

        self.factorize(Element.DH_STANDARD)
        self.print("FACTORIZED")

        if not self.is_valid():
            print("DHFactor: error: Incomplete factorization, no DH equivalent found")

    def is_valid(self):
        iprev = -1
        for i, e in enumerate(self):
            if e.type == Element.DH_STANDARD:
                if iprev >= 0 and (i - iprev) > 1:
                    return False
                iprev = i
        return True

    # def simplify(self):
    #     """Iterative simplification logic based on the Java program rules."""
    #     self._merge_all()
    #     self.print("MERGE")
    #     self._swap_all(Element.DH_STANDARD)
    #     self.print("SWAP")
    #     self._merge_all()
    #     self.print("MERGE")
    #     self._float_right()
    #     self.print("FLOAT")
    #     self._merge_all()
    #     self.print("MERGE")
    #     self._substitute_to_z()
    #     self.print("SUB_Z")
    #     self._merge_all()
    #     self.print("MERGE")
    #     print("----------------------")

    #     n_loops = 0
    #     while n_loops < 10:
    #         changes = 0
    #         changes += self._merge_all()
    #         self.print("MERGE")

    #         changes += self._swap_all(Element.DH_STANDARD)
    #         self.print("SWAP")

    #         changes += self._merge_all()
    #         self.print("MERGE")

    #         changes += self._substitute_y()
    #         self.print("SUB_Y")

    #         changes += self._merge_all()
    #         self.print("MERGE")

    #         if changes == 0:
    #             changes += self._substitute_to_z2()
    #             self.print("SUBZ2")
    #             changes += self._merge_all()
    #             self.print("MERGE")

    #         if changes == 0:
    #             break
    #         n_loops += 1

    def simplify(self):
        """Iterative simplification logic based on the Java program rules."""
        self._merge_all()
        self.print("MERGE")
        

        n_loops = 0
        while n_loops < 10:
            changes = 0

            changes += self._substitute_to_z2()
            self.print("SUBZ2")
            changes += self._merge_all()
            self.print("MERGE")

            if changes == 0:
                break
            n_loops += 1


    def factorize(self, dh_type):
        """Greedily identifies sequences of transforms that match DH parameters."""
        dh_order = (
            Element.DH_STANDARD_ORDER
            if dh_type == Element.DH_STANDARD
            else Element.DH_MODIFIED_ORDER
        )

        i = 0
        while i < len(self):
            match_count = 0
            joint_vars = 0

            # Check for sequence matching DH terms
            idx = i
            for f in range(4):
                if idx >= len(self):
                    break

                target_type, joint_allowed = dh_order[f]
                el = self[idx]

                if el.type == target_type and not (not joint_allowed and el.is_joint):
                    match_count += 1
                    idx += 1
                    if el.is_joint:
                        joint_vars += 1
                    if joint_vars > 1:
                        break  # Max 1 joint per DH row
                else:
                    continue

            if match_count > 0 and joint_vars > 0:
                # Merge into a DH element
                dh_el = DH(modified=False)
                for _ in range(match_count):
                    dh_el.add_to_dh(self.pop(i))
                self.insert(i, dh_el)
                i += 1
            else:
                i += 1

    # --- Internal Simplification Methods ---

    def _merge_all(self):
        changes = 0
        i = 0
        while i < len(self) - 1:
            while True:
                merged = self[i].merge(self[i + 1])
                if merged is not None:
                    if merged.val == 0:
                        # Null transform, remove both
                        self[i : i + 2] = []
                    else:
                        self[i : i + 2] = [merged]
                    changes += 1
                    if i >= len(self) - 1:
                        break
                else:
                    break
            i += 1
        return changes

    def _swap_all(self, dh_type):
        total_changes = 0
        while True:
            changes = 0
            for i in range(len(self) - 1):
                if self._should_swap(self[i], self[i + 1], dh_type):
                    self[i], self[i + 1] = (
                        self[i + 1],
                        self[i],
                    )
                    changes += 1
            total_changes += changes
            if changes == 0:
                break
        return total_changes

    def _should_swap(self, e1, e2, dh_type):
        if e1.is_joint and e2.is_joint:
            return False
        if dh_type == Element.DH_STANDARD:
            # Java logic: push constant translations through joints, order RZ, TX, TZ, RX
            if (
                (e1.type == Element.TX and e2.type == Element.TZ)
                or (e1.type == Element.TX and e2.type == Element.RX and e2.is_joint)
                or (e1.type == Element.TY and e2.type == Element.RY and e2.is_joint)
                or (e1.type == Element.TZ and e2.type == Element.RZ and e2.is_joint)
                or (not e1.is_joint and e1.type == Element.RX and e2.type == Element.TX)
                or (not e1.is_joint and e1.type == Element.RY and e2.type == Element.TY)
                or (
                    not e1.is_joint
                    and not e2.is_joint
                    and e1.type == Element.TZ
                    and e2.type == Element.RZ
                )
                or (e1.type == Element.TY and e2.type in (Element.TZ, Element.TX))
            ):
                return True
        return False

    def _float_right(self):
        changes = 0
        i = 0
        while i < len(self) - 1:
            e = self[i]
            if not e.is_joint and e.is_trans:
                crossed = False
                target_idx = i
                for j in range(i + 1, len(self)):
                    f = self[j]
                    if f.is_trans:
                        continue
                    if f.is_rot and f.axis == e.axis:
                        crossed = True
                        target_idx = j
                        continue
                    break
                if crossed:
                    self.pop(i)
                    self.insert(target_idx + 1, e)
                    changes += 1
                    # i -= 1  # change
            i += 1
        return changes

    def _substitute_to_z(self):
        changes = 0
        i = 0
        while i < len(self):
            e = self[i]
            if e.is_joint:
                repl = None
                if e.type == Element.RX:
                    repl = [
                        Element(Element.RY, 90),
                        e.clone(Element.RZ),
                        Element(Element.RY, -90),
                    ]
                elif e.type == Element.RY:
                    repl = [
                        Element(Element.RX, -90),
                        e.clone(Element.RZ),
                        Element(Element.RX, 90),
                    ]
                elif e.type == Element.TX:
                    repl = [
                        Element(Element.RY, 90),
                        e.clone(Element.TZ),
                        Element(Element.RY, -90),
                    ]
                elif e.type == Element.TY:
                    repl = [
                        Element(Element.RX, -90),
                        e.clone(Element.TZ),
                        Element(Element.RX, 90),
                    ]

                if repl:
                    self.pop(i)
                    for r in reversed(repl):
                        self.insert(i, r)
                    i += len(repl) - 1
                    changes += 1
            i += 1
        return changes

    def _substitute_y(self):
        changes = 0
        i = 1
        while i < len(self):
            prev, curr = self[i - 1], self[i]
            if prev.is_joint or curr.is_joint:
                i += 1
                continue

            repl = None
            if prev.type == Element.RX and curr.type == Element.TY:
                # RX.TY -> TZ.RX
                repl = [
                    curr.clone(Element.TZ, prev.val),
                    prev.clone(prev.type),
                ]
            elif prev.type == Element.RX and curr.type == Element.TZ:
                # RX.TZ -> TY.RX
                repl = [
                    curr.clone(Element.TY, -curr.val),
                    prev.clone(prev.type),
                ]
            elif prev.type == Element.RY and curr.type == Element.TX:
                # RY.TX -> TZ.RY
                repl = [
                    curr.clone(Element.TZ, -curr.val),
                    prev.clone(prev.type),
                ]
            elif prev.type == Element.RY and curr.type == Element.TZ:
                # RY.TZ -> TX.RY
                repl = [
                    curr.clone(Element.TX, curr.val),
                    prev.clone(prev.type),
                ]
            elif prev.type == Element.TX and curr.type == Element.RZ:
                # TX.RZ -> RY.TX
                repl = [
                    curr.clone(Element.RY, curr.val),
                    prev.clone(prev.type),
                ]
            elif prev.type == Element.RY and curr.type == Element.RX:
                # RY.RX -> RZ.RY
                repl = [
                    curr.clone(Element.RZ, -1),
                    prev.clone(prev.type),
                ]
            elif prev.type == Element.RX and curr.type == Element.RY:
                # RX.RY -> RZ.RX
                repl = [
                    curr.clone(Element.RZ),
                    prev.clone(prev.type),
                ]

            if repl:
                # self.pop(i)
                # self.pop(i - 1)
                # for r in reversed(repl):
                #     self.insert(i - 1, r)
                self[i - 1 : i + 1] = repl
                changes += 1
            else:
                i += 1
        return changes

    def _substitute_to_z2(self):
        changes = 0
        joint_yet = False
        i = 0
        while i < len(self):
            e = self[i]
            if e.is_joint:
                joint_yet = True
            if not joint_yet or i == 0 or e.is_joint:
                i += 1
                continue

            prev = self[i - 1]
            repl = None
            if e.type == Element.RY:
                repl = [
                    Element(Element.RZ, 90),
                    e.clone(Element.RX),
                    Element(Element.RZ, -90),
                ]
            elif e.type == Element.TY:
                if prev.type == Element.RZ:
                    repl = [
                        Element(Element.RZ, 90),
                        e.clone(Element.TX),
                        Element(Element.RZ, -90),
                    ]
                else:
                    repl = [
                        Element(Element.RX, -90),
                        e.clone(Element.TZ),
                        Element(Element.RX, 90),
                    ]

            if repl:
                self.pop(i)
                for r in reversed(repl):
                    self.insert(i, r)
                i += len(repl) - 1
                changes += 1
            i += 1
        return changes

    # --- Output Methods ---

    def _angle_str(self, a):
        if a == 0:
            return "0"
        if a == 90:
            return "pi/2"
        if a == -90:
            return "-pi/2"
        if a == 180:
            return "pi"
        if a == -180:
            return "-pi"
        raise ValueError(f"Bad transform angle: {a}")

    def base(self):
        for i, e in enumerate(self):
            if e.type in (Element.DH_STANDARD, Element.DH_MODIFIED):
                return self._el2matlab(0, i)
        return "eye(4,4)"

    def tool(self):
        for i in range(len(self) - 1, -1, -1):
            e = self[i]
            if e.type in (Element.DH_STANDARD, Element.DH_MODIFIED):
                return self._el2matlab(i + 1, len(self))
        return "eye(4,4)"

    def _el2matlab(self, start, end):
        parts = []
        for i in range(start, end):
            e = self[i]
            if e.type == Element.RX:
                parts.append(f"trotx({self._angle_str(e.constant)})")
            elif e.type == Element.RY:
                parts.append(f"troty({self._angle_str(e.constant)})")
            elif e.type == Element.RZ:
                parts.append(f"trotz({self._angle_str(e.constant)})")
            elif e.type == Element.TX:
                parts.append(f"transl({e.symconst},0,0)")
            elif e.type == Element.TY:
                parts.append(f"transl(0,{e.symconst},0)")
            elif e.type == Element.TZ:
                parts.append(f"transl(0,0,{e.symconst})")
        return "*".join(parts) if parts else "eye(4,4)"

    def dh_matrix(self):
        rows = []
        for e in self:
            if e.type == Element.DH_STANDARD:
                theta = "0" if e.prismatic == 0 else self._angle_str(e.theta)
                d = (
                    e.D
                    if (e.prismatic == 0 and e.D)
                    else ("0" if e.prismatic == 0 else "0")
                )
                a = e.A if e.A else "0"
                alpha = self._angle_str(e.alpha)
                rows.append(f"{theta}, {d}, {a}, {alpha}, {e.prismatic}")
        return "[" + "; ".join(rows) + "]"

    def offsets(self):
        offs = [
            self._angle_str(e.offset) for e in self if e.type == Element.DH_STANDARD
        ]
        return "[" + " ".join(offs) + "]"

    def command(self, name="robot"):
        if self.is_valid():
            return (
                f"SerialLink({self.dh_matrix()}, 'name', '{name}', "
                f"'base', {self.base()}, 'tool', {self.tool()}, "
                f"'offset', {self.offsets()})"
            )
        return "error('incompletely factored transform string')"

    def __str__(self):
        return ".".join(str(e) for e in self)

    def __repr__(self):
        return __str__(self)


def main():
    # Handle command line input
    input_str = ""
    if len(sys.argv) > 1:
        input_str = sys.argv[1]
    else:
        # Check if data is piped via stdin
        if not sys.stdin.isatty():
            input_str = sys.stdin.read().strip()

    if not input_str:
        print('Usage: python dhfactor.py "Rx(q1)Tx(L1)..."')
        sys.exit(1)

    # 
    # print("\nMatlab Command:")
    # print(factor.command("R"))


if __name__ == "__main__":
    s = "Tz(L0)Rz(q1)Tx(L1)Rx(q2)Tz(-L2)Ry(180)"

    s = "Tz(L1)Rz(q1)Ry(q2)Ty(L2)Tz(L3)Ry(q3)Tx(L6)Ty(L4)Tz(L5)Rz(q4)Ry(q5)Rz(q6)Tz(L7)"
    # s = "Tz(L1)Rz(q1)Tx(d2)Rx(q2)Tz(q3)Rx(q4)Rz(q5)Rx(q6)Tz(L7)"

    factor = DHFactor(s)
    print(factor)
