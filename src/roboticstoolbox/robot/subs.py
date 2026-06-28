"""Rule-based substitution for lists of kinematic transformation elements."""

import re
from typing import List, Optional, Tuple, Union


class ET:
    """Kinematic transformation element with type, value, and joint flag."""

    VALID_TYPES = frozenset({'Rx', 'Ry', 'Rz', 'Tx', 'Ty', 'Tz'})

    def __init__(self, type_name: str, value: Union[str, int, float] = 0, joint: bool = False):
        if type_name not in self.VALID_TYPES:
            raise ValueError(f"Invalid type {type_name!r}")
        self.type = type_name
        self.value = value
        self.joint = joint

    def copy(self) -> 'ET':
        return ET(self.type, self.value, self.joint)

    def __repr__(self) -> str:
        return f"{self.type}({self.value!r}, joint={self.joint})"

    def __eq__(self, other: object) -> bool:
        return (isinstance(other, ET)
                and self.type == other.type
                and self.value == other.value
                and self.joint == other.joint)


def Rx(value=0, joint=False): return ET('Rx', value, joint)
def Ry(value=0, joint=False): return ET('Ry', value, joint)
def Rz(value=0, joint=False): return ET('Rz', value, joint)
def Tx(value=0, joint=False): return ET('Tx', value, joint)
def Ty(value=0, joint=False): return ET('Ty', value, joint)
def Tz(value=0, joint=False): return ET('Tz', value, joint)


# ── internal rule representation ──────────────────────────────────────────────

class _PatElem:
    """One element of a match (LHS) pattern."""
    __slots__ = ('type_pat', 'joint_req')

    def __init__(self, type_pat: str, joint_req: Optional[bool]):
        self.type_pat = type_pat
        self.joint_req = joint_req

    def matches(self, et: ET, wc: Optional[str]) -> Tuple[bool, Optional[str]]:
        """Return (ok, updated_wildcard_axis)."""
        if self.joint_req is not None and et.joint != self.joint_req:
            return False, wc
        if '?' in self.type_pat:
            if et.type[0] != self.type_pat[0]:
                return False, wc
            axis = et.type[1]
            if wc is not None and wc != axis:
                return False, wc
            return True, axis
        return self.type_pat == et.type, wc


class _ReplElem:
    """One element of a replacement (RHS) pattern."""
    __slots__ = ('type_pat', 'value_expr', 'joint_set', 'negate')

    def __init__(self, type_pat: str, value_expr: Optional[str],
                 joint_set: Optional[bool], negate: bool):
        self.type_pat = type_pat
        self.value_expr = value_expr
        self.joint_set = joint_set
        self.negate = negate

    def resolve_type(self, wc: Optional[str]) -> str:
        return self.type_pat.replace('?', wc) if '?' in self.type_pat else self.type_pat


class Rule:
    """A parsed substitution rule."""
    __slots__ = ('lookbehind', 'lhs', 'rhs', 'lookahead')

    def __init__(self, lookbehind: Optional[_PatElem], lhs: List[_PatElem],
                 rhs: List[_ReplElem], lookahead: Optional[_PatElem]):
        self.lookbehind = lookbehind
        self.lhs = lhs
        self.rhs = rhs
        self.lookahead = lookahead


# ── parsing ───────────────────────────────────────────────────────────────────

def _parse_pat_token(tok: str) -> _PatElem:
    joint_req: Optional[bool] = None
    if tok.startswith('@'):
        joint_req = True
        tok = tok[1:]
    elif tok.startswith('='):
        joint_req = False
        tok = tok[1:]
    return _PatElem(tok, joint_req)


def _parse_repl_token(tok: str) -> _ReplElem:
    negate = False
    joint_set: Optional[bool] = None
    if tok.startswith('~'):
        negate = True
        tok = tok[1:]
    elif tok.startswith('@'):
        joint_set = True
        tok = tok[1:]
    elif tok.startswith('='):
        joint_set = False
        tok = tok[1:]
    m = re.fullmatch(r'([A-Z][a-z?])(?:\(([^)]+)\))?', tok)
    if not m:
        raise ValueError(f"Cannot parse replacement token: {tok!r}")
    return _ReplElem(m.group(1), m.group(2), joint_set, negate)


def parse_rule(rule_str: str) -> Rule:
    """Parse a rule string such as ``'Rx Ty -> Tz($2) Rx($1)'`` into a Rule."""
    if '->' not in rule_str:
        raise ValueError(f"Rule must contain '->': {rule_str!r}")
    lhs_side, rhs_side = rule_str.split('->', 1)

    lookbehind: Optional[_PatElem] = None
    if '|' in lhs_side:
        lb, lhs_side = lhs_side.split('|', 1)
        lookbehind = _parse_pat_token(lb.strip())

    lookahead: Optional[_PatElem] = None
    if '|' in rhs_side:
        rhs_side, la = rhs_side.rsplit('|', 1)
        lookahead = _parse_pat_token(la.strip())

    lhs = [_parse_pat_token(t) for t in lhs_side.split()]
    rhs = [_parse_repl_token(t) for t in rhs_side.split()]
    rule = Rule(lookbehind, lhs, rhs, lookahead)
    _validate_rule(rule, rule_str)
    return rule


def _validate_rule(rule: Rule, rule_str: str) -> None:
    n = len(rule.lhs)
    lhs_has_wildcard = any('?' in p.type_pat for p in rule.lhs)

    for r in rule.rhs:
        if r.value_expr is not None:
            for m in re.finditer(r'\$(\d+)', r.value_expr):
                idx = int(m.group(1))
                if idx < 1 or idx > n:
                    raise ValueError(
                        f"${idx} in {r.value_expr!r} is out of range: "
                        f"LHS has {n} element(s) in rule {rule_str!r}"
                    )

        if '?' in r.type_pat and not lhs_has_wildcard:
            raise ValueError(
                f"RHS uses wildcard '?' in {r.type_pat!r} but LHS has none "
                f"in rule {rule_str!r}"
            )


# ── matching ──────────────────────────────────────────────────────────────────

def _try_match(lst: List[ET], i: int, rule: Rule
               ) -> Optional[Tuple[List[ET], Optional[str]]]:
    if rule.lookbehind is not None:
        if i == 0:
            return None
        ok, _ = rule.lookbehind.matches(lst[i - 1], None)
        if not ok:
            return None

    n = len(rule.lhs)
    if i + n > len(lst):
        return None

    wc: Optional[str] = None
    matched: List[ET] = []
    for pat, et in zip(rule.lhs, lst[i:i + n]):
        ok, wc = pat.matches(et, wc)
        if not ok:
            return None
        matched.append(et)

    if rule.lookahead is not None:
        j = i + n
        if j >= len(lst):
            return None
        ok, _ = rule.lookahead.matches(lst[j], None)
        if not ok:
            return None

    return matched, wc


# ── value expression evaluation ───────────────────────────────────────────────

def _eval_expr(expr: str, matched: List[ET]) -> Union[int, float]:
    resolved = re.sub(r'\$(\d+)',
                      lambda m: str(matched[int(m.group(1)) - 1].value),
                      expr)
    if not re.fullmatch(r'[\d\s.+\-eE]+', resolved):
        raise ValueError(f"Unsupported expression after substitution: {resolved!r}")
    return eval(resolved)  # safe: only numeric literals and +/- pass the check


# ── replacement building ──────────────────────────────────────────────────────

def _build_replacement(rhs: List[_ReplElem], matched: List[ET],
                       wc: Optional[str]) -> List[ET]:
    n_lhs = len(matched)
    n_rhs = len(rhs)
    auto_val: list = [None] * n_rhs
    auto_joint: List[bool] = [False] * n_rhs
    used = [False] * n_lhs

    # Phase 1: match each RHS element to a same-typed LHS element
    for j, r in enumerate(rhs):
        if r.value_expr is not None or '?' in r.type_pat:
            continue
        for i, et in enumerate(matched):
            if not used[i] and r.type_pat == et.type:
                auto_val[j] = et.value
                auto_joint[j] = et.joint
                used[i] = True
                break

    # Phase 2: assign remaining LHS values positionally to unresolved RHS slots
    rem_lhs = [(i, matched[i]) for i in range(n_lhs) if not used[i]]
    rem_rhs = [j for j in range(n_rhs)
               if auto_val[j] is None and rhs[j].value_expr is None]
    for j, (_, et) in zip(rem_rhs, rem_lhs):
        auto_val[j] = et.value
        auto_joint[j] = et.joint

    result: List[ET] = []
    for j, r in enumerate(rhs):
        et_type = r.resolve_type(wc)

        if r.value_expr is not None:
            val = _eval_expr(r.value_expr, matched)
        else:
            val = auto_val[j] if auto_val[j] is not None else 0

        if r.negate:
            val = -val

        joint = r.joint_set if r.joint_set is not None else auto_joint[j]
        result.append(ET(et_type, val, joint))

    return result


# ── public API ────────────────────────────────────────────────────────────────

def subs(lst: List[ET], rule: Union[str, Rule]) -> List[ET]:
    """Apply *rule* to *lst*, replacing all non-overlapping left-to-right matches.

    Rule syntax examples::

        "Rx Ty -> Tz($2) Rx($1)"      # swap with explicit value mapping
        "Rx @Ty -> @Tz Rx"            # @prefix = joint must be True
        "Rx =Ty -> Tz Rx"             # =prefix = joint must be False
        "Ty Rx -> Rx ~Tz"             # ~ negates the auto-assigned value
        "R? R? -> R?($1+$2)"          # ? wildcard matches same axis
        "Rz | Rx Ty -> Tz Rx"         # | on LHS = lookbehind
        "Rx Ty -> Tz Rx | Rx"         # | on RHS = lookahead
    """
    if isinstance(rule, str):
        rule = parse_rule(rule)

    result: List[ET] = []
    i = 0
    while i < len(lst):
        m = _try_match(lst, i, rule)
        if m is not None:
            matched, wc = m
            result.extend(_build_replacement(rule.rhs, matched, wc))
            i += len(rule.lhs)
        else:
            result.append(lst[i].copy())
            i += 1
    return result
