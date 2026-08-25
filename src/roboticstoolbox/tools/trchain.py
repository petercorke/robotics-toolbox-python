"""Evaluate chains of elementary homogeneous transforms."""

import ast
import operator
import re
from collections.abc import Callable, Mapping, Sequence
from typing import NamedTuple

import numpy as np
from spatialmath.base import symbolic as sym
from spatialmath.base import transl, transl2, trot2, trotx, troty, trotz


class TrChainToken(NamedTuple):
    """A parsed elementary-transform token.

    :param op: elementary-transform operator
    :param arg: expression inside the operator's parentheses
    :param index: one-based joint index, or zero for a constant expression
    """

    op: str
    arg: str
    index: int


_TOKEN_RE = re.compile(r"(?P<op>R.?|T.)\(")
_BINARY_OPERATORS: dict[type[ast.operator], Callable[[object, object], object]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPERATORS: dict[type[ast.unaryop], Callable[[object], object]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}
_DEFAULT_VALUES = {
    "pi": np.pi,
    "sin": sym.sin,
    "cos": sym.cos,
    "tan": sym.tan,
    "sqrt": sym.sqrt,
}

_Transform = Callable[[object, str], np.ndarray]
_TRANSFORMS_3D: dict[str, _Transform] = {
    "Rx": lambda value, unit: trotx(value, unit),
    "Ry": lambda value, unit: troty(value, unit),
    "Rz": lambda value, unit: trotz(value, unit),
    "Tx": lambda value, _unit: transl(value, 0, 0),
    "Ty": lambda value, _unit: transl(0, value, 0),
    "Tz": lambda value, _unit: transl(0, 0, value),
}
_TRANSFORMS_2D: dict[str, _Transform] = {
    "R": lambda value, unit: trot2(value, unit),
    "Rz": lambda value, unit: trot2(value, unit),
    "Tx": lambda value, _unit: transl2(value, 0),
    "Ty": lambda value, _unit: transl2(0, value),
}


def _parse(chain: str, qvar: str) -> tuple[TrChainToken, ...]:
    q_re = re.compile(rf"\b{re.escape(qvar)}([1-9][0-9]*)\b")
    tokens = []
    position = 0

    while match := _TOKEN_RE.search(chain, position):
        depth = 1
        for position in range(match.end(), len(chain)):
            if chain[position] == "(":
                depth += 1
            elif chain[position] == ")":
                depth -= 1
                if depth == 0:
                    break
        else:
            raise ValueError(f"unclosed transform {match['op']!r}")

        arg = chain[match.end() : position]
        indices = q_re.findall(arg)
        if len(indices) > 1:
            raise ValueError("only one joint variable is allowed in each transform")
        tokens.append(TrChainToken(match["op"], arg, int(indices[0]) if indices else 0))
        position += 1

    return tuple(tokens)


def _eval_expression(expression: str, values: Mapping[str, object]) -> object:
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"cannot evaluate expression {expression!r}") from exc

    def evaluate(node: ast.AST) -> object:
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.Name):
            try:
                return values[node.id]
            except KeyError as exc:
                raise ValueError(
                    f"unknown name {node.id!r} in expression {expression!r}"
                ) from exc
        if isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPERATORS:
            return _BINARY_OPERATORS[type(node.op)](
                evaluate(node.left), evaluate(node.right)
            )
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPERATORS:
            return _UNARY_OPERATORS[type(node.op)](evaluate(node.operand))
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and not node.keywords
        ):
            function = values.get(node.func.id)
            if callable(function):
                return function(*(evaluate(arg) for arg in node.args))
        raise ValueError(f"unsupported expression {expression!r}")

    return evaluate(tree.body)


def _evaluate_chain(
    chain: str | Sequence[TrChainToken],
    q: Sequence[object] | np.ndarray | None,
    unit: str,
    qvar: str,
    variables: Mapping[str, object] | None,
    transforms: Mapping[str, _Transform],
    size: int,
) -> tuple[np.ndarray, tuple[TrChainToken, ...]]:
    if unit not in {"rad", "deg"}:
        raise ValueError("unit must be 'rad' or 'deg'")
    if not qvar.isidentifier():
        raise ValueError("qvar must be a valid identifier")

    if isinstance(chain, str):
        tokens = _parse(chain, qvar)
    else:
        tokens = tuple(chain)
        if not all(isinstance(token, TrChainToken) for token in tokens):
            raise TypeError("chain must be a string or a sequence of TrChainToken")

    q_values = np.asarray(()) if q is None else np.asarray(q).reshape(-1)
    values = {**_DEFAULT_VALUES, **(variables or {})}
    T = np.eye(size)

    for token in tokens:
        try:
            transform = transforms[token.op]
        except KeyError as exc:
            raise ValueError(f"unknown transform {token.op!r}") from exc

        token_values = values
        if token.index:
            if token.index > len(q_values):
                raise ValueError("q has insufficient values")
            token_values = {
                **values,
                f"{qvar}{token.index}": q_values[token.index - 1],
            }

        T = T @ transform(_eval_expression(token.arg, token_values), unit)

    return T, tokens


def trchain(
    chain: str | Sequence[TrChainToken],
    q: Sequence[object] | np.ndarray | None = None,
    unit: str = "rad",
    *,
    qvar: str = "q",
    variables: Mapping[str, object] | None = None,
    return_tokens: bool = False,
) -> np.ndarray | tuple[np.ndarray, tuple[TrChainToken, ...]]:
    """Compound SE(3) elementary transforms from a string.

    :param chain: transform chain or tokens returned by an earlier call
    :param q: joint values referenced as ``q1``, ``q2``, and so on
    :param unit: angular unit, ``"rad"`` or ``"deg"``
    :param qvar: joint-variable prefix
    :param variables: values and functions used by token expressions
    :param return_tokens: also return the parsed tokens for reuse
    :returns: the SE(3) matrix, optionally followed by its parsed tokens
    :rtype: ndarray(4, 4) or tuple

    The chain contains ``Rx``, ``Ry``, ``Rz``, ``Tx``, ``Ty``, and ``Tz``
    tokens. Expressions support arithmetic, named values, ``pi``, and direct
    calls to named functions. Names other than joint variables are supplied by
    ``variables``.

    :seealso: :func:`trchain2`, :class:`roboticstoolbox.ETS`
    """
    result = _evaluate_chain(chain, q, unit, qvar, variables, _TRANSFORMS_3D, 4)
    return result if return_tokens else result[0]


def trchain2(
    chain: str | Sequence[TrChainToken],
    q: Sequence[object] | np.ndarray | None = None,
    unit: str = "rad",
    *,
    qvar: str = "q",
    variables: Mapping[str, object] | None = None,
    return_tokens: bool = False,
) -> np.ndarray | tuple[np.ndarray, tuple[TrChainToken, ...]]:
    """Compound SE(2) elementary transforms from a string.

    :param chain: transform chain or tokens returned by an earlier call
    :param q: joint values referenced as ``q1``, ``q2``, and so on
    :param unit: angular unit, ``"rad"`` or ``"deg"``
    :param qvar: joint-variable prefix
    :param variables: values and functions used by token expressions
    :param return_tokens: also return the parsed tokens for reuse
    :returns: the SE(2) matrix, optionally followed by its parsed tokens
    :rtype: ndarray(3, 3) or tuple

    The chain contains ``R`` (or ``Rz``), ``Tx``, and ``Ty`` tokens.
    Expressions are evaluated as described by :func:`trchain`.

    :seealso: :func:`trchain`, :class:`roboticstoolbox.ETS2`
    """
    result = _evaluate_chain(chain, q, unit, qvar, variables, _TRANSFORMS_2D, 3)
    return result if return_tokens else result[0]


__all__ = ["TrChainToken", "trchain", "trchain2"]
