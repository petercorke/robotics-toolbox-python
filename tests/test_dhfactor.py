"""Unit tests for subs.py rule-based substitution system."""

import pytest
from subs import ET, Rx, Ry, Rz, Tx, Ty, Tz, subs, parse_rule


# ── ET class ──────────────────────────────────────────────────────────────────

class TestET:
    def test_int_value(self):
        e = Rx(90)
        assert e.type == 'Rx' and e.value == 90 and e.joint is False

    def test_float_value(self):
        assert Ry(1.5).value == 1.5

    def test_string_value(self):
        assert Rz("theta").value == "theta"

    def test_joint_true(self):
        assert Tx(0, joint=True).joint is True

    def test_joint_default_false(self):
        assert Ty(0).joint is False

    def test_invalid_type(self):
        with pytest.raises(ValueError):
            ET('Qx', 0)

    def test_copy_equal_not_same(self):
        e = Rx(10, joint=True)
        c = e.copy()
        assert c == e and c is not e

    def test_copy_independent(self):
        e = Rx(10)
        c = e.copy()
        c.value = 99
        assert e.value == 10

    def test_equality(self):
        assert Rx(1) == Rx(1)
        assert Rx(1) != Rx(2)
        assert Rx(1) != Ry(1)
        assert Rx(1, joint=True) != Rx(1, joint=False)

    def test_all_constructors(self):
        for fn, name in [(Rx,'Rx'),(Ry,'Ry'),(Rz,'Rz'),(Tx,'Tx'),(Ty,'Ty'),(Tz,'Tz')]:
            assert fn(5).type == name


# ── parse_rule ────────────────────────────────────────────────────────────────

class TestParseRule:
    def test_simple(self):
        r = parse_rule("Rx -> Ry")
        assert r.lhs[0].type_pat == 'Rx'
        assert r.rhs[0].type_pat == 'Ry'

    def test_no_arrow_raises(self):
        with pytest.raises(ValueError):
            parse_rule("Rx Ry")

    def test_lookbehind(self):
        r = parse_rule("Rz | Rx Ty -> Tz Rx")
        assert r.lookbehind.type_pat == 'Rz'
        assert len(r.lhs) == 2

    def test_lookahead(self):
        r = parse_rule("Rx Ty -> Tz Rx | Rx")
        assert r.lookahead.type_pat == 'Rx'
        assert len(r.rhs) == 2

    def test_joint_at_on_lhs(self):
        assert parse_rule("Rx @Ty -> Tz Rx").lhs[1].joint_req is True

    def test_joint_eq_on_lhs(self):
        assert parse_rule("Rx =Ty -> Tz Rx").lhs[1].joint_req is False

    def test_value_expr(self):
        r = parse_rule("Rx Ty -> Tz($2) Rx($1)")
        assert r.rhs[0].value_expr == '$2'
        assert r.rhs[1].value_expr == '$1'

    def test_negate(self):
        assert parse_rule("Ty Rx -> Rx ~Tz").rhs[1].negate is True

    def test_wildcard_lhs_rhs(self):
        r = parse_rule("R? R? -> R?($1+$2)")
        assert r.lhs[0].type_pat == 'R?'
        assert r.rhs[0].type_pat == 'R?'


class TestValidation:
    def test_dollar_index_zero_raises(self):
        with pytest.raises(ValueError, match=r"\$0"):
            parse_rule("Rx -> Rx($0)")

    def test_dollar_index_too_large_raises(self):
        with pytest.raises(ValueError, match=r"\$2"):
            parse_rule("Rx -> Rx($2)")

    def test_dollar_index_too_large_two_lhs(self):
        with pytest.raises(ValueError, match=r"\$3"):
            parse_rule("Rx Ty -> Tz($3)")

    def test_dollar_in_sum_out_of_range(self):
        with pytest.raises(ValueError, match=r"\$2"):
            parse_rule("Rx -> Rx($1+$2)")

    def test_wildcard_rhs_without_lhs_raises(self):
        with pytest.raises(ValueError, match="wildcard"):
            parse_rule("Rx -> R?")

    def test_valid_dollar_at_boundary(self):
        parse_rule("Rx Ty -> Tz($2) Rx($1)")   # $1 and $2 both valid

    def test_valid_wildcard_both_sides(self):
        parse_rule("R? R? -> R?($1+$2)")   # wildcard on LHS, so RHS ? is valid

    def test_valid_wildcard_lhs_only(self):
        parse_rule("R? -> Rx")   # wildcard on LHS but not RHS is fine


# ── basic substitution ────────────────────────────────────────────────────────

class TestBasic:
    def test_empty_list(self):
        assert subs([], "Rx -> Ry") == []

    def test_no_match(self):
        lst = [Rx(1), Ry(2)]
        assert subs(lst, "Tz -> Tx") == [Rx(1), Ry(2)]

    def test_type_change_value_preserved(self):
        # "Rx Ty -> Tz Rx" - order changes, values unchanged, type changes
        result = subs([Rx(5), Ty(3)], "Rx Ty -> Tz Rx")
        assert result[0] == Tz(3)
        assert result[1] == Rx(5)

    def test_explicit_value_mapping(self):
        # same semantics with explicit form
        result = subs([Rx(5), Ty(3)], "Rx Ty -> Tz($2) Rx($1)")
        assert result[0] == Tz(3)
        assert result[1] == Rx(5)

    def test_implicit_and_explicit_equivalent(self):
        lst = [Rx(5), Ty(3)]
        r1 = subs(lst, "Rx Ty -> Tz Rx")
        r2 = subs(lst, "Rx Ty -> Tz($2) Rx($1)")
        assert r1 == r2

    def test_multiple_matches(self):
        lst = [Rx(5), Ty(3), Rx(1), Ty(2)]
        result = subs(lst, "Rx Ty -> Tz($2) Rx($1)")
        assert result == [Tz(3), Rx(5), Tz(2), Rx(1)]

    def test_non_matching_passthrough(self):
        lst = [Rz(99), Rx(5), Ty(3), Rz(88)]
        result = subs(lst, "Rx Ty -> Tz($2) Rx($1)")
        assert result == [Rz(99), Tz(3), Rx(5), Rz(88)]

    def test_partial_match_at_end(self):
        lst = [Ry(1), Rx(5)]
        assert subs(lst, "Rx Ty -> Tz Rx") == [Ry(1), Rx(5)]

    def test_rule_object_accepted(self):
        rule = parse_rule("Rx -> Ry")
        assert subs([Rx(7)], rule) == [Ry(7)]

    def test_same_type_swap(self):
        # Both types appear on both sides — matched by type
        result = subs([Rz(7), Rx(5)], "Rz Rx -> Rx Rz")
        assert result[0] == Rx(5)
        assert result[1] == Rz(7)


# ── joint constraints ─────────────────────────────────────────────────────────

class TestJointConstraints:
    def test_at_matches_joint_true(self):
        result = subs([Rx(5), Ty(3, joint=True)], "Rx @Ty -> @Tz Rx")
        assert result[0].type == 'Tz'
        assert result[0].joint is True

    def test_at_no_match_joint_false(self):
        lst = [Rx(5), Ty(3, joint=False)]
        assert subs(lst, "Rx @Ty -> @Tz Rx") == lst

    def test_eq_matches_joint_false(self):
        result = subs([Rx(5), Ty(3, joint=False)], "Rx =Ty -> Tz Rx")
        assert result[0].type == 'Tz'
        assert result[1].type == 'Rx'

    def test_eq_no_match_joint_true(self):
        lst = [Rx(5), Ty(3, joint=True)]
        assert subs(lst, "Rx =Ty -> Tz Rx") == lst

    def test_joint_inherited_by_default(self):
        result = subs([Rx(5, joint=True)], "Rx -> Ry")
        assert result[0].joint is True

    def test_at_on_rhs_overrides(self):
        result = subs([Rx(5, joint=False)], "Rx -> @Ry")
        assert result[0].joint is True

    def test_eq_on_rhs_overrides(self):
        result = subs([Rx(5, joint=True)], "Rx -> =Ry")
        assert result[0].joint is False

    def test_rhs_at_preserves_value(self):
        result = subs([Rx(5), Ty(3, joint=True)], "Rx @Ty -> @Tz Rx")
        assert result[0].value == 3   # Tz gets Ty's value
        assert result[1].value == 5   # Rx keeps its value


# ── tilde (negation) ──────────────────────────────────────────────────────────

class TestNegation:
    def test_tilde_gives_negated_value(self):
        # "Ty Rx -> Rx ~Tz"  ≡  "Ty Rx -> Rx($2) Tz(-$1)"
        result = subs([Ty(4), Rx(2)], "Ty Rx -> Rx ~Tz")
        assert result[0] == Rx(2)
        assert result[1] == Tz(-4)

    def test_tilde_matches_explicit_form(self):
        lst = [Ty(4), Rx(2)]
        r1 = subs(lst, "Ty Rx -> Rx ~Tz")
        r2 = subs(lst, "Ty Rx -> Rx($2) Tz(-$1)")
        assert r1 == r2

    def test_explicit_unary_minus_in_expr(self):
        result = subs([Rx(5)], "Rx -> Rx(-$1)")
        assert result[0].value == -5


# ── wildcard ? ────────────────────────────────────────────────────────────────

class TestWildcard:
    def test_rz_rz_add(self):
        assert subs([Rz(10), Rz(20)], "R? R? -> R?($1+$2)") == [Rz(30)]

    def test_rx_rx_add(self):
        assert subs([Rx(3), Rx(7)], "R? R? -> R?($1+$2)") == [Rx(10)]

    def test_ry_ry_add(self):
        assert subs([Ry(1), Ry(2)], "R? R? -> R?($1+$2)") == [Ry(3)]

    def test_different_axis_no_match(self):
        lst = [Rx(10), Ry(20)]
        assert subs(lst, "R? R? -> R?($1+$2)") == lst

    def test_translation_wildcard(self):
        assert subs([Tx(5), Tx(3)], "T? T? -> T?($1+$2)") == [Tx(8)]

    def test_wildcard_preserves_axis_in_result(self):
        result = subs([Rz(5), Rz(5)], "R? R? -> R?($1+$2)")
        assert result[0].type == 'Rz'

    def test_wildcard_three_elements_partial(self):
        # First two match; third passes through
        lst = [Rx(3), Rx(4), Ry(1)]
        result = subs(lst, "R? R? -> R?($1+$2)")
        assert result == [Rx(7), Ry(1)]


# ── lookbehind ────────────────────────────────────────────────────────────────

class TestLookbehind:
    def test_match_with_lookbehind(self):
        lst = [Rz(0), Rx(5), Ty(3)]
        result = subs(lst, "Rz | Rx Ty -> Tz Rx")
        assert result[0] == Rz(0)
        assert result[1].type == 'Tz'
        assert result[2].type == 'Rx'

    def test_no_preceding_element(self):
        lst = [Rx(5), Ty(3)]
        assert subs(lst, "Rz | Rx Ty -> Tz Rx") == lst

    def test_wrong_preceding_type(self):
        lst = [Ry(0), Rx(5), Ty(3)]
        assert subs(lst, "Rz | Rx Ty -> Tz Rx") == lst

    def test_lookbehind_not_consumed(self):
        result = subs([Rz(99), Rx(5), Ty(3)], "Rz | Rx Ty -> Tz Rx")
        assert result[0] == Rz(99)
        assert len(result) == 3

    def test_lookbehind_joint_constraint(self):
        lst = [Rz(0, joint=True), Rx(5), Ty(3)]
        result = subs(lst, "@Rz | Rx Ty -> Tz Rx")
        assert result[1].type == 'Tz'

    def test_lookbehind_joint_constraint_fail(self):
        lst = [Rz(0, joint=False), Rx(5), Ty(3)]
        result = subs(lst, "@Rz | Rx Ty -> Tz Rx")
        assert result[1].type == 'Rx'   # not substituted


# ── lookahead ─────────────────────────────────────────────────────────────────

class TestLookahead:
    def test_match_with_lookahead(self):
        lst = [Rx(5), Ty(3), Rx(1)]
        result = subs(lst, "Rx Ty -> Tz Rx | Rx")
        assert result[0].type == 'Tz'
        assert result[1].type == 'Rx'
        assert result[2] == Rx(1)

    def test_no_following_element(self):
        lst = [Rx(5), Ty(3)]
        assert subs(lst, "Rx Ty -> Tz Rx | Rx") == lst

    def test_wrong_following_type(self):
        lst = [Rx(5), Ty(3), Rz(1)]
        assert subs(lst, "Rx Ty -> Tz Rx | Rx") == lst

    def test_lookahead_not_consumed(self):
        result = subs([Rx(5), Ty(3), Rx(99)], "Rx Ty -> Tz Rx | Rx")
        assert result[2] == Rx(99)


# ── arithmetic expressions ────────────────────────────────────────────────────

class TestArithmetic:
    def test_addition(self):
        assert subs([Rx(3), Rx(4)], "R? R? -> R?($1+$2)")[0].value == 7

    def test_subtraction(self):
        result = subs([Tz(10), Tz(4)], "T? T? -> T?($1-$2)")
        assert result[0].value == 6

    def test_unary_minus(self):
        assert subs([Rx(5)], "Rx -> Rx(-$1)")[0].value == -5

    def test_float_values(self):
        result = subs([Rx(1.5), Rx(2.5)], "R? R? -> R?($1+$2)")
        assert abs(result[0].value - 4.0) < 1e-10

    def test_two_variable_expr(self):
        result = subs([Rx(3), Ty(2)], "Rx Ty -> Tz($2) Rx($1)")
        assert result[0].value == 2
        assert result[1].value == 3
