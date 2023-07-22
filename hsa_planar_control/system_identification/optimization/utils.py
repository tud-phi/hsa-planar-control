import sympy as sp
from typing import List


def isolateVariablesToLeftHandSide(eq: sp.Eq, syms: List[sp.Symbol]) -> sp.Eq:
    """
    Isolate the variables referenced in syms to the left-hand side of the equation.
    Args:
        eq: sympy equation
        syms: list of sympy symbols

    Returns:

    """
    lhs = sp.zeros(*eq.args[0].shape)
    rhs = eq.args[1] - eq.args[0]
    # important: we first need to remove all unnecessary parentheses
    # otherwise the as_independent function will not work
    rhs = sp.expand(rhs)
    # loop through the rows of the equation
    new_lhs_rows = []
    new_rhs_rows = []
    for i in range(rhs.shape[0]):
        lhs_row, rhs_row = lhs[i], rhs[i]
        for e in syms:
            # move the dependent terms to the left-hand side
            ind = rhs_row.as_independent(e)[1]
            lhs_row = lhs_row - ind
            rhs_row = rhs_row - ind
        new_lhs_rows.append(lhs_row)
        new_rhs_rows.append(rhs_row)
    lhs = sp.Matrix(new_lhs_rows)
    rhs = sp.Matrix(new_rhs_rows)
    return sp.Eq(lhs, rhs)
