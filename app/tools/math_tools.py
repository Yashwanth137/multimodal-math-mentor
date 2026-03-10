import sympy as sp
import numpy as np
from typing import Union, List, Optional
import json

def sympy_solver(expression: str) -> str:
    """Solves a symbolic math expression or equation."""
    try:
        expr = sp.sympify(expression)
        symbols = expr.free_symbols
        if not symbols:
            return str(expr.evalf())
        
        solutions = sp.solve(expr, list(symbols))
        return str(solutions)
    except Exception as e:
        return f"Error: {str(e)}"

def derivative_solver(expression: str, variable: str = 'x') -> str:
    """Computes the derivative of an expression."""
    try:
        var = sp.Symbol(variable)
        expr = sp.sympify(expression)
        diff_expr = sp.diff(expr, var)
        return str(diff_expr)
    except Exception as e:
        return f"Error: {str(e)}"

# THE FIX: Changed 'Any' to 'Optional[str]' so Pydantic can parse it into a JSON Schema for Gemini
def integral_solver(expression: str, variable: str = 'x', lower: Optional[str] = None, upper: Optional[str] = None) -> str:
    """Computes indefinite or definite integrals."""
    try:
        var = sp.Symbol(variable)
        expr = sp.sympify(expression)
        if lower is not None and upper is not None:
            # THE FIX: Convert string bounds back to symbolic/numeric for SymPy
            lower_bound = sp.sympify(lower) if isinstance(lower, str) else lower
            upper_bound = sp.sympify(upper) if isinstance(upper, str) else upper
            result = sp.integrate(expr, (var, lower_bound, upper_bound))
        else:
            result = sp.integrate(expr, var)
        return str(result)
    except Exception as e:
        return f"Error: {str(e)}"

def simplify_expression(expression: str) -> str:
    """Simplifies a mathematical expression."""
    try:
        expr = sp.sympify(expression)
        return str(sp.simplify(expr))
    except Exception as e:
        return f"Error: {str(e)}"

def matrix_operations(operation: str, matrices: List[List[List[float]]]) -> str:
    """
    Performs matrix operations like addition, multiplication, or determinant.
    Operation can be 'add', 'mul', 'det', 'inv'.
    """
    try:
        m_objs = [sp.Matrix(m) for m in matrices]
        if operation == 'add':
            result = m_objs[0] + m_objs[1]
        elif operation == 'mul':
            result = m_objs[0] * m_objs[1]
        elif operation == 'det':
            result = m_objs[0].det()
        elif operation == 'inv':
            result = m_objs[0].inv()
        else:
            return "Unsupported matrix operation"
        return str(result)
    except Exception as e:
        return f"Error: {str(e)}"

def numerical_calculator(expression: str) -> str:
    """Evaluates a numerical expression."""
    try:
        result = sp.sympify(expression).evalf()
        return str(result)
    except Exception as e:
        return f"Error: {str(e)}"

# Registry for Tool-Calling
MATH_TOOLS = {
    "sympy_solver": sympy_solver,
    "derivative_solver": derivative_solver,
    "integral_solver": integral_solver,
    "simplify_expression": simplify_expression,
    "matrix_operations": matrix_operations,
    "numerical_calculator": numerical_calculator
}

if __name__ == "__main__":
    print("Math Tools initialized.")