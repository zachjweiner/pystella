__copyright__ = "Copyright (C) 2019 Zachary J Weiner"

__license__ = """
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""


import pystella as ps
from pymbolic import parse, var
from pystella.field import shift_fields
import pytest


def test_field(proc_shape):
    if proc_shape != (1, 1, 1):
        pytest.skip("test field only on one rank")

    y = ps.Field('y', offset='h')
    result = ps.index_fields(y)
    assert result == parse("y[i + h, j + h, k + h]"), result

    y = ps.Field('y', offset='h', indices=('a', 'b', 'c'))
    result = ps.index_fields(y)
    assert result == parse("y[a + h, b + h, c + h]"), result

    y = ps.Field('y', ignore_prepends=True)
    result = ps.index_fields(y, prepend_with=(0, 1))
    assert result == parse("y[i, j, k]"), result

    y = ps.Field('y[4, 5]', ignore_prepends=True)
    result = ps.index_fields(y, prepend_with=(0, 1))
    assert result == parse("y[4, 5, i, j, k]"), result

    y = ps.Field('y', ignore_prepends=True)
    result = ps.index_fields(y[2, 3], prepend_with=(0, 1))
    assert result == parse("y[2, 3, i, j, k]"), result

    y = ps.Field('y[4, 5]', ignore_prepends=True)
    result = ps.index_fields(y[2, 3], prepend_with=(0, 1))
    assert result == parse("y[2, 3, 4, 5, i, j, k]"), result

    y = ps.Field('y', ignore_prepends=False)
    result = ps.index_fields(y, prepend_with=(0, 1))
    assert result == parse("y[0, 1, i, j, k]"), result

    y = ps.Field('y[4, 5]', ignore_prepends=False)
    result = ps.index_fields(y, prepend_with=(0, 1))
    assert result == parse("y[0, 1, 4, 5, i, j, k]"), result

    y = ps.Field('y', ignore_prepends=False)
    result = ps.index_fields(y[2, 3], prepend_with=(0, 1))
    assert result == parse("y[0, 1, 2, 3, i, j, k]"), result

    y = ps.Field('y[4, 5]', ignore_prepends=False)
    result = ps.index_fields(y[2, 3], prepend_with=(0, 1))
    assert result == parse("y[0, 1, 2, 3, 4, 5, i, j, k]"), result

    y = ps.Field('y', offset=('hx', 'hy', 'hz'))
    result = ps.index_fields(shift_fields(y, (1, 2, 3)))
    assert result == parse("y[i + hx + 1, j + hy + 2, k + hz + 3]"), result

    y = ps.Field('y', offset=('hx', var('hy'), 'hz'))
    result = ps.index_fields(shift_fields(y, (1, 2, var('a'))))
    assert result == parse("y[i + hx + 1, j + hy + 2, k + hz + a]"), result


def test_dynamic_field(proc_shape):
    if proc_shape != (1, 1, 1):
        pytest.skip("test field only on one rank")

    y = ps.DynamicField('y', offset='h')

    result = ps.index_fields(y)
    assert result == parse("y[i + h, j + h, k + h]"), result

    result = ps.index_fields(y.lap)
    assert result == parse("lap_y[i, j, k]"), result

    result = ps.index_fields(y.dot)
    assert result == parse("dydt[i + h, j + h, k + h]"), result

    result = ps.index_fields(y.pd[var('x')])
    assert result == parse("dydx[x, i, j, k]"), result

    result = ps.index_fields(y.d(1, 0))
    assert result == parse("dydt[1, i + h, j + h, k + h]"), result

    result = ps.index_fields(y.d(1, 1))
    assert result == parse("dydx[1, 0, i, j, k]"), result


def test_field_diff(proc_shape):
    if proc_shape != (1, 1, 1):
        pytest.skip("test field only on one rank")

    from pystella import diff

    y = ps.Field('y')
    assert diff(y, y) == 1
    assert diff(y[0], y[0]) == 1
    assert diff(y[0], y[1]) == 0

    y = ps.DynamicField('y')
    assert diff(y, y) == 1
    assert diff(y[0], y[0]) == 1
    assert diff(y[0], y[1]) == 0

    import pymbolic.primitives as pp
    assert diff(y**3, y, 't') == pp.Product((3, 2, y, y.d(0)))
    assert diff(y**3, 't', y) == pp.Product((3, y.d(0), 2, y))

    for i, x in enumerate(['t', 'x', 'y', 'z']):
        assert diff(y, x) == y.d(i)
        assert diff(y[1, 3], x) == y.d(1, 3, i)
        assert diff(y[1]**2, x) == 2 * y[1] * y.d(1, i)


def test_get_field_args(proc_shape):
    if proc_shape != (1, 1, 1):
        pytest.skip("test field only on one rank")

    from pystella import Field, get_field_args

    x = Field('x', offset=(1, 2, 3))
    y = Field('y', offset='h')
    z = Field('z')

    from loopy import GlobalArg
    true_args = [
        GlobalArg('x', shape='(Nx+2, Ny+4, Nz+6)'),
        GlobalArg('y', shape='(Nx+2*h, Ny+2*h, Nz+2*h)'),
        GlobalArg('z', shape='(Nx, Ny, Nz)'),
    ]

    def lists_equal(a, b):
        equal = True
        for x in a:
            equal *= x in b
        for x in b:
            equal *= x in a
        return equal

    expressions = {x: y, y: x * z}
    args = get_field_args(expressions)
    assert lists_equal(args, true_args)

    expressions = x * y + z
    args = get_field_args(expressions)
    assert lists_equal(args, true_args)

    expressions = [x, y, y * z**2]
    args = get_field_args(expressions)
    assert lists_equal(args, true_args)

    expressions = [shift_fields(x, (1, 2, 3)), y, y * z**2]
    args = get_field_args(expressions)
    assert lists_equal(args, true_args)


def test_sympy_interop(proc_shape):
    if proc_shape != (1, 1, 1):
        pytest.skip("test field only on one rank")

    from pystella.field.sympy import pymbolic_to_sympy, sympy_to_pymbolic
    import sympy as sym

    f = ps.Field('f', offset='h')
    g = ps.Field('g', offset='h')

    expr = f[0]**2 * g + 2 * g[1] * f
    sympy_expr = pymbolic_to_sympy(expr)
    new_expr = sympy_to_pymbolic(sympy_expr)
    sympy_expr_2 = pymbolic_to_sympy(new_expr)
    assert sym.simplify(sympy_expr - sympy_expr_2) == 0, \
        "sympy <-> pymbolic conversion not invertible"

    expr = f + shift_fields(f, (1, 2, 3))
    sympy_expr = pymbolic_to_sympy(expr)
    new_expr = sympy_to_pymbolic(sympy_expr)
    sympy_expr_2 = pymbolic_to_sympy(new_expr)
    assert sym.simplify(sympy_expr - sympy_expr_2) == 0, \
        "sympy <-> pymbolic conversion not invertible with shifted indices"

    # from pymbolic.functions import fabs, exp, exmp1
    fabs = parse('math.fabs')
    exp = parse('math.exp')
    expm1 = parse('math.expm1')
    x = sym.Symbol('x')

    expr = sym.Abs(x)
    assert sympy_to_pymbolic(expr) == fabs(var('x'))

    expr = sym.exp(x)
    assert sympy_to_pymbolic(expr) == exp(var('x'))

    expr = sym.Function('expm1')(x)
    assert sympy_to_pymbolic(expr) == expm1(var('x'))

    expr = sym.Function('aaa')(x)
    from pymbolic.primitives import Call, Variable
    assert sympy_to_pymbolic(expr) == Call(Variable('aaa'), (Variable('x'),))


if __name__ == "__main__":
    test_field((1, 1, 1))
    test_dynamic_field((1, 1, 1))
    test_field_diff((1, 1, 1))
    test_get_field_args((1, 1, 1))
    test_sympy_interop((1, 1, 1))
