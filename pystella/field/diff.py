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


import pymbolic.primitives as pp
from pymbolic.mapper.differentiator import DifferentiationMapper
from pymbolic import var


class FieldDifferentiationMapper(DifferentiationMapper):
    def __init__(self, variable, xmu=None):
        if xmu is not None:
            self.xmu = xmu
        else:
            self.xmu = {var('t'): 0, var('x'): 1, var('y'): 2, var('z'): 3}
        super().__init__(variable)

    map_field = DifferentiationMapper.map_variable

    def map_dynamic_field(self, expr, *args):
        if self.variable in self.xmu:
            return expr.d(*args, self.xmu[self.variable])
        else:
            return self.map_field(expr, *args)

    def map_subscript(self, expr, *args):
        from pystella.field import DynamicField
        if isinstance(expr.aggregate, DynamicField) and self.variable in self.xmu:
            return self.rec(expr.aggregate, *expr.index_tuple)
        else:
            return super().map_subscript(expr, *args)

    def map_if(self, expr, *args):
        from pymbolic.primitives import If
        return If(expr.condition, self.rec(expr.then), self.rec(expr.else_))


def diff(f, *x):
    """
    A differentiator which computes :math:`\\partial f / \\partial x` and understands
    :class:`Field`'s. If ``x`` is one of ``t``, ``x``, ``y``, or ``z`` and ``f``
    is a :class:`DynamicField`, the corresponding derivative :class:`Field` is
    returned.

    Examples::

        >>> f = DynamicField('f')
        >>> print(diff(f**3, f))
        3*f**2
        >>> print(diff(f**3, f, f))
        3*2*f
        >>> print(diff(f**3, 't'))
        3*f**2*dfdt
        >>> print(diff(f**3, f, 't'))
        3*2*f*dfdt
        >>> print(diff(f + 2, 'x'))
        dfdx[0]

    :arg f: A :mod:`pymbolic` expression to be differentiated.

    :arg x: A :class:`pymbolic.primitives.Expression` or a string to be parsed
        (or multiple thereof). If multiple positional arguments are provided,
        derivatives are taken with respect to each in order.
        (See the examples above.)
    """

    if len(x) > 1:
        return diff(diff(f, x[0]), *x[1:])
    else:
        return FieldDifferentiationMapper(pp.make_variable(x[0]))(f)
