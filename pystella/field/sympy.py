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


import sympy as sym
import pymbolic.primitives as pp
from pymbolic.interop.sympy import PymbolicToSympyMapper, SympyToPymbolicMapper

__doc__ = """
.. currentmodule:: pystella.field.sympy

Sympy interoperability
^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: pymbolic_to_sympy
.. autofunction:: sympy_to_pymbolic
.. autofunction:: simplify
"""


class SympyField(sym.Indexed):
    def __new__(cls, field, base, *args, subscript=None, **kwargs):
        if subscript is None:
            subscript = tuple()
        symb = super().__new__(cls, base, *subscript, *args, **kwargs)
        symb.field = field
        symb.subscript = subscript

        return symb


class PymbolicToSympyMapperWithField(PymbolicToSympyMapper):
    def map_lookup(self, expr, *args, **kwargs):
        return pp.Variable(expr.name)

    def map_call(self, expr):
        function = self.rec(expr.function)
        if isinstance(function, pp.Variable):
            func_name = function.name
            try:
                func = getattr(self.sym.functions, func_name)
            except AttributeError:
                func = self.sym.Function(func_name)
            return func(*[self.rec(par) for par in expr.parameters])
        else:
            self.raise_conversion_error(expr)

    def map_field(self, expr):
        indices = tuple(self.rec(i) for i in expr.index_tuple)
        return SympyField(expr, self.rec(expr.child), *indices)

    def map_subscript(self, expr):
        from pystella import Field
        if isinstance(expr.aggregate, Field):
            f = expr.aggregate
            subscript = tuple(self.rec(i) for i in expr.index_tuple)
            indices = tuple(self.rec(i) for i in f.index_tuple)
            return SympyField(f, self.rec(f.child), *indices, subscript=subscript)
        else:
            return super().map_subscript(expr)

    map_dynamic_field = map_field


class SympyToPymbolicMapperMathLookup(SympyToPymbolicMapper):
    functions = {'exp', 'expm1', 'log',
                 'sin', 'cos', 'tan',
                 'sinh', 'cosh', 'tanh',
                 'fabs', 'Abs', 'sign'}

    def map_Function(self, expr):
        name = self.function_name(expr)
        if name in self.functions:
            args = tuple(self.rec(arg) for arg in expr.args)

            from pymbolic.primitives import Variable, Lookup
            if name == 'Abs':
                call = Lookup(Variable('math'), 'fabs')
            elif name == 'sign':
                call = Lookup(Variable('math'), 'copysign')
                args = (1,)+args
            else:
                call = Lookup(Variable('math'), name)
            return call(*args)
        else:
            return self.not_supported(expr)


class SympyToPymbolicMapperWithField(SympyToPymbolicMapperMathLookup):
    def map_SympyField(self, expr):
        f = expr.field
        if expr.subscript is not None:
            subscript = tuple(self.rec(i) for i in expr.subscript)
            return pp.Subscript(f, subscript)
        else:
            return f


#: A mapper which converts :class:`pymbolic.primitives.Expression`'s into
#: :mod:`sympy` expressions and understands :class:`~pystella.Field`'s.
#: The result can be converted back to a :class:`pymbolic.primitives.Expression`
#: with all :class:`~pystella.Field`'s in place, accomplished via a subclass
#: of :class:`sympy.Symbol` which retains a copy of the :class:`~pystella.Field`.
#:
#: :arg expr: The :mod:`pymbolic` expression to be mapped.
#:
#: .. warning::
#:
#:    Currently, :class:`~pystella.Field`'s of the form
#:    ``Field('f[0]')`` will not be processed correctly.
#:
pymbolic_to_sympy = PymbolicToSympyMapperWithField()

#: A mapper which converts :mod:`sympy` expressions into
#: :class:`pymbolic.primitives.Expression`'s and understands the custom :mod:`sympy`
#: type used to represent :class:`~pystella.Field`'s by :func:`pymbolic_to_sympy`.
#:
#: :arg expr: The :mod:`sympy` expression to be mapped.
#:
#: .. warning::
#:
#:    Currently, any modifications to the indices of a :class:`SympyField`
#:    will not be reflected when mapped back to a :class:`~pystella.Field`.
#:    Use :class:`pymbolic.primitives.Subscript` instead (i.e., process
#:    :class:`~pystella.Field`'s with :func:`~pystella.index_fields` first).
#:
sympy_to_pymbolic = SympyToPymbolicMapperWithField()


def simplify(expr, sympy_out=False):
    """
    A wrapper to :func:`sympy.simplify`.

    :arg expr: The expression to be simplified. May either be a
        :class:`pymbolic.primitives.Expression` or a :mod:`sympy` expression.

    The following keyword arguments are recognized:

    :arg sympy_out: A :class:`bool` determining whether to return the simplified
        :mod:`sympy` expression or to first convert it to a
        :class:`pymbolic.primitives.Expression`.
        Defaults to *False*.

    :returns: A :class:`pymbolic.primitives.Expression` containing the
        simplified form of ``expr`` if ``sympy_out`` is *True*, else a
        :mod:`sympy` expression.
    """

    if isinstance(expr, pp.Expression):
        expr = pymbolic_to_sympy(expr)
    expr = sym.simplify(expr)

    if sympy_out:
        return expr
    else:
        return sympy_to_pymbolic(expr)
