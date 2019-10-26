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
from pymbolic import parse
from pymbolic.mapper import IdentityMapper, Collector
from pymbolic.mapper.stringifier import StringifyMapper
from pystella.field.diff import diff
# from pystella.field.sympy import pymbolic_to_sympy, sympy_to_pymbolic, simplify

__doc__ = """
.. currentmodule:: pystella
.. autoclass:: Field
.. autoclass:: DynamicField
.. autofunction:: index_fields
.. autofunction:: diff
.. currentmodule:: pystella.field
.. autofunction:: get_field_args
.. automodule:: pystella.field.sympy
"""


def parse_if_str(expr):
    return parse(expr) if isinstance(expr, str) else expr


class Field(pp.AlgebraicLeaf):
    """
    A :class:`pymbolic.primitives.Expression` designed to mimic an array by carrying
    information about indexing. Kernel generators (:class:`Reduction`,
    :class:`ElementWiseMap`, and subclasses) automatically append indexing
    specified by the attributes :attr:`indices` and :attr:`offset`
    (via :attr:`index_tuple`) by pre-processing
    the expressions with :func:`index_fields`.

    Examples::

        >>> f = Field('f', offset='h')
        >>> print(index_fields(f))
        f[i + h, j + h, k + h]
        >>> print(index_fields(f[0]))
        f[0, i + h, j + h, k + h]

    See `test_field.py
    <https://github.com/zachjweiner/pystella/blob/master/test/test_field.py>`_
    for more examples of
    the intended functionality.

    .. automethod:: __init__
    .. autoattribute:: index_tuple
    """

    init_arg_names = ('child', 'offset', 'indices', 'base_offset', 'ignore_prepends')

    def __init__(self, child, offset=0, indices=('i', 'j', 'k'), base_offset=None,
                 ignore_prepends=False):
        """
        :arg child: The child expression representing the unsubscripted field.
            May be a string, a :class:`pymbolic.primitives.Variable`, or a
            :class:`pymbolic.primitives.Subscript`.

        :arg indices: A tuple of (symbolic) array indices that will subscript
            the array.
            Each entry may be a :class:`pymbolic.primitives.Variable` or a string
            which parses to one.
            Defaults to ``('i', 'j', 'k')``

        :arg offset: The amount of padding by which to offset the array axes
            corresponding to the elements of :attr:`indices`. May be a tuple with
            the same length as :attr:`indices` or a single value.
            In the latter case, the input is transformed into a tuple with the same
            length as :attr:`indices`, each with the same value.
            Defaults to ``0``.

        :arg ignore_prepends: Whether to ignore array subscripts prepended when
            processed with :func:`index_fields`. Useful for timestepping kernels
            (e.g., :class:`~pystella.step.RungeKuttaStepper`) which prepend array
            indices corresponding to extra storage axes (to specify that an array
            does not have this axis).
            Defaults to *False*.
        """

        self.child = parse_if_str(child)
        if isinstance(self.child, pp.Subscript):
            self.name = self.child.aggregate.name
        else:
            self.name = self.child.name

        if not isinstance(offset, (list, tuple)):
            offset = (offset,)*len(indices)
        if len(offset) != len(indices):
            raise ValueError(
                'offset (if not length-1) must have same length as indices'
            )

        self.offset = tuple(parse_if_str(o) for o in offset)
        self.base_offset = base_offset or self.offset
        self.indices = tuple(parse_if_str(i) for i in indices)

        self.ignore_prepends = ignore_prepends

    def __getinitargs__(self):
        return (self.child, self.offset, self.indices, self.base_offset,
                self.ignore_prepends)

    mapper_method = "map_field"

    @property
    def index_tuple(self):
        """
        The fully-expanded subscript (i.e., :attr:`indices`
        offset by :attr:`offset`.)
        """

        return tuple(i + o for i, o in zip(self.indices, self.offset))

    def make_stringifier(self, originating_stringifier=None):
        # FIXME: do something with originating_stringifier?
        return FieldStringifyMapper()

    def new_with_changes(self, **kwargs):
        init_kwargs = dict(zip(self.init_arg_names, self.__getinitargs__()))
        init_kwargs.update(kwargs)
        return Field(**init_kwargs)


class FieldStringifyMapper(StringifyMapper):
    def map_field(self, expr, enclosing_prec, *args, **kwargs):
        return self.rec(expr.child, enclosing_prec, *args, **kwargs)

    map_dynamic_field = map_field


class DynamicField(Field):
    """
    A subclass of :class:`Field` which also contains associated :class:`Field`
    instances representing various derivatives of the base :class:`Field`.

    .. attribute:: dot

        A :class:`Field` representing the time derivative of the base
        :class:`Field`. It shares the same :attr:`indices` and :attr:`offset`
        as the base :class:`Field`. Its name defaults to ``d{self.child}dt``,
        but may be specified via the argument ``dot_child``.

    .. attribute:: lap

        A :class:`Field` representing the Laplacian of the base
        :class:`Field`. It shares the same :attr:`indices` as the base
        :class:`Field` but with ``offset = 0``. Its name defaults to
        ``lap_{self.child}``, but may be specified via the argument
        ``lap_child``.

    .. attribute:: pd

        A :class:`Field` representing the spatial derivative(s) of the base
        :class:`Field`. It shares the same :attr:`indices` as the base
        :class:`Field` but with ``offset = 0``. Its name defaults to
        ``d{self.child}dx``, but may be specified via the argument
        ``pd_child``.

    .. automethod:: d
    """

    init_arg_names = ('child', 'offset', 'indices', 'base_offset',
                      'dot_child', 'lap_child', 'pd_child')

    def __init__(self, child, offset='0', indices=('i', 'j', 'k'),
                 base_offset=None, dot_child=None, lap_child=None, pd_child=None):
        super().__init__(child, offset, indices, base_offset=base_offset)

        self.dot = Field(dot_child or 'd' + str(child) + 'dt',
                         offset, indices=indices)

        self.lap = Field(lap_child or 'lap_' + str(child),
                         offset=0, indices=indices, ignore_prepends=True)

        self.pd = Field(pd_child or'd' + str(child) + 'dx',
                        offset=0, indices=indices, ignore_prepends=True)

    def __getinitargs__(self):
        return (self.child, self.offset, self.indices, self.base_offset,
                self.dot.child, self.lap.child, self.pd.child)

    def d(self, *args):
        """
        Returns the (subscripted) derivative of the base :class:`Field`, i.e.,
        either :attr:`dot` or :attr:`pd` with the appropriate index.

        For example, the "time" derivative of a field would be

            >>> f = DynamicField('f')
            >>> print(f.d(0))  # x^0 = "time"
            dfdt

        Additional arguments are interpreted as subscripts to the resulting array;
        the final argument corresponds to the coordinate being differentiated with
        respect to.

            >>> print(f.d(1, 2, 0))
            dfdt[1, 2]

        Spatial indices ``1`` through ``3`` denote spatial derivatives (whose
        array subscripts are ``0`` through ``2``).

            >>> print(f.d(2))  # x^2 = y
            dfdx[1]
            >>> print(f.d(0, 1, 3))  # x^3 = z
            dfdx[0, 1, 2]
        """
        mu = args[-1]
        indices = args[:-1]+(mu-1,)
        return self.dot[args[:-1]] if mu == 0 else self.pd[indices]

    mapper_method = "map_dynamic_field"


def parse_prepend(*indices):
    def parse_if_str(x):
        return parse(x) if isinstance(x, str) else x

    return tuple(map(parse_if_str, indices))


class IndexMapper(IdentityMapper):
    def map_field(self, expr, *args, **kwargs):
        if expr.ignore_prepends:
            pre_index = ()
        else:
            pre_index = parse_prepend(*kwargs.pop('prepend_with', ()))

        pre_index = pre_index + kwargs.pop('outer_subscript', ())
        full_index = pre_index + expr.index_tuple

        if full_index == tuple():
            x = expr.child
        else:
            if isinstance(expr.child, pp.Subscript):
                full_index = pre_index + expr.child.index_tuple + expr.index_tuple
                x = pp.Subscript(expr.child.aggregate, self.rec(full_index))
            else:
                x = pp.Subscript(expr.child, self.rec(full_index))

        return self.rec(x, *args, **kwargs)

    map_dynamic_field = map_field

    def map_subscript(self, expr, *args, **kwargs):
        if isinstance(expr.aggregate, Field):
            return self.rec(expr.aggregate, *args, **kwargs,
                            outer_subscript=expr.index_tuple)
        else:
            return super().map_subscript(expr, *args, **kwargs)

    def map_lookup(self, expr, *args, **kwargs):
        return self.rec(pp.Variable(expr.name))


#: Appends subscripts to :class:`Field`
#: instances in an expression, turning them into ordinary
#: :class:`pymbolic.primitives.Subscript`'s.
#: See the documentation of :class:`Field` for examples.
#:
#: :arg expr: The :mod:`pymbolic` expression to be mapped.
#:
#: :arg prepend_with: A :class:`tuple` of indices to prepend to the subscript
#:  of any :class:`Field`'s in ``expr`` (unless a given :class:`Field` has
#:  :attr:ignore_prepends` set to *False*. Passed by keyword.
#:  Defaults to an empty :class:`tuple`.
#:
index_fields = IndexMapper()


class Shifter(IdentityMapper):
    def map_field(self, expr, shift=(0, 0, 0), *args, **kwargs):
        new_offset = tuple(o + s for o, s in zip(expr.offset, shift))
        return expr.new_with_changes(offset=new_offset)

    map_dynamic_field = map_field


shift_fields = Shifter()


class FieldCollector(Collector):
    def map_field(self, expr):
        return set([expr])

    map_dynamic_field = map_field


def get_field_args(expressions, unpadded_shape=None):
    """
    A :class:`pymbolic.mapper.Collector` which collects all
    :class:`~pystella.Field`'s from ``expressions`` and returns a corresponding
    list of :class:`loopy.ArrayArg`'s, using information about array indexing offsets
    to determine their shape.

    .. warning::

        This method currently does not correctly process
        :class:`~pystella.Field`'s which are subscripted (i.e., nested
        inside a :class:`pymbolic.primitives.Subscript`).
        That is, it disregards any information about outer axes as represented
        by subscripting.

    :arg expressions: The expressions from which to collect
        :class:`~pystella.Field`'s.
        May be one of the following:

            * A :class:`dict`, in which case all keys and values are iterated over.

            * A :class:`list`, in which case all elements are iterated over.

            * A :class:`pymbolic.primitives.Expression`.

    The following keyword arguments are recognized:

    :arg unpadded_shape: The shape of :class:`~pystella.Field`'s in ``expressions``
        (sans padding).
        Defaults to ``(Nx, Ny, Nz)``.

    :returns: A :class:`list` of :class:`loopy.ArrayArg`'s.

    Example::

        >>> f = Field('f', offset='h)
        >>> get_field_args(f)
        [<f: type: <auto/runtime>, shape: (Nx + 2*h, Ny + 2*h, Nz + 2*h)
        aspace: global>]
    """

    all_exprs = []
    if isinstance(expressions, dict):
        for k, v in expressions.items():
            all_exprs.append(k)
            all_exprs.append(v)
    elif isinstance(expressions, list):
        all_exprs = expressions
    else:
        all_exprs = [expressions]

    if unpadded_shape is None:
        unpadded_shape = parse('Nx, Ny, Nz')

    from loopy import GlobalArg

    fields = FieldCollector()(all_exprs)
    args = []
    for f in fields:
        shape = tuple(N + 2 * h for N, h in zip(unpadded_shape, f.base_offset))
        args.append(GlobalArg(f.child.name, shape=shape))

    return sorted(args, key=lambda f: f.name)


__all__ = [
    "Field",
    "DynamicField",
    "index_fields",
    "diff",
    "get_field_args",
    # "pymbolic_to_sympy",
    # "sympy_to_pymbolic",
    # "simplify",
]
