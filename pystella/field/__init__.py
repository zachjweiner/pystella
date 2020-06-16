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
from pymbolic.mapper import Collector
import loopy as lp
from loopy.symbolic import (
    IdentityMapper as IdentityMapperBase,
    CombineMapper as CombineMapperBase,
    SubstitutionMapper as SubstitutionMapperBase)
from pymbolic.mapper.stringifier import StringifyMapper
from pystella.field.diff import diff
# from pystella.field.sympy import pymbolic_to_sympy, sympy_to_pymbolic, simplify

__doc__ = """
.. currentmodule:: pystella
.. autoclass:: Field
.. autoclass:: DynamicField
.. autofunction:: index_fields
.. autofunction:: shift_fields
.. autofunction:: diff
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
    (via :attr:`index_tuple`) by preprocessing
    expressions with :func:`index_fields`.

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

    .. attribute:: child

        The child expression representing the unsubscripted field.
        May be a string, a :class:`pymbolic.primitives.Variable`, or a
        :class:`pymbolic.primitives.Subscript`.

    .. attribute:: offset

        The amount of padding by which to offset the array axes
        corresponding to the elements of :attr:`indices`. May be a tuple with
        the same length as :attr:`indices` or a single value.
        In the latter case, the input is transformed into a tuple with the same
        length as :attr:`indices`, each with the same value.
        Defaults to ``0``.

    .. attribute:: shape

        The shape of axes preceding those indexed by `indices`.
        For example, ``Field('f', shape=(3, 'n'))``
        would correspond to an array with shape ``(3, n, Nx, Ny, Nz)``
        (using ``(Nx, Ny, Nz)`` as the shape along the final three axes
        indexed with ``indices``).
        Used by :meth:`get_field_args`.
        Defaults to an empty :class:`tuple`.

    .. attribute:: indices

        A tuple of (symbolic) array indices that will subscript
        the array.
        Each entry may be a :class:`pymbolic.primitives.Variable` or a string
        which parses to one.
        Defaults to ``('i', 'j', 'k')``

    .. attribute:: ignore_prepends

        Whether to ignore array subscripts prepended when
        processed with :func:`index_fields`. Useful for timestepping kernels
        (e.g., :class:`~pystella.step.RungeKuttaStepper`) which prepend array
        indices corresponding to extra storage axes (to specify that an array
        does not have this axis).
        Defaults to *False*.

    .. attribute:: base_offset

        The amount of padding by which to offset the array axes
        corresponding to the elements of :attr:`indices`.
        In contrast to :attr:`offset`, denotes the offset of an "unshifted"
        array access, so that this attribute is used in determining the
        fully-padded shape of the underlying array, while use of
        :func:`shift_fields` may specify offset array accesses by modifying
        :attr:`offset`.

    .. attribute:: dtype

        The datatype of the field.
        Defaults to *None*, in which case datatypes are inferred by :mod:`loopy`
        at kernel invocation.

    .. autoattribute:: index_tuple

    .. versionchanged:: 2020.2

        Added :attr:`dtype`.

    .. versionchanged:: 2020.1

        Added :attr:`shape`.
    """

    init_arg_names = ('child', 'offset', 'shape', 'indices',
                      'ignore_prepends', 'base_offset', 'dtype')

    def __init__(self, child, offset=0, shape=tuple(), indices=('i', 'j', 'k'),
                 ignore_prepends=False, base_offset=None, dtype=None):
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
        self.shape = shape
        self.ignore_prepends = ignore_prepends
        self.dtype = dtype

    def __getinitargs__(self):
        return (self.child, self.offset, self.shape, self.indices,
                self.ignore_prepends, self.base_offset)

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

    def copy(self, **kwargs):
        init_kwargs = dict(zip(self.init_arg_names, self.__getinitargs__()))
        init_kwargs.update(kwargs)
        return type(self)(**init_kwargs)


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
        :class:`Field`.
        Defaults to a :class:`Field` with name ``d{self.child}dt`` with the
        same :attr:`shape`, :attr:`indices`, and :attr:`offset`,
        but may be specified via the argument ``dot``.

    .. attribute:: lap

        A :class:`Field` representing the Laplacian of the base
        :class:`Field`.
        Defaults to a :class:`Field` with name ``lap_{self.child}`` with the
        same :attr:`shape` and :attr:`indices` but with zero :attr:`offset`,
        but may be specified via the argument ``lap``.

    .. attribute:: pd

        A :class:`Field` representing the spatial derivative(s) of the base
        :class:`Field`.
        Defaults to a :class:`Field` with name ``d{self.child}dx`` with shape
        ``(3,)+shape``, the same :attr:`indices`, and zero :attr:`offset`,
        but may be specified via the argument ``pd``.

    .. automethod:: d

    .. versionchanged:: 2020.1

        Specifying the names of :attr:`dot`, :attr:`lap`, and :attr:`pd` was
        replaced by passing actual :class:`Field` instances.
    """

    init_arg_names = ('child', 'offset', 'shape', 'indices', 'base_offset',
                      'dot', 'lap', 'pd', 'dtype')

    def __init__(self, child, offset='0', shape=tuple(), indices=('i', 'j', 'k'),
                 base_offset=None, dot=None, lap=None, pd=None, dtype=None):
        super().__init__(child, offset=offset, indices=indices,
                         base_offset=base_offset, shape=shape, dtype=dtype)

        self.dot = dot or Field('d%sdt' % str(child), shape=shape,
                                offset=offset, indices=indices, dtype=dtype)

        self.lap = lap or Field('lap_%s' % str(child), shape=shape,
                                offset=0, indices=indices, ignore_prepends=True,
                                dtype=dtype)

        self.pd = pd or Field('d%sdx' % str(child), shape=shape+(3,),
                              offset=0, indices=indices, ignore_prepends=True,
                              dtype=dtype)

    def __getinitargs__(self):
        return (self.child, self.offset, self.shape, self.indices, self.base_offset,
                self.dot, self.lap, self.pd)

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


class IdentityMapperMixin:
    def map_field(self, expr, *args, **kwargs):
        return expr.copy(
            child=self.rec(expr.child, *args, **kwargs),
            indices=self.rec(expr.indices, *args, **kwargs),
            offset=self.rec(expr.offset, *args, **kwargs),
        )

    def map_dynamic_field(self, expr, *args, **kwargs):
        return expr.copy(
            child=self.rec(expr.child, *args, **kwargs),
            indices=self.rec(expr.indices, *args, **kwargs),
            offset=self.rec(expr.offset, *args, **kwargs),
            dot=self.rec(expr.dot, *args, **kwargs),
            pd=self.rec(expr.pd, *args, **kwargs),
            lap=self.rec(expr.lap, *args, **kwargs),
        )

    def map_derivative(self, expr, *args, **kwargs):
        return type(expr)(
            self.rec(expr.child, *args, **kwargs),
            self.rec(expr.variables, *args, **kwargs))

    def map_dict(self, expr, *args, **kwargs):
        return dict(self.rec(list(expr.items()), *args, **kwargs))

    def map_assignment(self, expr, *args, **kwargs):
        return expr.copy(
            assignee=self.rec(expr.assignee, *args, **kwargs),
            expression=self.rec(expr.expression, *args, **kwargs),
        )

    def map_foreign(self, expr, *args, **kwargs):
        if isinstance(expr, dict):
            return self.map_dict(expr, *args, **kwargs)
        elif isinstance(expr, lp.Assignment):
            return self.map_assignment(expr, *args, **kwargs)
        elif isinstance(expr, lp.InstructionBase):
            return expr
        else:
            return super().map_foreign(expr, *args, **kwargs)


class IdentityMapper(IdentityMapperMixin, IdentityMapperBase):
    pass


class CombineMapperMixin:
    def map_field(self, expr, *args, **kwargs):
        return set()

    map_dynamic_field = map_field

    def map_derivative(self, expr, *args, **kwargs):
        return self.combine((
            self.rec(expr.child, *args, **kwargs),
            self.rec(expr.variables, *args, **kwargs)))

    def map_dict(self, expr, *args, **kwargs):
        return self.rec(list(expr.items()), *args, **kwargs)

    def map_assignment(self, expr, *args, **kwargs):
        return self.combine((
            self.rec(expr.assignee, *args, **kwargs),
            self.rec(expr.expression, *args, **kwargs)))

    def map_foreign(self, expr, *args, **kwargs):
        if isinstance(expr, dict):
            return self.map_dict(expr, *args, **kwargs)
        elif isinstance(expr, lp.Assignment):
            return self.map_assignment(expr, *args, **kwargs)
        elif isinstance(expr, lp.InstructionBase):
            return set()
        else:
            return super().map_foreign(expr, *args, **kwargs)


class CombineMapper(CombineMapperMixin, CombineMapperBase):
    pass


class IndexMapper(IdentityMapper):
    def map_lookup(self, expr, *args, **kwargs):
        return self.rec(pp.Variable(expr.name))

    def map_field(self, expr, *args, **kwargs):
        if expr.ignore_prepends:
            pre_index = ()
        else:
            prepend = kwargs.get('prepend_with') or ()
            pre_index = tuple(parse_if_str(x) for x in prepend)

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


def index_fields(expr, prepend_with=None):
    """
    Appends subscripts to :class:`Field`
    instances in an expression, turning them into ordinary
    :class:`pymbolic.primitives.Subscript`'s.
    See the documentation of :class:`Field` for examples.

    :arg expr: The expression(s) to be mapped.

    :arg prepend_with: A :class:`tuple` of indices to prepend to the subscript
        of any :class:`Field`'s in ``expr`` (unless a given :class:`Field` has
        :attr:`ignore_prepends` set to *False*. Passed by keyword.
        Defaults to an empty :class:`tuple`.

    .. versionadded:: 2020.1

        Replaced :func:`Indexer`.
    """

    return IndexMapper()(expr, prepend_with=prepend_with)


class Shifter(IdentityMapper):
    def map_field(self, expr, shift=(0, 0, 0), *args, **kwargs):
        new_offset = tuple(o + s for o, s in zip(expr.offset, shift))
        return expr.copy(offset=new_offset)

    map_dynamic_field = map_field


def shift_fields(expr, shift):
    """
    Returns an expression with all :class:`Field`'s shifted by ``shift``--i.e.,
    with ``shift`` added elementwise to each :class:`Field`'s ``offset`` attribute.

    :arg expr: The expression(s) to be mapped.

    :arg shift: A :class:`tuple`.

    .. versionadded:: 2020.1
    """

    return Shifter()(expr, shift=shift)


class SubstitutionMapper(IdentityMapperMixin, SubstitutionMapperBase):
    def map_algebraic_leaf(self, expr, *args, **kwargs):
        result = self.subst_func(expr)
        if result is not None:
            return result
        else:
            method = getattr(IdentityMapper, expr.mapper_method)
            return method(self, expr, *args, **kwargs)

    map_sum = map_algebraic_leaf
    map_product = map_algebraic_leaf
    map_quotient = map_algebraic_leaf
    map_floor_div = map_algebraic_leaf
    map_remainder = map_algebraic_leaf
    map_power = map_algebraic_leaf
    map_if = map_algebraic_leaf
    map_call = map_algebraic_leaf
    map_product = map_algebraic_leaf
    map_lookup = map_algebraic_leaf
    map_derivative = map_algebraic_leaf
    map_field = map_algebraic_leaf
    map_dynamic_field = map_algebraic_leaf
    map_reduction = map_algebraic_leaf


def substitute(expression, variable_assignments={}, **kwargs):
    variable_assignments = variable_assignments.copy()
    variable_assignments.update(kwargs)

    from pymbolic.mapper.substitutor import make_subst_func
    return SubstitutionMapper(make_subst_func(variable_assignments))(expression)


class FieldCollector(CombineMapper, Collector):
    def map_field(self, expr, *args, **kwargs):
        return set([expr])

    map_dynamic_field = map_field


def get_field_args(expressions, unpadded_shape=None, prepend_with=None):
    """
    Collects all :class:`~pystella.Field`'s from ``expressions`` and returns a
    corresponding list of :class:`loopy.ArrayArg`'s, using their ``offset``
    and ``shape`` attributes to determine their full shape.

    :arg expressions: The expressions from which to collect
        :class:`~pystella.Field`'s.

    The following keyword arguments are recognized:

    :arg unpadded_shape: The shape of :class:`~pystella.Field`'s in ``expressions``
        (sans padding).
        Defaults to ``(Nx, Ny, Nz)``.

    :arg prepend_with: A :class:`tuple` to prepend to the shape
        of any :class:`Field`'s in ``expressions`` (unless a given :class:`Field` has
        :attr:`ignore_prepends` set to *False*.
        Passed by keyword.
        Defaults to an empty :class:`tuple`.

    :returns: A :class:`list` of :class:`loopy.ArrayArg`'s.

    Example::

        >>> f, g = Field('f', offset='h'), Field('g', shape=(3, 'a'), offset=1)
        >>> get_field_args({f: g + 1})

    would return the equivalent of::

        >>> [lp.GlobalArg('f', shape='(Nx+2*h, Ny+2*h, Nz+2*h)', offset=lp.auto),
        ...  lp.GlobalArg('g', shape='(3, a, Nx+2, Ny+2, Nz+2)', offset=lp.auto)]

    .. versionchanged:: 2020.1

        Uses :attr:`Field.shape` to determine the full array shape.
    """

    from pymbolic import parse
    unpadded_shape = unpadded_shape or parse('Nx, Ny, Nz')

    fields = FieldCollector()(expressions)

    field_args = {}
    for f in fields:
        spatial_shape = \
            tuple(N + 2 * h for N, h in zip(unpadded_shape, f.base_offset))
        full_shape = f.shape + spatial_shape

        if prepend_with is not None and not f.ignore_prepends:
            full_shape = prepend_with + full_shape

        if full_shape == tuple():
            arg = lp.ValueArg(f.name)
        else:
            arg = lp.GlobalArg(f.name, shape=full_shape, offset=lp.auto,
                               dtype=f.dtype)

        if f.name in field_args:
            other_arg = field_args[f.name]
            if arg.shape != other_arg.shape:
                raise ValueError(
                    "Encountered instances of field '%s' with conflicting shapes"
                    % f.name)
        else:
            field_args[f.name] = arg

    return list(sorted(field_args.values(), key=lambda f: f.name))


def collect_field_indices(expressions):
    fields = FieldCollector()(expressions)

    all_indices = set()
    for f in fields:
        all_indices |= set(f.indices)

    def get_name(expr):
        try:
            return expr.name
        except AttributeError:
            return str(expr)

    all_indices = sorted(tuple(get_name(i) for i in all_indices))

    return set(all_indices)


def indices_to_domain(indices):
    constraints = " and ".join("0 <= %s < N%s" % (idx, idx) for idx in indices)
    domain = "{[%s]: %s}" % (",".join(indices), constraints)
    return domain


def infer_field_domains(expressions):
    all_indices = collect_field_indices(expressions)
    return indices_to_domain(all_indices)


__all__ = [
    "Field",
    "DynamicField",
    "index_fields",
    "shift_fields",
    "substitute",
    "get_field_args",
    "collect_field_indices",
    "indices_to_domain",
    "infer_field_domains",
    "diff",
    # "pymbolic_to_sympy",
    # "sympy_to_pymbolic",
    # "simplify",
]
