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


import numpy as np
import pyopencl.array as cla  # noqa: F401
import loopy as lp
from pystella.elementwise import ElementWiseMap

from warnings import filterwarnings
from loopy.diagnostic import ParameterFinderWarning
filterwarnings('ignore', category=ParameterFinderWarning)

__doc__ = """
.. currentmodule:: pystella
.. autoclass:: Reduction
"""


def get_mpi_reduction_op(op):
    from mpi4py import MPI
    _MPI_REDUCTION_OPS = {
        "avg": MPI.SUM,
        "sum": MPI.SUM,
        "prod": MPI.PROD,
        "max": MPI.MAX,
        "min": MPI.MIN,
        }

    if op in _MPI_REDUCTION_OPS:
        return _MPI_REDUCTION_OPS[op]
    else:
        raise NotImplementedError('MPI allreduce for operation %s' % op)


def get_numpy_reduction_op(op):
    _NUMPY_REDUCTION_OPS = {
        "avg": np.sum,
        "sum": np.sum,
        "prod": np.prod,
        "max": np.max,
        "min": np.min,
        }

    if op in _NUMPY_REDUCTION_OPS:
        return _NUMPY_REDUCTION_OPS[op]
    else:
        raise NotImplementedError('numpy reduction for operation %s' % op)


def get_cl_reduction_op(op):
    _CL_REDUCTION_OPS = {
        "avg": cla.sum,
        "sum": cla.sum,
        "max": cla.max,
        "min": cla.min,
        }

    if op in _CL_REDUCTION_OPS:
        return _CL_REDUCTION_OPS[op]
    else:
        raise NotImplementedError('pyopencl reduction for operation %s' % op)


class Reduction(ElementWiseMap):
    """
    A subclass of :class:`ElementWiseMap` which computes (an arbitrary
    number of) reductions.

    .. automethod:: __init__
    .. automethod:: __call__
    """

    def parallelize(self, knl, lsize):
        knl = lp.split_iname(knl, "k", lsize[0], outer_tag="g.0", inner_tag="l.0")
        knl = lp.split_iname(knl, "j", lsize[1], outer_tag="g.1", inner_tag="ilp")
        return knl

    def __init__(self, decomp, input, **kwargs):
        """
        :arg decomp: An instance of :class:`DomainDecomposition`.

        :arg input: May be one of the following:

            * a :class:`dict`. The output of :meth:`__call__` will be a dictionary
              with the same keys whose values are the corresponding reductions
              of the :class:`dict`'s values. The values may either be lists of
              :mod:`pymbolic` expressions or a lists of :class:`tuple`'s
              ``(expr, op)``, where ``expr`` is a :mod:`pymbolic` expression and
              ``op`` is the reduction operation to perform. Valid options are
              ``'avg'`` (default), ``'sum'``, ``'prod'``, ``'max'``, and ``'min'``.

            * a :class:`Sector`. In this case, the reduction dictionary will be
              obtained from :attr:`Sector.reducers`.

            * a :class:`list` of :class:`Sector`'s. In this case, the input
              obtained from each :class:`Sector` (as described above) will be
              combined.

        The following keyword-only arguments are recognized (in addition to those
        accepted by :class:`ElementWiseMap`):

        :arg grid_size: The total number of gridpoints on the entire computational
            grid.
            Defaults to *None*, in which case it will be inferred at
            :meth:`__call__` (if averages are being performed).

        :arg callback: A :class:`callable` used to process the reduction results
            before :meth:`__call__` returns.
            Defaults to ``lambda x: x``, i.e., doing nothing.
        """

        self.decomp = decomp
        from pystella import Sector
        if isinstance(input, Sector):
            self.reducers = input.reducers
        elif isinstance(input, list):
            self.reducers = dict(i for s in input for i in s.reducers.items())
        elif isinstance(input, dict):
            self.reducers = input
        else:
            raise NotImplementedError

        reducers = self.reducers
        self.grid_size = kwargs.pop('grid_size', None)
        self.callback = kwargs.pop('callback', lambda x: x)

        self.num_reductions = sum(len(i) for i in reducers.values())

        from pymbolic import var
        tmp = var('tmp')
        self.tmp_dict = {}
        i = 0
        for key, val in reducers.items():
            inext = i + len(val)
            self.tmp_dict[key] = range(i, inext)
            i = inext

        # flatten and process inputs into expression and operation
        flat_reducers = []
        reduction_ops = []
        for val in reducers.values():
            for v in val:
                if isinstance(v, tuple):
                    flat_reducers.append(v[0])
                    reduction_ops.append(v[1])
                else:
                    flat_reducers.append(v)
                    reduction_ops.append('avg')
        self.reduction_ops = reduction_ops

        def reduction(expr, op):
            return lp.symbolic.Reduction(operation=op, inames=('i',), expr=expr,
                                         allow_simultaneous=True)

        statements = [
            (tmp[i, var('j'), var('k')],
             reduction(expr, 'sum' if op == 'avg' else op))
            for i, (expr, op) in enumerate(zip(flat_reducers, reduction_ops))
        ]
        statements += [(var('Nx_'), var('Nx'))]

        args = [lp.GlobalArg('Nx_', shape=(), dtype='int')]
        args += kwargs.pop('args', [...])

        super().__init__(statements, **kwargs, args=args, seq_dependencies=False,
                         lsize=(32, 2, 1), options=lp.Options(return_dict=True))

    def reduce_array(self, arr, op):
        if op == 'prod':
            np_op = get_numpy_reduction_op(op)
            rank_sum = np_op(arr.get())
        else:
            cl_op = get_cl_reduction_op(op)
            rank_sum = cl_op(arr).get()

        if self.decomp.comm is not None:
            mpi_op = get_mpi_reduction_op(op)
            return self.decomp.allreduce(rank_sum, op=mpi_op)
        else:
            return rank_sum

    def __call__(self, queue, filter_args=False, **kwargs):
        """
        Performs reductions by calling :attr:`knl` and
        :meth:`DomainDecomposition.allreduce`.

        :arg queue: The :class:`pyopencl.CommandQueue` on which to enqueue the
            kernel.
            If *None*, ``queue`` is not passed (i.e., for
            :class:`loopy.ExecutableCTarget`)

        The following keyword arguments are recognized:

        :arg filter_args: Whether to filter ``kwargs`` such that no unexpected
            arguments are passed to the :attr:`knl`. Defaults to *False*.

        :arg allocator: A :mod:`pyopencl` allocator used to allocate temporary
            arrays, i.e., most usefully a :class:`pyopencl.tools.MemoryPool`.
            See the note in the documentation of
            :meth:`SpectralCollocator`.

        The remaining keyword arguments are passed to :attr:`knl`.

        :returns: A :class:`dict` with the same keys as (interpreted from) ``input``
            whose values are the corresponding (lists of) reduced values.
            Averages are obtained by dividing by :attr:`grid_size`.
            If ``grid_size`` was not supplied at :meth:`__init__`, it is inferred
            (at a slight performance penalty).
        """

        evt, output = super().__call__(queue, filter_args=filter_args, **kwargs)
        tmp = output['tmp']
        vals = {}
        for key, sub_indices in self.tmp_dict.items():
            reductions = []
            for j in sub_indices:
                op = self.reduction_ops[j]
                val = self.reduce_array(tmp[j], op)
                if op == 'avg':
                    if self.grid_size is None:
                        Nx = output['Nx_'].get()
                        sub_grid_size = Nx * np.product(tmp[j].shape)
                        grid_size = self.decomp.allreduce(sub_grid_size)
                    else:
                        grid_size = self.grid_size
                    val /= grid_size
                reductions.append(val)
            vals[key] = np.array(reductions)
        return self.callback(vals)


class FieldStatistics(Reduction):
    """
    A subclass of :class:`Reduction` which computes the mean and variance of
    fields.

    .. automethod:: __init__
    .. automethod:: __call__
    """

    def __init__(self, decomp, halo_shape, **kwargs):
        """
        :arg decomp: An instance of :class:`DomainDecomposition`.

        :arg halo_shape: The number of halo layers on (both sides of) each axis of
            the computational grid.
            May either be an :class:`int`, interpreted as a value to fix the
            parameter ``h`` to, or a :class:`tuple`, interpreted as values for
            ``hx``, ``hy``, and ``hz``.
            Defaults to *None*, in which case no such values are fixed at kernel
            creation.

        The following keyword-only arguments are recognized:

        :arg max_min: A :class:`bool` determining whether to also compute the
            actual and absolute maxima and minima of fields.
            Defaults to *False*.

        Any remaining keyword arguments are passed to :meth:`Reduction.__init__`.
        """

        self.min_max = kwargs.pop('max_min', False)

        from pystella import Field
        f = Field('f', offset='h')
        reducers = {}
        reducers['mean'] = [f]
        reducers['variance'] = [f**2]
        if self.min_max:
            reducers['max'] = [(f, 'max')]
            reducers['min'] = [(f, 'min')]
            # from pymbolic.functions import fabs
            from pymbolic import var
            fabs = var('fabs')
            reducers['abs_max'] = [(fabs(f), 'max')]
            reducers['abs_min'] = [(fabs(f), 'min')]
        self.reducers = reducers

        super().__init__(decomp, reducers, halo_shape=halo_shape, **kwargs)

    def __call__(self, f, queue=None, allocator=None):
        """
        :arg f: The array whose statistics will be computed.
            If ``f`` has more than three axes, all the outer axes are looped over.
            As an example, if ``f`` has shape ``(2, 3, 130, 130, 130)``,
            this method loops over the outermost two axes with shape ``(2, 3)``, and
            the resulting output data would have the same shape.

        The following keyword arguments are recognized:

        :arg queue: A :class:`pyopencl.CommandQueue`.
            Defaults to ``fx.queue``.

        :arg allocator: A :mod:`pyopencl` allocator used to allocate temporary
            arrays, i.e., most usefully a :class:`pyopencl.tools.MemoryPool`.
            See the note in the documentation of
            :meth:`SpectralCollocator`.

        :returns: A :class:`dict` of means and variances, whose values are lists
            of the statistic (key) for each array in ``fields``.
        """

        queue = queue or f.queue

        outer_shape = f.shape[:-3]
        from itertools import product
        slices = list(product(*[range(n) for n in outer_shape]))

        out = {k: np.zeros(outer_shape) for k in self.reducers.keys()}
        for s in slices:
            stats = super().__call__(queue, f=f[s], allocator=allocator)
            for k, v in stats.items():
                if k == 'variance':
                    out[k][s] = stats['variance'][0] - stats['mean'][0]**2
                else:
                    out[k][s] = v[0]

        out = {k: np.array(v) for k, v in out.items()}

        return out
