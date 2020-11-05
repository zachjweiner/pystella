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
import loopy as lp
from pystella import ElementWiseMap, Reduction, Field

__doc__ = """
.. currentmodule:: pystella
.. autoclass:: Histogrammer
"""


class Histogrammer(ElementWiseMap):
    """
    A subclass of :class:`ElementWiseMap` which computes (an arbitrary
    number of) histograms.

    :arg decomp: An instance of :class:`DomainDecomposition`.

    :arg histograms: A :class:`dict` with values of the form
        ``(bin_expr, weight_expr)``, which are :mod:`pymbolic` expressions
        whose result determines the bin number and the associated weight
        contributed to that bin's count, respectively.
        The output of :meth:`__call__` will be a dictionary
        with the same keys whose values are the specified histograms.

        .. note::

            The values computed by ``bin_expr`` will by default be cast to
            integers by truncation. To instead round to the nearest integer,
            wrap the expression in a call to ``round``.

    :arg num_bins: The number of bins of the computed histograms.

    :arg dtype: The datatype of the resulting histogram.

    In addition, any keyword-only arguments accepted by :class:`ElementWiseMap`
    are also recognized.

    .. versionadded:: 2020.1

    .. versionchanged:: 2020.2

        Positional argument ``rank_shape`` no longer required.

    .. automethod:: __call__
    """

    def parallelize(self, knl, lsize):
        # global hist is zeroed in its own kernel
        knl = lp.split_iname(knl, "bb", lsize[0], outer_tag="g.0", inner_tag="l.0",
                             within="id:zero_hist*")
        # outer_tag of loops writing/reading temp_hist cannot be a global index
        knl = lp.split_iname(knl, "bb", lsize[0], inner_tag="l.0",
                             within="id:zero_temp*")
        knl = lp.split_iname(knl, "k", lsize[0], inner_tag="l.0",
                             within="not id:zero* and not id:glb*")
        knl = lp.split_iname(knl, "b", lsize[0], inner_tag="l.0",
                             within="id:glb*")
        knl = lp.tag_inames(knl, "j:g.0")
        return knl

    def __init__(self, decomp, histograms, num_bins, dtype, **kwargs):
        self.decomp = decomp
        self.histograms = histograms
        self.num_bins = num_bins
        num_hists = len(histograms)

        from pymbolic import var
        _bin = var('bin')
        b = var('b')
        bb = var('bb')
        hist = var('hist')
        temp = var('temp')
        weight_val = var('weight')

        args = kwargs.pop('args', [])
        args += [
            lp.TemporaryVariable("temp", dtype, shape=(num_hists, self.num_bins,),
                                 for_atomic=True,
                                 address_space=lp.AddressSpace.LOCAL),
            lp.TemporaryVariable("bin", 'int', shape=(num_hists,)),
            lp.TemporaryVariable("weight", dtype, shape=(num_hists,)),
            lp.GlobalArg("hist", dtype, shape=(num_hists, self.num_bins,),
                         for_atomic=True),
        ]

        fixed_pars = kwargs.pop("fixed_parameters", dict())
        fixed_pars.update(dict(num_bins=num_bins, num_hists=num_hists))

        silenced_warnings = kwargs.pop("silenced_warnings", [])
        silenced_warnings += ['write_race(tmp*)', 'write_race(glb*)']

        domains = """
        [Nx, Ny, Nz, num_bins] ->
           {[i, j, k, b, bb]: 0<=i<Nx and 0<=j<Ny and 0<=k<Nz and 0<=b<num_bins
                              and 0<=bb<num_bins}
        """

        insns = [
            lp.Assignment(
                hist[j, bb], 0,
                id=f'zero_hist_{j}', within_inames=frozenset('bb'),
                atomicity=(lp.AtomicInit(str(hist)),)
            )
            for j in range(num_hists)
        ]
        insns.append(
            lp.BarrierInstruction('post_zero_barrier', synchronization_kind='global')
        )
        insns.extend([
            lp.Assignment(
                temp[j, bb], 0,
                id=f'zero_temp_{j}', within_inames=frozenset(('j', 'bb')),
                atomicity=(lp.AtomicInit(str(temp)),)
            )
            for j in range(num_hists)
        ])
        for j, (bin_expr, weight_expr) in enumerate(histograms.values()):
            insns.extend([
                lp.Assignment(
                    _bin[j], var('floor')(bin_expr),
                    id=f"set_bin_{j}", within_inames=frozenset(('i', 'j', 'k'))
                ),
                lp.Assignment(
                    weight_val[j], weight_expr,
                    id=f"set_weight_{j}", within_inames=frozenset(('i', 'j', 'k'))
                ),
                lp.Assignment(
                    temp[j, _bin[j]], temp[j, _bin[j]] + weight_val[j],
                    id=f'tmp_{j}', within_inames=frozenset(('i', 'j', 'k')),
                    atomicity=(lp.AtomicUpdate(str(temp)),)
                )
            ])

        insns.extend([
            lp.Assignment(
                hist[j, b], hist[j, b] + temp[j, b],
                id=f'glb_{j}', within_inames=frozenset(('j', 'b')),
                atomicity=(lp.AtomicUpdate(str(hist)),)
            )
            for j in range(num_hists)
        ])

        lsize = [min(256, self.num_bins)]

        super().__init__(insns, args=args, lsize=lsize,
                         fixed_parameters=fixed_pars, domains=domains,
                         silenced_warnings=silenced_warnings, **kwargs)

    def __call__(self, queue=None, filter_args=False, **kwargs):
        """
        Computes histograms by calling :attr:`knl` and
        :meth:`DomainDecomposition.allreduce`.

        In addition to the arguments required by the actual kernel
        (passed by keyword only), the following keyword arguments are recognized:

        :arg queue: The :class:`pyopencl.CommandQueue` on which to enqueue the
            kernel.
            Defaults to *None*, in which case ``queue`` is not passed (i.e., for
            :class:`loopy.ExecutableCTarget`)

        :arg filter_args: Whether to filter ``kwargs`` such that no unexpected
            arguments are passed to the :attr:`knl`.
            Defaults to *False*.

        :arg allocator: A :mod:`pyopencl` allocator used to allocate temporary
            arrays, i.e., most usefully a :class:`pyopencl.tools.MemoryPool`.
            See the note in the documentation of
            :meth:`SpectralCollocator`.

        Any remaining keyword arguments are passed to :attr:`knl`.

        :returns: A :class:`dict` with the same keys as the input
            whose values are the corresponding histograms.
        """

        evt, (hist,) = super().__call__(queue, filter_args=filter_args,
                                        **kwargs)
        full_hist = self.decomp.allreduce(hist.get())

        result = dict()
        for j, name in enumerate(self.histograms.keys()):
            result[name] = full_hist[j]

        return result


class FieldHistogrammer(Histogrammer):
    """
    A subclass of :class:`Histogrammer` which computes field histograms with
    both linear and logarithmic binning.

    :arg decomp: An instance of :class:`DomainDecomposition`.

    :arg num_bins: The number of bins of the computed histograms.

    :arg dtype: The datatype of the resulting histogram.

    The following keyword-only arguments are recognized (in addition to those
    accepted by :class:`ElementWiseMap`):

    :arg halo_shape: The number of halo layers on (both sides of) each axis of
        the computational grid.
        May either be an :class:`int`, interpreted as a value to fix the
        parameter ``h`` to, or a :class:`tuple`, interpreted as values for
        ``hx``, ``hy``, and ``hz``.
        Defaults to ``0``, i.e., no padding.

    .. versionadded:: 2020.1

    .. versionchanged:: 2020.2

        Positional argument ``rank_shape`` no longer required.

    .. automethod:: __call__
    """

    def __init__(self, decomp, num_bins, dtype, **kwargs):
        from pymbolic import parse
        import pymbolic.functions as pf

        max_f, min_f = parse('max_f, min_f')
        max_log_f, min_log_f = parse('max_log_f, min_log_f')

        halo_shape = kwargs.pop('halo_shape', 0)
        f = Field('f', offset=halo_shape)

        def clip(expr):
            _min, _max = parse('min, max')
            return _max(_min(expr, num_bins - 1), 0)

        linear_bin = (f - min_f) / (max_f - min_f)
        log_bin = (pf.log(pf.fabs(f)) - min_log_f) / (max_log_f - min_log_f)
        histograms = {
            'linear': (clip(linear_bin * num_bins), 1),
            'log': (clip(log_bin * num_bins), 1)
        }

        super().__init__(decomp, histograms, num_bins, dtype, **kwargs)

        reducers = {}
        reducers['max_f'] = [(f, 'max')]
        reducers['min_f'] = [(f, 'min')]
        reducers['max_log_f'] = [(pf.log(pf.fabs(f)), 'max')]
        reducers['min_log_f'] = [(pf.log(pf.fabs(f)), 'min')]

        self.get_min_max = Reduction(decomp, reducers, halo_shape=halo_shape,
                                     **kwargs)

    def __call__(self, f, queue=None, **kwargs):
        """
        :arg f: The array whose histograms will be computed.
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

        In addition, the keyword arguments ``min_f``, ``max_f``, ``min_log_f``,
        and ``max_log_f`` are recognized and used to define binning.
        Each must have the same shape as the outer axes of ``f`` (e.g.,
        ``(2, 3)`` in the example above).
        Unless values for each of these is passed, they all will be computed
        automatically.

        .. warning::

            This class prevents any out-of-bounds writes when calculating
            the bin number, ensuring that ``0 <= bin < num_bins``.
            When passing minimum and maximum values, the first and last
            bins may be incorrect if ``f`` in fact has values outside
            of the passed minimum/maximum values.

        :returns: A :class:`dict` with the the following items:

            * ``'linear'``: the histogram(s) of ``f`` with linear binning

            * ``'linear_bins'``: the bins used for the linear histogram(s) of ``f``

            * ``'log'``: the histogram(s) of ``f`` with logarithmic binning

            * ``'log_bins'``: the bins used for the logarithmic histogram(s) of
              ``f``

            Each :mod:`numpy` array has shape ``f.shape[:-3] + (num_bins,)``.
        """

        queue = queue or f.queue

        outer_shape = f.shape[:-3]
        from itertools import product
        slices = list(product(*[range(n) for n in outer_shape]))

        min_max_kwargs = set(self.get_min_max.reducers.keys())
        bounds_passed = min_max_kwargs.issubset(set(kwargs.keys()))

        out = dict()
        for key in ('linear', 'log'):
            out[key] = np.zeros(outer_shape+(self.num_bins,))
            out[key+'_bins'] = np.zeros(outer_shape+(self.num_bins+1,))

        for s in slices:
            if not bounds_passed:
                bounds = self.get_min_max(queue, f=f[s])
                bounds = {key: val[0] for key, val in bounds.items()}
            else:
                bounds = {key: kwargs[key][s] for key in min_max_kwargs}

            hists = super().__call__(queue, f=f[s], **bounds)
            for key, val in hists.items():
                out[key][s] = val

            out['linear_bins'][s] = np.linspace(bounds['min_f'], bounds['max_f'],
                                                self.num_bins+1)
            out['log_bins'][s] = np.exp(np.linspace(bounds['min_log_f'],
                                                    bounds['max_log_f'],
                                                    self.num_bins+1))

        return out
