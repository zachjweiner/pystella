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

import loopy as lp
from pystella import ElementWiseMap


class Histogrammer(ElementWiseMap):
    """
    A subclass of :class:`ElementWiseMap` which computes (an arbitrary
    number of) histograms.

    .. automethod:: __init__
    .. automethod:: __call__
    """

    def parallelize(self, knl, lsize):
        knl = lp.split_iname(knl, "k", self.rank_shape[2],
                             outer_tag="g.0", inner_tag="l.0")
        knl = lp.split_iname(knl, "b", min(1024, self.num_bins),
                             outer_tag="g.0", inner_tag="l.0")
        knl = lp.tag_inames(knl, "j:g.1")
        return knl

    def __init__(self, decomp, bin_expr, weight_expr, num_bins, rank_shape, dtype,
                 **kwargs):
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

        The following keyword-only arguments are recognized (in addition to those
        accepted by :class:`ElementWiseMap`):

        :arg callback: A :class:`callable` used to process the reduction results
            before :meth:`__call__` returns.
            Defaults to ``lambda x: x``, i.e., doing nothing.
        """

        self.decomp = decomp
        self.rank_shape = rank_shape
        self.num_bins = num_bins

        from pymbolic import var
        bin_num = var('bin')
        b = var('b')
        hist = var('hist')
        temp = var('temp')
        temp_value = var('temp_value')

        args = kwargs.pop('args', [])
        args += [
            lp.TemporaryVariable("temp", dtype, shape=(self.num_bins,),
                                 for_atomic=True,
                                 address_space=lp.AddressSpace.LOCAL),
            lp.GlobalArg("hist", dtype, shape=(self.num_bins,), for_atomic=True),
        ]

        fixed_pars = kwargs.pop("fixed_parameters", dict())
        fixed_pars.update(dict(num_bins=num_bins))

        silenced_warnings = kwargs.pop("silenced_warnings", [])
        silenced_warnings += ['write_race(tmp)', 'write_race(glb)']

        domains = "[Nx, Ny, Nz, num_bins] -> \
           { [i,j,k,b]: 0<=i<Nx and 0<=j<Ny and 0<=k<Nz and 0<=b<num_bins}"

        insns = [
            lp.Assignment(hist[b], 0,
                          atomicity=(lp.AtomicUpdate(str(hist)),)),
            lp.BarrierInstruction('post_zero_barrier',
                                  synchronization_kind='global'),
            lp.Assignment(temp[b], 0,
                          within_inames=frozenset(('j', 'b')),
                          atomicity=(lp.AtomicUpdate(str(temp)),)),
            lp.Assignment(bin_num, bin_expr,
                          within_inames=frozenset(('i', 'j', 'k')),
                          temp_var_type=lp.Optional('int')),
            lp.Assignment(temp_value, weight_expr,
                          within_inames=frozenset(('i', 'j', 'k')),
                          temp_var_type=lp.Optional(dtype)),
            lp.Assignment(temp[bin_num], temp[bin_num] + temp_value, id='tmp',
                          within_inames=frozenset(('i', 'j', 'k')),
                          atomicity=(lp.AtomicUpdate(str(temp)),)),
            lp.Assignment(hist[b], hist[b] + temp[b], id='glb',
                          within_inames=frozenset(('j', 'b')),
                          atomicity=(lp.AtomicUpdate(str(hist)),))
        ]
        super().__init__(insns, rank_shape=rank_shape, args=args,
                         fixed_parameters=fixed_pars, domains=domains,
                         silenced_warnings=silenced_warnings,
                         **kwargs)

    def __call__(self, queue=None, filter_args=False, **kwargs):
        evt, (hist,) = super().__call__(queue=queue, filter_args=filter_args,
                                        **kwargs)
        full_hist = self.decomp.allreduce(hist.get())
        return full_hist
