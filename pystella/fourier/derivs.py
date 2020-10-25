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
import pyopencl.array as cla


class SpectralCollocator:
    """
    Interface (analagous to :class:`~pystella.FiniteDifferencer`)
    for computing spatial derivatives via spectral collocation.

    The following arguments are required:

    :arg fft: An FFT object as returned by :func:`~pystella.DFT`.
        ``grid_shape`` and ``dtype`` are determined by ``fft``'s attributes.

    :arg dk: A 3-:class:`tuple` of the momentum-space grid spacing of each
        axis (i.e., the infrared cutoff of the grid in each direction).

    .. automethod:: __call__
    .. automethod:: divergence
    """

    def __init__(self, fft, dk):
        self.fft = fft
        grid_size = fft.grid_shape[0] * fft.grid_shape[1] * fft.grid_shape[2]

        queue = self.fft.sub_k['momenta_x'].queue
        sub_k = list(x.get().astype('int') for x in self.fft.sub_k.values())
        k_names = ('k_x', 'k_y', 'k_z')
        self.momenta = {}
        for mu, (name, kk) in enumerate(zip(k_names, sub_k)):
            kk_mu = dk[mu] * kk.astype(fft.rdtype)
            self.momenta[name+'_2'] = cla.to_device(queue, kk_mu)

            # zero Nyquist mode for first derivatives
            kk_mu[abs(sub_k[mu]) == fft.grid_shape[mu]//2] = 0.
            kk_mu[sub_k[mu] == 0] = 0.
            self.momenta[name+'_1'] = cla.to_device(queue, kk_mu)

        args = [
            lp.GlobalArg('fk', shape="(Nx, Ny, Nz)"),
            lp.GlobalArg("k_x_1, k_x_2", fft.rdtype, shape=('Nx',)),
            lp.GlobalArg("k_y_1, k_y_2", fft.rdtype, shape=('Ny',)),
            lp.GlobalArg("k_z_1, k_z_2", fft.rdtype, shape=('Nz',)),
        ]

        from pystella.field import Field
        fk = Field('fk')
        pd = tuple(Field(pdi) for pdi in ('pdx_k', 'pdy_k', 'pdz_k'))

        indices = fk.indices

        from pymbolic import var
        mom_vars = tuple(var(name+'_1') for name in k_names)

        fk_tmp = var('fk_tmp')
        tmp_insns = [(fk_tmp, fk * (1/grid_size))]

        pdx, pdy, pdz = ({pdi: kk_i[indices[i]] * 1j * fk_tmp}
                         for i, (pdi, kk_i) in enumerate(zip(pd, mom_vars)))

        pdx_incr, pdy_incr, pdz_incr = (
            {Field('div'): Field('div') + kk_i[indices[i]] * 1j * fk_tmp}
            for i, kk_i in enumerate(mom_vars)
        )

        mom_vars = tuple(var(name+'_2') for name in k_names)
        kmag_sq = sum(kk_i[x_i]**2 for kk_i, x_i in zip(mom_vars, indices))
        lap = {Field('lap_k'): - kmag_sq * fk_tmp}

        from pystella.elementwise import ElementWiseMap
        common_args = dict(halo_shape=0, args=args, lsize=(16, 2, 1),
                           tmp_instructions=tmp_insns,
                           options=lp.Options(return_dict=True))
        self.pdx_knl = ElementWiseMap(pdx, **common_args)
        self.pdy_knl = ElementWiseMap(pdy, **common_args)
        self.pdz_knl = ElementWiseMap(pdz, **common_args)
        self.pdx_incr_knl = ElementWiseMap(pdx_incr, **common_args)
        self.pdy_incr_knl = ElementWiseMap(pdy_incr, **common_args)
        self.pdz_incr_knl = ElementWiseMap(pdz_incr, **common_args)
        self.lap_knl = ElementWiseMap(lap, **common_args)

        common_args['lsize'] = (16, 1, 1)
        self.grad_knl = ElementWiseMap({**pdx, **pdy, **pdz}, **common_args)
        self.grad_lap_knl = ElementWiseMap({**pdx, **pdy, **pdz, **lap},
                                           **common_args)

    def __call__(self, queue, fx, *,
                 lap=None, pdx=None, pdy=None, pdz=None, grd=None,
                 allocator=None):
        """
        Computes requested derivatives of the input ``fx``.
        Provides the same interface as
        :meth:`pystella.FiniteDifferencer.__call__`, while additionally accepting
        the following arguments:

        :arg allocator: A :mod:`pyopencl` allocator used to allocate temporary
            arrays, i.e., most usefully a :class:`pyopencl.tools.MemoryPool`.

        .. note::

            This method allocates extra temporary arrays
            (when computing more than one derivative), since in-place DFTs
            are not yet supported.
            Passing a :class:`pyopencl.tools.MemoryPool` is highly recommended to
            amortize the cost of memory allocation at each invocation, e.g.::

                >>> derivs = SpectralCollocator(fft, dk)
                >>> import pyopencl.tools as clt
                >>> pool = clt.MemoryPool(clt.ImmediateAllocator(queue))
                >>> derivs(queue, fx, lap=lap, allocator=pool)
        """

        from itertools import product
        slices = list(product(*[range(n) for n in fx.shape[:-3]]))

        if grd is not None:
            if isinstance(grd, (tuple, list)):
                pdx, pdy, pdz = grd
            else:
                pdx = grd[..., 0, :, :, :]
                pdy = grd[..., 1, :, :, :]
                pdz = grd[..., 2, :, :, :]

        for s in slices:
            fk = self.fft.dft(fx[s])
            arguments = {'queue': queue, 'fk': fk,
                         **self.momenta, 'allocator': allocator}

            if (lap is not None and pdx is not None
                    and pdy is not None and pdz is not None):
                evt, out = self.grad_lap_knl(**arguments, lap_k=fk)
            elif pdx is not None and pdy is not None and pdz is not None:
                evt, out = self.grad_knl(**arguments, pdx_k=fk, filter_args=True)
            elif lap is not None:
                evt, out = self.lap_knl(**arguments, lap_k=fk, filter_args=True)
            elif pdx is not None:
                evt, out = self.pdx_knl(**arguments, pdx_k=fk, filter_args=True)
            elif pdy is not None:
                evt, out = self.pdy_knl(**arguments, pdy_k=fk, filter_args=True)
            elif pdz is not None:
                evt, out = self.pdz_knl(**arguments, pdz_k=fk, filter_args=True)

            if 'lap_k' in out:
                self.fft.idft(out['lap_k'], lap[s])
            if 'pdx_k' in out:
                self.fft.idft(out['pdx_k'], pdx[s])
            if 'pdy_k' in out:
                self.fft.idft(out['pdy_k'], pdy[s])
            if 'pdz_k' in out:
                self.fft.idft(out['pdz_k'], pdz[s])

        return None

    def divergence(self, queue, vec, div, allocator=None):
        """
        Computes the divergence of the input ``vec``.
        Provides the same interface as
        :meth:`pystella.FiniteDifferencer.divergence`, while additionally accepting
        the following arguments:

        :arg allocator: A :mod:`pyopencl` allocator used to allocate temporary
            arrays, i.e., most usefully a :class:`pyopencl.tools.MemoryPool`.
        """

        from itertools import product
        slices = list(product(*[range(n) for n in vec.shape[:-4]]))

        for s in slices:
            arguments = {'queue': queue, **self.momenta, 'allocator': allocator}

            fk = self.fft.dft(vec[s][0])
            evt, out = self.pdx_knl(fk=fk, **arguments, filter_args=True)
            div_k = out['pdx_k']
            fk = self.fft.dft(vec[s][1])
            evt, _ = self.pdy_incr_knl(fk=fk, **arguments, div=div_k,
                                       filter_args=True)
            fk = self.fft.dft(vec[s][2])
            evt, _ = self.pdz_incr_knl(fk=fk, **arguments, div=div_k,
                                       filter_args=True)
            self.fft.idft(div_k, div[s])

        return None
