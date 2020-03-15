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


__doc__ = """
.. currentmodule:: pystella
.. autoclass:: Projector
"""


class Projector:
    """
    Constructs kernels to projector vectors to and from their polarization basis
    and to project out longitudinal modes, and to project a tensor field to its
    transverse and traceless component.

    .. automethod:: __init__
    .. automethod:: transversify
    .. automethod:: pol_to_vec
    .. automethod:: vec_to_pol
    .. automethod:: transverse_traceless
    """

    def __init__(self, fft, effective_k):
        """
        :arg fft: An FFT object as returned by :func:`DFT`.
            ``grid_shape`` and ``dtype`` are determined by ``fft``'s attributes.

        :arg effective_k: A :class:`callable` with signature ``(k, dx)`` returning
            the effective momentum of the corresponding stencil :math:`\\Delta`,
            i.e., :math:`k_\\mathrm{eff}` such that
            :math:`\\Delta e^{i k x} = i k_\\mathrm{eff} e^{i k x}`.
            That is, projections are implemented relative to the stencil
            whose eigenvalues (divided by :math:`i`) are returned by this function.
        """

        self.fft = fft

        if not callable(effective_k):
            if effective_k != 0:
                from pystella.derivs import FirstCenteredDifference
                h = effective_k
                effective_k = FirstCenteredDifference(h).get_eigenvalues
            else:
                def effective_k(k, dx):  # pylint: disable=function-redefined
                    return k

        from math import pi
        grid_shape = fft.grid_shape
        # since projectors only need the unit momentum vectors, can pass
        # k = k_hat * dk * dx = k_hat * 2 * pi * grid_shape and dx = 1,
        # where k_hat is the integer momentum gridpoint
        dk_dx = tuple(2 * pi / Ni for Ni in grid_shape)

        queue = self.fft.sub_k['momenta_x'].queue
        sub_k = list(x.get().astype('int') for x in self.fft.sub_k.values())
        eff_mom_names = ('eff_mom_x', 'eff_mom_y', 'eff_mom_z')
        self.eff_mom = {}
        for mu, (name, kk) in enumerate(zip(eff_mom_names, sub_k)):
            eff_k = effective_k(kk.astype(fft.dtype) * dk_dx[mu], 1)
            eff_k[abs(sub_k[mu]) == fft.grid_shape[mu]//2] = 0.
            eff_k[sub_k[mu] == 0] = 0.

            import pyopencl.array as cla
            self.eff_mom[name] = cla.to_device(queue, eff_k)

        from pymbolic import var, parse
        from pymbolic.primitives import If, Comparison, LogicalAnd
        from pystella import Field
        indices = parse('i, j, k')
        eff_k = tuple(var(array)[mu] for array, mu in zip(eff_mom_names, indices))
        fabs, sqrt, conj = parse('fabs, sqrt, conj')
        kmag = sqrt(sum(kk**2 for kk in eff_k))

        from pystella import ElementWiseMap
        vector = Field('vector', shape=(3,))
        vector_T = Field('vector_T', shape=(3,))

        kvec_zero = LogicalAnd(
            tuple(Comparison(fabs(eff_k[mu]), '<', 1e-14) for mu in range(3))
        )

        import loopy as lp

        def assign(asignee, expr, **kwargs):
            default = dict(within_inames=frozenset(('i', 'j', 'k')),
                           no_sync_with=[('*', 'any')])
            default.update(kwargs)
            return lp.Assignment(asignee, expr, **default)

        div = var('div')
        tmp = [assign(div, sum(eff_k[mu] * vector[mu] for mu in range(3)),
                      temp_var_type=lp.Optional(None))]
        self.transversify_knl = ElementWiseMap(
            {vector_T[mu]: If(kvec_zero, 0, vector[mu] - eff_k[mu] / kmag**2 * div)
             for mu in range(3)},
            tmp_instructions=tmp, lsize=(32, 1, 1), rank_shape=fft.shape(True),
        )

        kmag, Kappa = parse('kmag, Kappa')
        tmp = [assign(kmag, sqrt(sum(kk**2 for kk in eff_k))),
               assign(Kappa, sqrt(sum(kk**2 for kk in eff_k[:2])))]

        zero = fft.cdtype.type(0)
        kx_ky_zero = LogicalAnd(tuple(Comparison(fabs(eff_k[mu]), '<', 1e-10)
                                      for mu in range(2)))
        kz_nonzero = Comparison(fabs(eff_k[2]), '>', 1e-10)

        eps = var('eps')
        tmp.extend([
            assign(eps[0],
                   If(kx_ky_zero,
                      If(kz_nonzero, fft.cdtype.type(1 / 2**.5), zero),
                      (eff_k[0]*eff_k[2]/kmag - 1j*eff_k[1]) / Kappa / 2**.5)),
            assign(eps[1],
                   If(kx_ky_zero,
                      If(kz_nonzero, fft.cdtype.type(1j / 2**(1/2)), zero),
                      (eff_k[1]*eff_k[2]/kmag + 1j*eff_k[0]) / Kappa / 2**.5)),
            assign(eps[2], If(kx_ky_zero, zero, - Kappa / kmag / 2**.5))
        ])

        args = [
            lp.TemporaryVariable('eps', shape=(3,)),
            lp.TemporaryVariable('kmag'),
            lp.TemporaryVariable('Kappa'),
            ...
        ]

        plus, minus = Field('plus'), Field('minus')

        self.vec_to_pol_knl = ElementWiseMap(
            {plus: sum(vector[mu] * conj(eps[mu]) for mu in range(3)),
             minus: sum(vector[mu] * eps[mu] for mu in range(3))},
            tmp_instructions=tmp, args=args,
            lsize=(32, 1, 1), rank_shape=fft.shape(True),
        )
        self.pol_to_vec_knl = ElementWiseMap(
            {vector[mu]: plus * eps[mu] + minus * conj(eps[mu]) for mu in range(3)},
            tmp_instructions=tmp, args=args,
            lsize=(32, 1, 1), rank_shape=fft.shape(True),
        )

        from pystella.sectors import tensor_index as tid

        eff_k_hat = tuple(kk / sqrt(sum(kk**2 for kk in eff_k)) for kk in eff_k)
        hij = Field('hij', shape=(6,))
        hij_TT = Field('hij_TT', shape=(6,))

        Pab = var('P')
        tmp = {Pab[tid(a, b)]: (If(Comparison(a, '==', b), 1, 0)
                                - eff_k_hat[a-1] * eff_k_hat[b-1])
               for a in range(1, 4) for b in range(a, 4)}

        def projected_hij(a, b):
            return sum(
                (Pab[tid(a, c)] * Pab[tid(d, b)]
                 - Pab[tid(a, b)] * Pab[tid(c, d)] / 2) * hij[tid(c, d)]
                for c in range(1, 4) for d in range(1, 4)
            )

        self.tt_knl = ElementWiseMap(
            {hij_TT[tid(a, b)]: projected_hij(a, b)
             for a in range(1, 4) for b in range(a, 4)},
            tmp_instructions=tmp, lsize=(32, 1, 1), rank_shape=fft.shape(True),
        )

    def transversify(self, queue, vector, vector_T=None):
        """
        Projects out longitudinal modes of a vector field.

        :arg queue: A :class:`pyopencl.CommandQueue`.

        :arg vector: The array containing the
            momentum-space vector field to be projected.
            Must have shape ``(3,)+k_shape``, where
            ``k_shape`` is the shape of a single momentum-space field array.

        :arg vector_T: The array in which the resulting
            projected vector field will be stored.
            Must have the same shape as ``vector``.
            Defaults to *None*, in which case the projection is performed in-place.

        :returns: The :class:`pyopencl.Event` associated with the kernel invocation.
        """

        vector_T = vector_T or vector
        evt, _ = self.transversify_knl(queue, **self.eff_mom,
                                       vector=vector, vector_T=vector)
        return evt

    def pol_to_vec(self, queue, plus, minus, vector):
        """
        Projects the plus and minus polarizations of a vector field onto the
        vector components.

        :arg queue: A :class:`pyopencl.CommandQueue`.

        :arg plus: The array containing the
            momentum-space field of the plus polarization.

        :arg minus: The array containing the
            momentum-space field of the minus polarization.

        :arg vector: The array into which the vector
            field will be stored.
            Must have shape ``(3,)+k_shape``, where ``k_shape`` is the shape of a
            single momentum-space field array.

        :returns: The :class:`pyopencl.Event` associated with the kernel invocation.
        """

        evt, _ = self.pol_to_vec_knl(queue, **self.eff_mom,
                                     vector=vector, plus=plus, minus=minus)
        return evt

    def vec_to_pol(self, queue, plus, minus, vector):
        """
        Projects the components of a vector field onto the basis of plus and
        minus polarizations.

        :arg queue: A :class:`pyopencl.CommandQueue`.

        :arg plus: The array into which will be stored the
            momentum-space field of the plus polarization.

        :arg minus: The array into which will be stored the
            momentum-space field of the minus polarization.

        :arg vector: The array whose polarization
            components will be computed.
            Must have shape ``(3,)+k_shape``, where ``k_shape`` is the shape of a
            single momentum-space field array.

        :returns: The :class:`pyopencl.Event` associated with the kernel invocation.
        """

        evt, _ = self.vec_to_pol_knl(queue, **self.eff_mom,
                                     vector=vector, plus=plus, minus=minus)
        return evt

    def transverse_traceless(self, queue, hij, hij_TT=None):
        """
        Projects a tensor field to be transverse and traceless.

        :arg queue: A :class:`pyopencl.CommandQueue`.

        :arg hij: The array containing the
            momentum-space tensor field to be projected.
            Must have shape ``(6,)+k_shape``, where
            ``k_shape`` is the shape of a single momentum-space field array.

        :arg hij_TT: The array in wihch the resulting projected
            tensor field will be stored.
            Must have the same shape as ``hij``.
            Defaults to *None*, in which case the projection is performed in-place.

        :returns: The :class:`pyopencl.Event` associated with the kernel invocation.
        """

        hij_TT = hij_TT or hij
        evt, _ = self.tt_knl(queue, hij=hij, hij_TT=hij_TT, **self.eff_mom)

        # re-set to zero
        for mu in range(6):
            self.fft.zero_corner_modes(hij_TT[mu])
