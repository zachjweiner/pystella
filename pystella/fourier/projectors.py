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
    Constructs kernels to projector vectors and tensors to and from their
    polarization basis, to project out the longitudinal component of a vector,
    and to project a tensor field to its transverse and traceless component.

    :arg fft: An FFT object as returned by :func:`DFT`.
        ``grid_shape`` and ``dtype`` are determined by ``fft``'s attributes.

    :arg effective_k: A :class:`~collections.abc.Callable`
        with signature ``(k, dx)`` returning
        the effective momentum of the corresponding stencil :math:`\\Delta`,
        i.e., :math:`k_\\mathrm{eff}` such that
        :math:`\\Delta e^{i k x} = i k_\\mathrm{eff} e^{i k x}`.
        That is, projections are implemented relative to the stencil
        whose eigenvalues (divided by :math:`i`) are returned by this function.

    :arg dk: A 3-:class:`tuple` of the momentum-space grid spacing of each
        axis (i.e., the infrared cutoff of the grid in each direction).

    :arg dx: A 3-:class:`tuple` specifying the grid spacing of each axis.

    .. versionchanged:: 2020.2

        Added new required arguments ``dk`` and ``dx``.

    .. automethod:: transversify
    .. automethod:: pol_to_vec
    .. automethod:: vec_to_pol
    .. automethod:: transverse_traceless
    .. automethod:: tensor_to_pol
    .. automethod:: pol_to_tensor
    """

    def __init__(self, fft, effective_k, dk, dx):
        self.fft = fft

        if not callable(effective_k):
            if effective_k != 0:
                from pystella.derivs import FirstCenteredDifference
                h = effective_k
                effective_k = FirstCenteredDifference(h).get_eigenvalues
            else:
                def effective_k(k, dx):  # pylint: disable=function-redefined
                    return k

        queue = self.fft.sub_k['momenta_x'].queue
        sub_k = list(x.get().astype('int') for x in self.fft.sub_k.values())
        eff_mom_names = ('eff_mom_x', 'eff_mom_y', 'eff_mom_z')
        self.eff_mom = {}
        for mu, (name, kk) in enumerate(zip(eff_mom_names, sub_k)):
            eff_k = effective_k(dk[mu] * kk.astype(fft.rdtype), dx[mu])
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

        # note: write all output via private temporaries to allow for in-place

        div = var('div')
        div_insn = [(div, sum(eff_k[mu] * vector[mu] for mu in range(3)))]
        self.transversify_knl = ElementWiseMap(
            {vector_T[mu]: If(kvec_zero, 0, vector[mu] - eff_k[mu] / kmag**2 * div)
             for mu in range(3)},
            tmp_instructions=div_insn, lsize=(32, 1, 1), rank_shape=fft.shape(True),
        )

        import loopy as lp

        def assign(asignee, expr, **kwargs):
            default = dict(within_inames=frozenset(('i', 'j', 'k')),
                           no_sync_with=[('*', 'any')])
            default.update(kwargs)
            return lp.Assignment(asignee, expr, **default)

        kmag, Kappa = parse('kmag, Kappa')
        eps_insns = [assign(kmag, sqrt(sum(kk**2 for kk in eff_k))),
                     assign(Kappa, sqrt(sum(kk**2 for kk in eff_k[:2])))]

        zero = fft.cdtype.type(0)
        kx_ky_zero = LogicalAnd(tuple(Comparison(fabs(eff_k[mu]), '<', 1e-10)
                                      for mu in range(2)))
        kz_nonzero = Comparison(fabs(eff_k[2]), '>', 1e-10)

        eps = var('eps')
        eps_insns.extend([
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

        plus, minus, lng = Field('plus'), Field('minus'), Field('lng')

        plus_tmp, minus_tmp = parse('plus_tmp, minus_tmp')
        pol_isns = [(plus_tmp, sum(vector[mu] * conj(eps[mu]) for mu in range(3))),
                    (minus_tmp, sum(vector[mu] * eps[mu] for mu in range(3)))]

        args = [lp.TemporaryVariable('kmag'), lp.TemporaryVariable('Kappa'),
                lp.TemporaryVariable('eps', shape=(3,)), ...]

        self.vec_to_pol_knl = ElementWiseMap(
            {plus: plus_tmp, minus: minus_tmp},
            tmp_instructions=eps_insns+pol_isns, args=args,
            lsize=(32, 1, 1), rank_shape=fft.shape(True),
        )

        vector_tmp = var('vector_tmp')
        vec_insns = [(vector_tmp[mu], plus * eps[mu] + minus * conj(eps[mu]))
                     for mu in range(3)]

        self.pol_to_vec_knl = ElementWiseMap(
            {vector[mu]: vector_tmp[mu] for mu in range(3)},
            tmp_instructions=eps_insns+vec_insns, args=args,
            lsize=(32, 1, 1), rank_shape=fft.shape(True),
        )

        ksq = sum(kk**2 for kk in eff_k)
        lng_rhs = If(kvec_zero, 0, - div / ksq * 1j)
        self.vec_decomp_knl = ElementWiseMap(
            {plus: plus_tmp, minus: minus_tmp, lng: lng_rhs},
            tmp_instructions=eps_insns+pol_isns+div_insn, args=args,
            lsize=(32, 1, 1), rank_shape=fft.shape(True),
        )
        lng_rhs = If(kvec_zero, 0, - div / ksq**.5 * 1j)
        self.vec_decomp_knl_times_abs_k = ElementWiseMap(
            {plus: plus_tmp, minus: minus_tmp, lng: lng_rhs},
            tmp_instructions=eps_insns+pol_isns+div_insn, args=args,
            lsize=(32, 1, 1), rank_shape=fft.shape(True),
        )

        from pystella.sectors import tensor_index as tid

        eff_k_hat = tuple(kk / sqrt(sum(kk**2 for kk in eff_k)) for kk in eff_k)
        hij = Field('hij', shape=(6,))
        hij_TT = Field('hij_TT', shape=(6,))

        Pab = var('P')
        Pab_insns = [
            (Pab[tid(a, b)],
             (If(Comparison(a, '==', b), 1, 0) - eff_k_hat[a-1] * eff_k_hat[b-1]))
            for a in range(1, 4) for b in range(a, 4)
        ]

        hij_TT_tmp = var('hij_TT_tmp')
        TT_insns = [
            (hij_TT_tmp[tid(a, b)],
             sum((Pab[tid(a, c)] * Pab[tid(d, b)]
                  - Pab[tid(a, b)] * Pab[tid(c, d)] / 2) * hij[tid(c, d)]
                 for c in range(1, 4) for d in range(1, 4)))
            for a in range(1, 4) for b in range(a, 4)
        ]
        # note: where conditionals (branch divergence) go can matter:
        # this kernel is twice as fast when putting the branching in the global
        # write, rather than when setting hij_TT_tmp
        write_insns = [(hij_TT[tid(a, b)], If(kvec_zero, 0, hij_TT_tmp[tid(a, b)]))
                       for a in range(1, 4) for b in range(a, 4)]
        self.tt_knl = ElementWiseMap(
            write_insns, tmp_instructions=Pab_insns+TT_insns,
            lsize=(32, 1, 1), rank_shape=fft.shape(True),
        )

        tensor_to_pol_insns = {
            plus: sum(hij[tid(c, d)] * conj(eps[c-1]) * conj(eps[d-1])
                      for c in range(1, 4) for d in range(1, 4)),
            minus: sum(hij[tid(c, d)] * eps[c-1] * eps[d-1]
                       for c in range(1, 4) for d in range(1, 4))
        }
        self.tensor_to_pol_knl = ElementWiseMap(
            tensor_to_pol_insns, tmp_instructions=eps_insns, args=args,
            lsize=(32, 1, 1), rank_shape=fft.shape(True),
        )

        pol_to_tensor_insns = {
            hij[tid(a, b)]: (plus * eps[a-1] * eps[b-1]
                             + minus * conj(eps[a-1]) * conj(eps[b-1]))
            for a in range(1, 4) for b in range(a, 4)
        }
        self.pol_to_tensor_knl = ElementWiseMap(
            pol_to_tensor_insns, tmp_instructions=eps_insns, args=args,
            lsize=(32, 1, 1), rank_shape=fft.shape(True),
        )

    def transversify(self, queue, vector, vector_T=None):
        """
        Projects out the longitudinal component of a vector field.

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

        if vector_T is None:
            vector_T = vector
        evt, _ = self.transversify_knl(queue, **self.eff_mom,
                                       vector=vector, vector_T=vector_T)
        return evt

    def pol_to_vec(self, queue, plus, minus, vector):
        """
        Projects the plus and minus polarizations of a vector field onto its
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

    def decompose_vector(self, queue, vector, plus, minus, lng,
                         *, times_abs_k=False):
        """
        Decomposes a vector field into its two transverse polarizations and
        longitudinal component.

        :arg queue: A :class:`pyopencl.CommandQueue`.

        :arg vector: The array whose polarization
            components will be computed.
            Must have shape ``(3,)+k_shape``, where ``k_shape`` is the shape of a
            single momentum-space field array.

        :arg plus: The array into which will be stored the
            momentum-space field of the plus polarization.

        :arg minus: The array into which will be stored the
            momentum-space field of the minus polarization.

        :arg lng: The array into which will be stored the
            momentum-space field of the longitudinal mode.

        :returns: The :class:`pyopencl.Event` associated with the kernel invocation.

        .. versionadded:: 2020.2
        """

        if not times_abs_k:
            evt, _ = self.vec_decomp_knl(
                queue, **self.eff_mom, lng=lng, vector=vector,
                plus=plus, minus=minus
            )
        else:
            evt, _ = self.vec_decomp_knl_times_abs_k(
                queue, **self.eff_mom, lng=lng, vector=vector,
                plus=plus, minus=minus
            )

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

        if hij_TT is None:
            hij_TT = hij
        evt, _ = self.tt_knl(queue, hij=hij, hij_TT=hij_TT, **self.eff_mom)
        return evt

    def tensor_to_pol(self, queue, plus, minus, hij):
        """
        Projects the components of a rank-2 tensor field onto the basis of plus and
        minus polarizations.

        :arg queue: A :class:`pyopencl.CommandQueue`.

        :arg plus: The array into which will be stored the
            momentum-space field of the plus polarization.

        :arg minus: The array into which will be stored the
            momentum-space field of the minus polarization.

        :arg hij: The array containing the
            momentum-space tensor field to be projected.
            Must have shape ``(6,)+k_shape``, where
            ``k_shape`` is the shape of a single momentum-space field array.

        :returns: The :class:`pyopencl.Event` associated with the kernel invocation.

        .. versionadded:: 2020.1
        """

        evt, _ = self.tensor_to_pol_knl(queue, **self.eff_mom,
                                        hij=hij, plus=plus, minus=minus)
        return evt

    def pol_to_tensor(self, queue, plus, minus, hij):
        """
        Projects the plus and minus polarizations of a rank-2 tensor field onto its
        tensor components.

        :arg queue: A :class:`pyopencl.CommandQueue`.

        :arg plus: The array into which will be stored the
            momentum-space field of the plus polarization.

        :arg minus: The array into which will be stored the
            momentum-space field of the minus polarization.

        :arg hij: The array containing the
            momentum-space tensor field to be projected.
            Must have shape ``(6,)+k_shape``, where
            ``k_shape`` is the shape of a single momentum-space field array.

        :returns: The :class:`pyopencl.Event` associated with the kernel invocation.

        .. versionadded:: 2020.1
        """

        evt, _ = self.pol_to_tensor_knl(queue, **self.eff_mom,
                                        hij=hij, plus=plus, minus=minus)
        return evt
