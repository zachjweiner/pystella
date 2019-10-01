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


import pyopencl.array as cla
import loopy as lp

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

    def get_pol_to_vec_knl(self):
        return lp.make_kernel(
            "[Nx, Ny, Nz] -> \
                { [i,j,k,mu]: 0<=i<Nx and 0<=j<Ny and 0<=k<Nz and 0<=mu<3}",
            """
            for i, j, k
                for mu
                    <> eps[mu] = 0
                end
                <> kx = eff_mom_x[i]
                <> ky = eff_mom_y[j]
                <> kz = eff_mom_z[k]

                if fabs(kx) < 1.e-10 and fabs(ky) < 1.e-10
                    if fabs(kz) > 1.e-10
                        eps[0] = 1 / sqrt2
                        eps[1] = 1j / sqrt2
                    end
                else
                    <> Kappa = sqrt(kx**2 + ky**2)
                    <> kmag = sqrt(kx**2 + ky**2 + kz**2)

                    eps[0] = (kx * kz / kmag - 1j * ky) / Kappa / sqrt2
                    eps[1] = (ky * kz / kmag + 1j * kx) / Kappa / sqrt2
                    eps[2] = - Kappa / kmag / sqrt2
                end

                vector[mu, i, j, k] = eps[mu] * plus[i, j, k] \
                                    + conj(eps[mu]) * minus[i, j, k] {dup=mu}
            end

            """,
            seq_dependencies=True,
            default_offset=lp.auto,
            lang_version=(2018, 2),
        )

    def get_vec_to_pol_knl(self):
        return lp.make_kernel(
            "[Nx, Ny, Nz] -> \
                { [i,j,k,mu]: 0<=i<Nx and 0<=j<Ny and 0<=k<Nz and 0<=mu<3}",
            """
            for i, j, k
                for mu
                    <> eps[mu] = 0
                end
                <> kx = eff_mom_x[i]
                <> ky = eff_mom_y[j]
                <> kz = eff_mom_z[k]

                if fabs(kx) < 1.e-10 and fabs(ky) < 1.e-10
                    if fabs(kz) > 1.e-10
                        eps[0] = 1 / sqrt2
                        eps[1] = 1j / sqrt2
                    end
                else
                    <> Kappa = sqrt(kx**2 + ky**2)
                    <> kmag = sqrt(kx**2 + ky**2 + kz**2)

                    eps[0] = (kx * kz / kmag - 1j * ky) / Kappa / sqrt2
                    eps[1] = (ky * kz / kmag + 1j * kx) / Kappa / sqrt2
                    eps[2] = - Kappa / kmag / sqrt2
                end

                plus[i, j, k] = sum(mu, conj(eps[mu]) * vector[mu, i, j, k]) {dup=mu}
                minus[i, j, k] = sum(mu, eps[mu] * vector[mu, i, j, k]) {dup=mu}
            end
            """,
            seq_dependencies=True,
            default_offset=lp.auto,
            lang_version=(2018, 2),
        )

    def get_transversify_knl(self):
        return lp.make_kernel(
            "[Nx, Ny, Nz] -> \
                { [i,j,k,mu]: 0<=i<Nx and 0<=j<Ny and 0<=k<Nz and 0<=mu<3}",
            """
            for i, j, k
                <> kvec[0] = eff_mom_x[i]
                kvec[1] = eff_mom_y[j]
                kvec[2] = eff_mom_z[k]
                if fabs(kvec[0]) < 1.e-14 \
                    and fabs(kvec[1]) < 1.e-14 \
                    and fabs(kvec[2]) < 1.e-14
                    vectorT[mu, i, j, k] = 0
                else
                    <> kmag = sqrt(sum(mu, kvec[mu]**2)) {dup=mu}
                    <> div = sum(mu, kvec[mu] * vector[mu, i, j, k]) {dup=mu}

                    vectorT[mu, i, j, k] = vector[mu, i, j, k] \
                                        - kvec[mu] / kmag**2 * div {dup=mu,nosync=*}
                end
            end
            """,
            seq_dependencies=True,
            default_offset=lp.auto,
            lang_version=(2018, 2),
        )

    def get_tt_knl(self):
        knl = lp.make_kernel(
            "[Nx, Ny, Nz] -> \
                { [i,j,k,a,b,c,d]: \
                    0<=i<Nx and 0<=j<Ny and 0<=k<Nz and 1<=a,b,c,d<=3}",
            """
            for i, j, k
                <> kvec[0] = eff_mom_x[i]
                kvec[1] = eff_mom_y[j]
                kvec[2] = eff_mom_z[k]
                <> kmag = sqrt(kvec[0]**2 + kvec[1]**2 + kvec[2]**2)
                kvec[0] = kvec[0] / kmag
                kvec[1] = kvec[1] / kmag
                kvec[2] = kvec[2] / kmag

                id(a, b) := ((7 - if(a <= b, a, b)) * if(a <= b, a, b)) // 2 \
                            - 4 + if(a <= b, b, a)
                P(a, b) := if(a == b, 1, 0) - kvec[a-1] * kvec[b-1]

                for a, b
                    if a <= b
                        hTT[id(a, b)] = sum((c, d), \
                                            (P(a, c) * P(d, b) \
                                            - .5 * P(a, b) * P(c, d)) \
                                            * hij[id(c, d), i, j, k])
                    end
                end

                for a, b
                    if a <= b
                        hijTT[id(a, b), i, j, k] = hTT[id(a, b)] {dup=a,dup=b}
                    end
                end
            end
            """,
            [
                lp.GlobalArg('hij', shape='(6, Nx, Ny, Nz)'),
                lp.GlobalArg('hijTT', shape='(6, Nx, Ny, Nz)'),
                lp.TemporaryVariable('hTT', shape='(6,)'),
                ...
            ],
            seq_dependencies=True,
            default_offset=lp.auto,
            lang_version=(2018, 2),
        )
        return lp.expand_subst(knl)

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
            self.eff_mom[name] = cla.to_device(queue, eff_k)

        def process(knl):
            knl = lp.fix_parameters(knl, sqrt2=2**.5)
            knl = lp.split_iname(knl, "k", 32, outer_tag="g.0", inner_tag="l.0")
            knl = lp.split_iname(knl, "j", 1, outer_tag="g.1", inner_tag="unr")
            knl = lp.split_iname(knl, "i", 1, outer_tag="g.2", inner_tag="unr")
            knl = lp.set_options(knl, enforce_variable_access_ordered="no_check")
            return knl

        self.pol_to_vec_knl = process(self.get_pol_to_vec_knl())
        self.vec_to_pol_knl = process(self.get_vec_to_pol_knl())
        self.transversify_knl = process(self.get_transversify_knl())
        self.tt_knl = process(self.get_tt_knl())

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
                                       vector=vector, vectorT=vector)
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
        evt, _ = self.tt_knl(queue, hij=hij, hijTT=hij_TT, **self.eff_mom)

        # re-set to zero
        for mu in range(6):
            self.fft.zero_corner_modes(hij_TT[mu])
