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
import pyopencl.array as cla
import pyopencl.clrandom as clr
import loopy as lp

__doc__ = """
.. currentmodule:: pystella
.. autoclass:: RayleighGenerator
"""


def make_hermitian(fk):
    grid_shape = list(fk.shape)
    grid_shape[-1] = 2 * (grid_shape[-1] - 1)
    pos = [np.arange(0, Ni//2+1) for Ni in grid_shape]
    neg = [np.concatenate([np.array([0]), np.arange(Ni-1, Ni//2-1, -1)])
           for Ni in grid_shape]

    for k in [0, grid_shape[-1]//2]:
        for n, p in zip(neg[0], pos[0]):
            fk[n, neg[1], k] = np.conj(fk[p, pos[1], k])
            fk[p, neg[1], k] = np.conj(fk[n, pos[1], k])
        for n, p in zip(neg[1], pos[1]):
            fk[neg[0], n, k] = np.conj(fk[pos[0], p, k])
            fk[neg[0], p, k] = np.conj(fk[pos[0], n, k])

    for i in [0, grid_shape[0]//2]:
        for j in [0, grid_shape[1]//2]:
            for k in [0, grid_shape[2]//2]:
                fk[i, j, k] = np.real(fk[i, j, k])
    return fk


class RayleighGenerator:
    """
    Constructs kernels to generate Gaussian-random fields with a chosen power
    spectrum in Fourier space by drawing modes according to the corresponding
    Rayleigh distribution.

    .. automethod:: __init__
    .. automethod:: generate
    .. automethod:: init_field
    .. automethod:: init_transverse_vector
    .. automethod:: init_vector_from_pol

    In addition, the following methods apply the WKB approximation to
    initialize a field and its (conformal-) time derivative in FLRW spacetime.

    .. automethod:: generate_WKB
    .. automethod:: init_WKB_fields

    .. versionchanged:: 2019.6

        Support for generating complex fields.
    """

    def get_wkb_knl(self):
        knl = lp.make_kernel(
            "[Nx, Ny, Nz] ->  { [i,j,k]: 0<=i<Nx and 0<=j<Ny and 0<=k<Nz }",
            """
            <> amp_1 = sqrt(- log(rands[0, i, j, k]))
            <> amp_2 = sqrt(- log(rands[2, i, j, k]))
            <> phs_1 = exp(1j * 2. * pi * rands[1, i, j, k])
            <> phs_2 = exp(1j * 2. * pi * rands[3, i, j, k])
            <> power = f_power[i, j, k]
            <> Lmode = phs_1 * amp_1 * sqrt(power)
            <> Rmode = phs_2 * amp_2 * sqrt(power)
            <> fk_ = (Lmode + Rmode) / sqrt2
            fk[i, j, k] = fk_
            dfk[i, j, k] = 1j * wk[i, j, k] * (Lmode - Rmode) / sqrt2 - hubble * fk_
            """,
            [
                lp.ValueArg("hubble", self.rdtype),
                lp.GlobalArg('fk, dfk', shape=lp.auto, dtype=self.cdtype),
                ...
            ],
            seq_dependencies=True,
            silenced_warnings=['inferred_iname'],
            lang_version=(2018, 2),
        )
        knl = lp.set_options(knl, return_dict=True)
        return knl

    def get_non_wkb_knl(self):
        knl = lp.make_kernel(
            "[Nx, Ny, Nz] ->  { [i,j,k]: 0<=i<Nx and 0<=j<Ny and 0<=k<Nz }",
            """
            <> amp = sqrt(- log(rands[0, i, j, k]))
            <> phs = exp(1j * 2. * pi * rands[1, i, j, k])
            fk[i, j, k] = phs * amp * sqrt(f_power[i, j, k])
            """,
            [lp.GlobalArg('fk', shape=lp.auto, dtype=self.cdtype), ...],
            seq_dependencies=True,
            lang_version=(2018, 2),
        )
        return knl

    def __init__(self, context, fft, dk, volume, **kwargs):
        """
        :arg context: A :class:`pyopencl.Context`.

        :arg fft: An FFT object as returned by :func:`DFT`.
            The datatype of position-space arrays will match that
            of the passed FFT object.

        :arg dk: A 3-:class:`tuple` of the momentum-space grid spacing of each
            axis (i.e., the infrared cutoff of the grid in each direction).

        :arg volume: The physical volume of the grid.

        The following keyword-only arguments are recognized:

        :arg seed: The seed to the random number generator.
            Defaults to ``13298``.
        """

        self.fft = fft
        self.dtype = fft.dtype
        self.rdtype = fft.rdtype
        self.cdtype = fft.cdtype
        self.volume = volume

        sub_k = list(x.get() for x in self.fft.sub_k.values())
        kvecs = np.meshgrid(*sub_k, indexing='ij', sparse=False)
        self.kmags = np.sqrt(sum((dki * ki)**2 for dki, ki in zip(dk, kvecs)))

        seed = kwargs.pop('seed', 13298)
        self.rng = clr.ThreefryGenerator(context, seed=seed)

        def parallelize(knl):
            knl = lp.fix_parameters(knl, pi=np.pi, sqrt2=np.sqrt(2.))
            knl = lp.split_iname(knl, 'k', 32, inner_tag='l.0', outer_tag='g.0')
            knl = lp.split_iname(knl, 'j', 1, inner_tag='unr', outer_tag='g.1')
            knl = lp.split_iname(knl, 'i', 1, inner_tag='unr', outer_tag='g.2')
            return knl

        self.wkb_knl = parallelize(self.get_wkb_knl())
        self.non_wkb_knl = parallelize(self.get_non_wkb_knl())

    def _post_process(self, fk):
        from pystella.fourier import gDFT
        if self.fft.is_real and isinstance(self.fft, gDFT):
            # real fields must be Hermitian-symmetric, and it seems we
            # need to do this manually when FFT'ing with gpyfft
            fk = make_hermitian(fk)

        if self.fft.is_real:
            # can at least do this in general
            self.fft.zero_corner_modes(fk, only_imag=True)

        return fk

    # wrapper to remove 1/0 and set homogeneous power to zero
    def _ps_wrapper(self, ps_func, wk, kmags):
        if kmags[0, 0, 0] == 0.:
            wk0 = wk[0, 0, 0]
            wk[0, 0, 0] = 1.
        power = ps_func(wk)
        if kmags[0, 0, 0] == 0.:
            power[0, 0, 0] = 0.
            wk[0, 0, 0] = wk0
        return power

    def generate(self, queue, random=True, field_ps=lambda kmag: 1/2/kmag,
                 norm=1, window=lambda kmag: 1.):
        """
        Generate a 3-D array of Fourier modes with a given power spectrum and
        random phases.

        :arg queue: A :class:`pyopencl.CommandQueue`.

        :arg random: Whether to randomly sample the Rayleigh distribution
            of mode amplitudes.
            Defaults to *True*.

        :arg field_ps: A :class:`callable` returning the desired
            power spectrum of the field as a function of momentum ``kmag``.
            Defaults to the Bunch-Davies vacuum,
            ``lambda kmag: 1/2/kmag``.

        :arg norm: A constant normalization factor by which to multiply all
            power spectra.
            Defaults to ``1``.

        :arg window: A :class:`callable` window function filtering initial mode
            amplitudes.
            Defaults to ``lambda kmag: 1``, i.e., no filter.

        :returns: An :class:`numpy.ndarray` containing the generated Fourier modes
            of the field.
        """

        amplitude_sq = norm / self.volume

        rands = self.rng.uniform(queue, (2,)+self.kmags.shape, self.rdtype)
        if not random:
            rands[0] = np.exp(-1)

        f_power = (amplitude_sq * window(self.kmags)**2
                   * self._ps_wrapper(field_ps, self.kmags, self.kmags))

        evt, (fk,) = self.non_wkb_knl(queue, rands=rands, f_power=f_power,
                                      out_host=True)

        return self._post_process(fk)

    def init_field(self, fx, queue=None, **kwargs):
        """
        A wrapper which calls :meth:`generate` to initialize a field
        in Fourier space and returns its inverse Fourier transform.

        :arg fx: The array in which the field will be stored.

        The following keyword arguments are recognized:

        :arg queue: A :class:`pyopencl.CommandQueue`.
            Defaults to ``fx.queue``.

        Any additional keyword arguments are passed to :meth:`generate`.
        """

        queue = queue or fx.queue
        fk = self.generate(queue, **kwargs)
        self.fft.idft(fk, fx)

    def init_transverse_vector(self, projector, vector, queue=None, **kwargs):
        """
        A wrapper which calls :meth:`generate` to initialize a transverse
        three-vector field in Fourier space and returns its inverse Fourier
        transform.
        Each component will have the same power spectrum.

        :arg projector: A :class:`Projector` used to project out
            longitudinal components of the vector field.

        :arg vector: The array in which the vector field will be stored.

        The following keyword arguments are recognized:

        :arg queue: A :class:`pyopencl.CommandQueue`.
            Defaults to ``vector.queue``.

        Any additional keyword arguments are passed to :meth:`generate`.
        """

        queue = queue or vector.queue

        vector_k = cla.empty(queue, (3,)+self.fft.shape(True), self.cdtype)

        for mu in range(3):
            fk = self.generate(queue, **kwargs)
            vector_k[mu].set(fk)

        projector.transversify(queue, vector_k)

        for mu in range(3):
            self.fft.idft(vector_k[mu], vector[mu])

    def init_vector_from_pol(self, projector, vector, plus_ps, minus_ps,
                             queue=None, **kwargs):
        """
        A wrapper which calls :meth:`generate` to initialize a transverse
        three-vector field in Fourier space and returns its inverse Fourier
        transform.
        In contrast to :meth:`init_transverse_vector`, modes are generated
        for the plus and minus polarizations of the vector field, from which
        the vector field itself is constructed.

        :arg projector: A :class:`Projector` used to project out
            longitudinal components of the vector field.

        :arg vector: The array in which the vector field will be stored.

        :arg plus_ps: A :class:`callable` returning the power spectrum of the
            plus polarization as a function of momentum ``kmag``.

        :arg minus_ps: A :class:`callable` returning the power spectrum of the
            minus polarization as a function of momentum ``kmag``.

        The following keyword arguments are recognized:

        :arg queue: A :class:`pyopencl.CommandQueue`.
            Defaults to ``vector.queue``.

        Any additional keyword arguments are passed to :meth:`generate`.
        """

        queue = queue or vector.queue

        fk = self.generate(queue, field_ps=plus_ps, **kwargs)
        plus_k = cla.to_device(queue, fk)

        fk = self.generate(queue, field_ps=minus_ps, **kwargs)
        minus_k = cla.to_device(queue, fk)

        vector_k = cla.empty(queue, (3,)+self.fft.shape(True), self.cdtype)
        projector.pol_to_vec(queue, plus_k, minus_k, vector_k)

        for mu in range(3):
            self.fft.idft(vector_k[mu], vector[mu])

    def generate_WKB(self, queue, random=True,
                     field_ps=lambda wk: 1/2/wk,
                     norm=1, omega_k=lambda kmag: kmag,
                     hubble=0., window=lambda kmag: 1.):
        """
        Generate a 3-D array of Fourier modes with a given power spectrum and
        random phases, along with that of its time derivative
        according to the WKB approximation (for Klein-Gordon fields in
        conformal FLRW spacetime).

        Arguments match those of :meth:`generate`, with the following
        exceptions/additions:

        :arg field_ps: A :class:`callable` returning the desired
            power spectrum of the field as a function of :math:`\\omega(k)``.
            Defaults to the Bunch-Davies vacuum, ``lambda wk: 1/2/wk``,
            where ``wk=omega_k(kmag)``.

        :arg omega_k: A :class:`callable` defining the dispersion relation
            of the field.
            Defaults to ``lambda kmag: kmag``.

        :arg hubble: The value of the (conformal) Hubble parameter to use in
            generating modes for the field's time derivative.
            Only used when ``WKB=True``.
            Defaults to ``0``.

        :returns: A tuple ``(fk, dfk)`` containing the generated Fourier modes
            of the field and its time derivative.
        """

        amplitude_sq = norm / self.volume
        kshape = self.kmags.shape

        rands = self.rng.uniform(queue, (4,)+kshape, self.rdtype)
        if not random:
            rands[0] = rands[2] = np.exp(-1)

        wk = omega_k(self.kmags)
        f_power = (amplitude_sq * window(self.kmags)**2
                   * self._ps_wrapper(field_ps, wk, self.kmags))

        evt, out = self.wkb_knl(queue, rands=rands, hubble=hubble,
                                wk=wk, f_power=f_power, out_host=True)

        fk = self._post_process(out['fk'])
        dfk = self._post_process(out['dfk'])

        return fk, dfk

    def init_WKB_fields(self, fx, dfx, queue=None, **kwargs):
        """
        A wrapper which calls :meth:`generate_WKB` to initialize a field and
        its time derivative in Fourier space and inverse Fourier transform
        the results.

        :arg fx: The array in which the field will be stored.

        :arg dfx: The array in which the field's time derivative will
            be stored.

        :arg queue: A :class:`pyopencl.CommandQueue`.
            Defaults to ``fx.queue``.

        Any additional keyword arguments are passed to :meth:`generate_WKB`.
        """

        queue = queue or fx.queue
        fk, dfk = self.generate_WKB(queue, **kwargs)
        self.fft.idft(fk, fx)
        self.fft.idft(dfk, dfx)
