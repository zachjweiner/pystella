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
import loopy as lp

from warnings import filterwarnings
filterwarnings("ignore", category=lp.diagnostic.LoopyAdvisory,
               message="could not find a conflict-free mem layout")
from pyopencl.characterize import CLCharacterizationWarning
filterwarnings("ignore", category=CLCharacterizationWarning)


class PowerSpectra:
    """
    A class for computing power spectra of fields.

    :arg decomp: A :class:`DomainDecomposition`.

    :arg fft: An FFT object as returned by :func:`DFT`.
        The datatype of position-space arrays will match that
        of the passed FFT object.

    :arg dk: A 3-:class:`tuple` of the momentum-space grid spacing of each
        axis (i.e., the infrared cutoff of the grid in each direction).

    :arg volume: The physical volume of the grid.

    The following keyword-only arguments are also recognized:

    :arg bin_width: A :class:`float` specifying the bin width to use.
        Defaults to ``min(dk)``.

    .. automethod:: __call__
    .. automethod:: bin_power
    .. automethod:: polarization
    .. automethod:: gw
    .. automethod:: gw_polarization

    .. versionchanged:: 2020.1

        Support for complex fields.
    """

    def __init__(self, decomp, fft, dk, volume, **kwargs):
        self.decomp = decomp
        self.fft = fft
        self.grid_shape = fft.grid_shape

        self.dtype = fft.dtype
        self.rdtype = fft.rdtype
        self.cdtype = fft.cdtype

        self.kshape = self.fft.shape(True)

        self.dk = dk
        self.bin_width = kwargs.pop("bin_width", min(dk))

        d3x = volume / np.product(self.grid_shape)
        self.norm = (1 / 2 / np.pi**2 / volume) * d3x**2

        sub_k = list(x.get() for x in self.fft.sub_k.values())
        kvecs = np.meshgrid(*sub_k, indexing="ij", sparse=False)
        kmags = np.sqrt(sum((dki * ki)**2 for dki, ki in zip(self.dk, kvecs)))

        if self.fft.is_real:
            counts = 2. * np.ones_like(kmags)
            counts[kvecs[2] == 0] = 1.
            counts[kvecs[2] == self.grid_shape[-1]//2] = 1.
        else:
            counts = 1. * np.ones_like(kmags)

        if self.decomp.nranks > 1:
            from mpi4py import MPI
            max_k = self.decomp.allreduce(np.max(kmags), MPI.MAX)
        else:
            max_k = np.max(kmags)

        self.num_bins = int(max_k / self.bin_width + .5) + 1
        bins = np.arange(-.5, self.num_bins + .5) * self.bin_width

        sub_bin_counts = np.histogram(kmags, weights=counts, bins=bins)[0]
        self.bin_counts = self.decomp.allreduce(sub_bin_counts)

        rank_shape = self.fft.shape(True)
        self.knl = self.make_spectra_knl(self.fft.is_real, rank_shape)

    def make_spectra_knl(self, is_real, rank_shape):
        from pymbolic import var, parse
        indices = i, j, k = parse("i, j, k")
        momenta = [var("momenta_"+xx) for xx in ("x", "y", "z")]
        ksq = sum((dk_i * mom[ii])**2
                  for mom, dk_i, ii in zip(momenta, self.dk, indices))
        kmag = var("sqrt")(ksq)
        bin_expr = var("round")(kmag / self.bin_width)

        if is_real:
            from pymbolic.primitives import If, Comparison, LogicalAnd
            nyq = self.grid_shape[-1] / 2
            condition = LogicalAnd((Comparison(momenta[2][k], ">", 0),
                                    Comparison(momenta[2][k], "<", nyq)))
            count = If(condition, 2, 1)
        else:
            count = 1

        fk = var("fk")[i, j, k]
        weight_expr = count * kmag**(var("k_power")) * var("abs")(fk)**2

        histograms = {"spectrum": (bin_expr, weight_expr)}

        args = [
            lp.GlobalArg("fk", self.cdtype, shape=("Nx", "Ny", "Nz"),
                         offset=lp.auto),
            lp.GlobalArg("momenta_x", self.rdtype, shape=("Nx",)),
            lp.GlobalArg("momenta_y", self.rdtype, shape=("Ny",)),
            lp.GlobalArg("momenta_z", self.rdtype, shape=("Nz",)),
            lp.ValueArg("k_power", self.rdtype),
            ...
        ]

        from pystella.histogram import Histogrammer
        return Histogrammer(self.decomp, histograms, self.num_bins,
                            self.rdtype, args=args, rank_shape=rank_shape)

    def bin_power(self, fk, queue=None, k_power=3, allocator=None):
        """
        Computes the binned power spectrum of a momentum-space field, weighted
        by :math:`k^n` where ``k_power`` specifies the value of :math:`n`.

        :arg fk: The array containing the complex-valued,
            momentum-space field whose power spectrum is to be computed.

        The following keyword arguments are recognized:

        :arg queue: A :class:`pyopencl.CommandQueue`.
            Defaults to ``fk.queue``.

        :arg k_power: The exponent :math:`n` to use for the weight
            :math:`\\vert \\mathbf{k} \\vert^n`.
            Defaults to 3 (to compute the "dimensionless" power spectrum).

        :arg allocator: A :mod:`pyopencl` allocator used to allocate temporary
            arrays, i.e., most usefully a :class:`pyopencl.tools.MemoryPool`.
            See the note in the documentation of
            :meth:`SpectralCollocator`.

        :returns: A :class:`numpy.ndarray` containing the unnormalized, binned power
            spectrum of ``fk``.
        """

        queue = queue or fk.queue
        if isinstance(fk, np.ndarray):
            # the generated kernel wrapper will send fk to device with the passed
            # allocator, so do it manually
            _fk = cla.to_device(queue, fk)
        else:
            _fk = fk
        result = self.knl(
            queue, allocator=allocator, fk=_fk, k_power=k_power, **self.fft.sub_k)
        return result["spectrum"] / self.bin_counts

    def __call__(self, fx, queue=None, k_power=3, allocator=None):
        """
        Computes the power spectrum of the position-space field ``fx``,

        .. math::

            \\Delta_f^2(k)
            = \\frac{1}{2 \\pi^2 V} \\int \\mathrm{d} \\Omega \\,
                \\left\\vert \\mathbf{k} \\right\\vert^n
                \\left\\vert f(\\mathbf{k}) \\right\\vert^2

        by first Fourier transforming ``fx`` and then calling :meth:`bin_power`.

        :arg fx: The array containing the position-space field
            whose power spectrum is to be computed.
            If ``fx`` has more than three axes, all the outer axes are looped over.
            As an example, if ``f`` has shape ``(2, 3, 130, 130, 130)``,
            this method loops over the outermost two axes with shape ``(2, 3)``, and
            the resulting output data would have the shape ``(2, 3, num_bins)``.

        The following keyword arguments are recognized:

        :arg queue: A :class:`pyopencl.CommandQueue`.
            Defaults to ``fx.queue``.

        :arg k_power: The exponent :math:`n` to use for the weight
            :math:`\\vert \\mathbf{k} \\vert^n`.
            Defaults to 3 (to compute the "dimensionless" power spectrum).

        :arg allocator: A :mod:`pyopencl` allocator used to allocate temporary
            arrays, i.e., most usefully a :class:`pyopencl.tools.MemoryPool`.
            See the note in the documentation of
            :meth:`SpectralCollocator`.

        :returns: A :class:`numpy.ndarray` containing the binned momentum-space
            power spectrum of ``fx``, with shape ``fx.shape[:-3]+(num_bins,)``.
        """

        queue = queue or fx.queue

        outer_shape = fx.shape[:-3]
        from itertools import product
        slices = list(product(*[range(n) for n in outer_shape]))

        result = np.zeros(outer_shape+(self.num_bins,), self.rdtype)
        for s in slices:
            fk = self.fft.dft(fx[s])
            result[s] = self.bin_power(fk, queue, k_power, allocator)

        return self.norm * result

    def polarization(self, vector, projector, queue=None, k_power=3, allocator=None):
        """
        Computes the power spectra of the plus and minus polarizations of a vector
        field.

        :arg vector: The array containing the position-space vector field
            whose power spectrum is to be computed.
            If ``vector`` has more than four axes, all the outer axes are
            looped over.
            As an example, if ``vector`` has shape ``(2, 3, 3, 130, 130, 130)``
            (where the fourth-to-last axis is the vector-component axis),
            this method loops over the outermost two axes with shape ``(2, 3)``, and
            the resulting output data would have the shape ``(2, 3, 2, num_bins)``
            (where the second-to-last axis is the polarization axis).

        :arg projector: A :class:`Projector`.

        The remaining arguments are the same as those to :meth:`__call__`.

        :returns: A :class:`numpy.ndarray` containing the polarization spectra
            with shape ``vector.shape[:-4]+(2, num_bins)``.
        """

        queue = queue or vector.queue

        vec_k = cla.empty(queue, (3,)+self.kshape, self.cdtype, allocator=None)
        # overwrite vec_k
        plus = vec_k[0]
        minus = vec_k[1]

        outer_shape = vector.shape[:-4]
        from itertools import product
        slices = list(product(*[range(n) for n in outer_shape]))

        result = np.zeros(outer_shape+(2, self.num_bins,), self.rdtype)
        for s in slices:
            for mu in range(3):
                self.fft.dft(vector[s][mu], vec_k[mu])

            projector.vec_to_pol(queue, plus, minus, vec_k)
            result[s][0] = self.bin_power(plus, queue, k_power, allocator=allocator)
            result[s][1] = self.bin_power(minus, queue, k_power, allocator=allocator)

        return self.norm * result

    def vector_decomposition(self, vector, projector, queue=None, k_power=3,
                             allocator=None):
        """
        Computes the power spectra of the plus and minus polarizations and
        longitudinal component of a vector field.

        :arg vector: The array containing the position-space vector field
            whose power spectrum is to be computed.
            If ``vector`` has more than four axes, all the outer axes are
            looped over.
            As an example, if ``vector`` has shape ``(2, 3, 3, 130, 130, 130)``
            (where the fourth-to-last axis is the vector-component axis),
            this method loops over the outermost two axes with shape ``(2, 3)``, and
            the resulting output data would have the shape ``(2, 3, 2, num_bins)``
            (where the second-to-last axis is the polarization axis).

        :arg projector: A :class:`Projector`.

        The remaining arguments are the same as those to :meth:`__call__`.

        :returns: A :class:`numpy.ndarray` containing the polarization and
            longitudinal spectra with shape ``vector.shape[:-4]+(3, num_bins)``.
        """

        queue = queue or vector.queue

        vec_k = cla.empty(queue, (3,)+self.kshape, self.cdtype, allocator=None)
        # overwrite vec_k
        plus = vec_k[0]
        minus = vec_k[1]
        lng = vec_k[2]

        outer_shape = vector.shape[:-4]
        from itertools import product
        slices = list(product(*[range(n) for n in outer_shape]))

        result = np.zeros(outer_shape+(3, self.num_bins,), self.rdtype)
        for s in slices:
            for mu in range(3):
                self.fft.dft(vector[s][mu], vec_k[mu])

            projector.decompose_vector(queue, vec_k, plus, minus, lng,
                                       times_abs_k=True)
            result[s][0] = self.bin_power(plus, queue, k_power, allocator=allocator)
            result[s][1] = self.bin_power(minus, queue, k_power, allocator=allocator)
            result[s][2] = self.bin_power(lng, queue, k_power, allocator=allocator)

        return self.norm * result

    def gw(self, hij, projector, hubble, queue=None, k_power=3, allocator=None):
        """
        Computes the present, transverse-traceless gravitational wave power spectrum.

        .. math::

            \\Delta_h^2(k)
            = \\frac{1}{24 \\pi^{2} \\mathcal{H}^{2}}
                \\frac{1}{V}
                \\sum_{i, j} \\int \\mathrm{d} \\Omega \\,
                \\left\\vert \\mathbf{k} \\right\\vert^3
                \\left\\vert h_{i j}^{\\prime}(k) \\right \\vert^{2}

        :arg hij: The array containing the
            position-space tensor field whose power spectrum is to be computed.
            Must be 4-dimensional, with the first axis being length-6.

        :arg projector: A :class:`Projector`.

        :arg hubble: The current value of the conformal Hubble parameter.

        The remaining arguments are the same as those to :meth:`__call__`.

        :returns: A :class:`numpy.ndarray` containing :math:`\\Delta_{h}^2(k)`.
        """

        queue = queue or hij.queue

        hij_k = cla.empty(queue, (6,)+self.kshape, self.cdtype, allocator=None)

        for mu in range(6):
            self.fft.dft(hij[mu], hij_k[mu])

        def tensor_id(i, j):
            a = i if i <= j else j
            b = j if i <= j else i
            return (7 - a) * a // 2 - 4 + b

        gw_spec = []
        projector.transverse_traceless(queue, hij_k)
        for mu in range(6):
            spec = self.bin_power(hij_k[mu], queue, k_power, allocator=allocator)
            gw_spec.append(spec)

        gw_tot = sum(gw_spec[tensor_id(i, j)]
                     for i in range(1, 4) for j in range(1, 4))

        return self.norm / 12 / hubble**2 * gw_tot

    def gw_polarization(self, hij, projector, hubble, queue=None, k_power=3,
                        allocator=None):
        """
        Computes the polarization components of the present gravitational wave
        power spectrum.

        .. math::

            \\Delta_{h_\\lambda}^2(k)
            = \\frac{1}{24 \\pi^{2} \\mathcal{H}^{2}}
                \\frac{1}{V}
                \\int \\mathrm{d} \\Omega \\,
                \\left\\vert \\mathbf{k} \\right\\vert^3
                \\left\\vert h_\\lambda^{\\prime}(k) \\right \\vert^{2}

        :arg hij: The array containing the
            position-space tensor field whose power spectrum is to be computed.
            Must be 4-dimensional, with the first axis being length-6.

        :arg projector: A :class:`Projector`.

        :arg hubble: The current value of the conformal Hubble parameter.

        The remaining arguments are the same as those to :meth:`__call__`.

        :returns: A :class:`numpy.ndarray` containing
            :math:`\\Delta_{h_\\lambda}^2(k)` with shape ``(2, num_bins)``.

        .. versionadded:: 2020.1
        """

        queue = queue or hij.queue

        hij_k = cla.empty(queue, (6,)+self.kshape, self.cdtype, allocator=None)
        # overwrite hij_k
        plus = hij_k[0]
        minus = hij_k[1]

        for mu in range(6):
            self.fft.dft(hij[mu], hij_k[mu])

        projector.tensor_to_pol(queue, plus, minus, hij_k)

        result = np.zeros((2, self.num_bins,), self.rdtype)
        result[0] = self.bin_power(plus, queue, k_power, allocator=allocator)
        result[1] = self.bin_power(minus, queue, k_power, allocator=allocator)

        return self.norm / 12 / hubble**2 * result
