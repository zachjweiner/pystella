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
filterwarnings('ignore', category=lp.diagnostic.LoopyAdvisory,
               message="could not find a conflict-free mem layout")
from pyopencl.characterize import CLCharacterizationWarning
filterwarnings('ignore', category=CLCharacterizationWarning)


class PowerSpectra:
    """
    A class for computing power spectra of fields.

    .. automethod:: __init__
    .. automethod:: __call__
    .. automethod:: bin_power
    .. automethod:: polarization
    .. automethod:: gw
    """

    def __init__(self, decomp, fft, dk, volume, **kwargs):
        """
        :arg decomp: A :class:`DomainDecomposition`.

        :arg fft: An FFT object as returned by :func:`DFT`.

        :arg dk: A 3-:class:`tuple` of the momentum-space grid spacing of each
            axis (i.e., the infrared cutoff of the grid in each direction).

        :arg volume: The physical volume of the grid.

        The following keyword-only arguments are also recognized:

        :arg bin_with: A :class:`float` specifying the bin width to use.
            Defaults to ``min(dk)``.
        """

        self.decomp = decomp
        self.fft = fft
        self.grid_shape = fft.grid_shape
        self.proc_shape = decomp.proc_shape

        self.dtype = fft.dtype
        self.cdtype = fft.cdtype
        self.kshape = self.fft.shape(True)

        self.dk = dk
        self.bin_width = kwargs.pop('bin_width', min(dk))

        d3x = volume / np.product(self.grid_shape)
        self.norm = (1 / 2 / np.pi**2 / volume) * d3x**2

        sub_k = list(x.get() for x in self.fft.sub_k.values())
        kvecs = np.meshgrid(*sub_k, indexing='ij', sparse=False)
        rkmags = np.sqrt(sum((dki * ki)**2 for dki, ki in zip(self.dk, kvecs)))

        counts = 2. * np.ones_like(rkmags)
        counts[kvecs[2] == 0] = 1.
        counts[kvecs[2] == self.grid_shape[-1]//2] = 1.

        from mpi4py import MPI
        max_k = self.decomp.allreduce(np.max(rkmags), MPI.MAX)
        self.num_bins = int(max_k / self.bin_width + .5) + 1
        bins = np.arange(-.5, self.num_bins + .5) * self.bin_width

        sub_bin_counts = np.histogram(rkmags, weights=counts, bins=bins)[0]
        self.bin_counts = self.decomp.allreduce(sub_bin_counts)

        self.real_spectra_knl = self.make_spectra_knl(True, self.kshape[-1])
        # FIXME: get complex Nz better
        _Nz = self.grid_shape[-1] // self.proc_shape[1]
        self.complex_spectra_knl = self.make_spectra_knl(False, _Nz)

    def make_spectra_knl(self, is_real, Nz):
        knl = lp.make_kernel(
            "[NZ, Nx, Ny, Nz, num_bins, is_real] -> \
                { [i,j,k,b]: 0<=i<Nx and 0<=j<Ny and 0<=k<Nz and 0<=b<num_bins}",
            """
            for b
                spectrum[b] = 0 {atomic}
            end
            ... gbarrier
            for j
                for b
                    temp[b] = 0 {id=init, atomic}
                end
                for i, k
                    <> k_i = momenta_x[i]
                    <> k_j = momenta_y[j]
                    <> k_k = momenta_z[k]
                    <> kmag = sqrt((dki * k_i)**2 + (dkj * k_j)**2 + (dkk * k_k)**2)
                    <int> bin = round(kmag / bin_width)
                    <> count = if(is_real and k_k > 0 and k_k < NZ/2, 2., 1.)
                    <> power = abs(fk[i, j, k])**2 * kmag**k_power * count
                    temp[bin] = temp[bin] + power {id=tmp, dep=init, atomic}
                end
                for b
                    spectrum[b] = spectrum[b] + temp[b] {id=glb, dep=tmp, atomic}
                end
            end
            """,
            [
                lp.GlobalArg("spectrum", self.dtype, shape=(self.num_bins,),
                             for_atomic=True),
                lp.GlobalArg("momenta_x", self.dtype, shape=('Nx',)),
                lp.GlobalArg("momenta_y", self.dtype, shape=('Ny',)),
                lp.GlobalArg("momenta_z", self.dtype, shape=('Nz',)),
                lp.TemporaryVariable("temp", self.dtype, shape=(self.num_bins,),
                                     for_atomic=True,
                                     address_space=lp.AddressSpace.LOCAL),
                lp.ValueArg("k_power, bin_width, dki, dkj, dkk", self.dtype),
                ...
            ],
            default_offset=lp.auto,
            silenced_warnings=['write_race(tmp)', 'write_race(glb)'],
            seq_dependencies=True,
            lang_version=(2018, 2),
        )
        # FIXME: count incorrect for complex?

        knl = lp.fix_parameters(knl, NZ=self.grid_shape[-1], num_bins=self.num_bins,
                                dki=self.dk[0], dkj=self.dk[1], dkk=self.dk[2],
                                Nz=Nz, is_real=is_real)
        knl = lp.split_iname(knl, "k", Nz, outer_tag="g.0", inner_tag="l.0")
        knl = lp.split_iname(knl, "b", min(1024, self.num_bins),
                             outer_tag="g.0", inner_tag="l.0")
        knl = lp.tag_inames(knl, "j:g.1")
        return knl

    def bin_power(self, fk, queue=None, k_power=3, is_real=True, allocator=None):
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

        :returns: The unnormalized, binned power spectrum of ``fk``.
        """

        queue = queue or fk.queue

        if is_real:
            evt, (spectrum,) = \
                self.real_spectra_knl(queue, allocator=allocator, fk=fk,
                                      k_power=k_power, **self.fft.sub_k,
                                      bin_width=self.bin_width)
        else:
            raise NotImplementedError('complex spectra, at least distributed')
            evt, (spectrum,) = \
                self.complex_spectra_knl(queue, allocator=allocator, fk=fk,
                                         k_power=k_power, **self.fft.sub_k_c,
                                         bin_width=self.bin_width)

        full_spectrum = self.decomp.allreduce(spectrum.get())
        return full_spectrum / self.bin_counts

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

        :returns: The binned momentum-space power spectrum of ``fx``.
        """

        queue = queue or fx.queue
        is_real = fx.dtype == np.float64 or fx.dtype == np.float32

        outer_shape = fx.shape[:-3]
        from itertools import product
        slices = list(product(*[range(n) for n in outer_shape]))

        result = np.zeros(outer_shape+(self.num_bins,), self.dtype)
        for s in slices:
            fk = self.fft.dft(fx[s])
            result[s] = self.bin_power(fk, queue, k_power, is_real, allocator)

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
        """

        queue = queue or vector.queue

        vec_k = cla.empty(queue, (3,)+self.kshape, self.cdtype, allocator=None)
        # overwrite vec_k
        plus = vec_k[0]
        minus = vec_k[1]

        outer_shape = vector.shape[:-4]
        from itertools import product
        slices = list(product(*[range(n) for n in outer_shape]))

        result = np.zeros(outer_shape+(2, self.num_bins,), self.dtype)
        for s in slices:
            for mu in range(3):
                self.fft.dft(vector[s][mu], vec_k[mu])

            projector.vec_to_pol(queue, plus, minus, vec_k)
            result[s][0] = self.bin_power(plus, queue, k_power, allocator=allocator)
            result[s][1] = self.bin_power(minus, queue, k_power, allocator=allocator)

        return self.norm * result

    def gw(self, hij, projector, hubble, queue=None, k_power=3, allocator=None):
        """
        Computes the present, transverse-traceless gravitational wave power spectrum.

        .. math::

            \\Delta_t^2(k)
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
