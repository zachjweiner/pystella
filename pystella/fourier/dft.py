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

import logging
logger = logging.getLogger(__name__)

__doc__ = """
.. currentmodule:: pystella
.. autofunction:: DFT
.. currentmodule:: pystella.fourier
.. autoclass:: pystella.fourier.dft.BaseDFT
.. autoclass:: pDFT
.. autoclass:: pyclDFT
.. currentmodule:: pystella
"""


def DFT(decomp, context, queue, grid_shape, dtype, backend=None, **kwargs):
    """
    A wrapper to the creation of various FFT class options which determines
    whether to use :class:`pystella.fourier.pyclDFT`
    (for single-device, OpenCL-based FFTs via :mod:`VkFFT` or :mod:`clfft`) or
    :class:`pystella.fourier.pDFT` (for distributed, CPU FFTs via
    :class:`mpi4py_fft.mpifft.PFFT`), based on the processor shape ``proc_shape``
    attribute of ``decomp`` and a flag ``backend``.

    :arg decomp: A :class:`DomainDecomposition`.

    :arg context: A :class:`pyopencl.Context`.

    :arg queue: A :class:`pyopencl.CommandQueue`.

    :arg grid_shape: A 3-:class:`tuple` specifying the shape of position-space arrays
        to be transformed.

    :arg dtype: The datatype of position-space arrays to be transformed. The complex
        datatype for momentum-space arrays is chosen to have the same precision.

    The following keyword-only arguments are recognized:

    :arg backend: A :class:`str` specifying the FFT backend to use.
        Valid options are ``"vkfft"``, ``"fftw"``, or ``"clfft"``.
        Defaults to *None*, in which case ``"vkfft"`` is selected for single-rank
        :class:`DomainDecomposition`\\ s and ``"fftw"`` otherwise.

    Any remaining keyword arguments are passed to :class:`pystella.fourier.pDFT`,
    should this function return such an object.

    .. versionchanged:: 2020.1

        Keyword argument ``use_fftw`` deprecated in favor of ``backend``.

    """

    if "use_fftw" in kwargs:
        from warnings import warn
        warn("Passing use_fftw is deprecated. Pass backend instead.",
             DeprecationWarning, stacklevel=2)
        if kwargs["use_fftw"]:
            backend = "fftw"

    if backend is None:
        backend = "vkfft" if tuple(decomp.proc_shape) == (1, 1, 1) else "fftw"

    if backend in ("vkfft", "clfft"):
        return pyclDFT(decomp, context, queue, grid_shape, dtype, backend=backend)
    elif backend == "fftw":
        return pDFT(decomp, queue, grid_shape, dtype, **kwargs)
    else:
        raise NotImplementedError(f"{backend} backend for DFTs.")


def _transfer_array(a, b):
    # set a = b
    if isinstance(a, np.ndarray) and isinstance(b, cla.Array):
        b.get(ary=a)
    elif isinstance(a, cla.Array) and isinstance(b, np.ndarray):
        a.set(b)
    return a


class BaseDFT:
    """
    Base class for all FFT options.

    .. automethod:: shape
    .. automethod:: dft
    .. automethod:: idft
    .. automethod:: zero_corner_modes

    .. attribute:: fx

        The (default) position-space array for transforms.

    .. attribute:: fk

        The (default) momentum-space array for transforms.
    """

    # pylint: disable=no-member
    def shape(self, forward_output=True):
        """
        :arg forward_output: A :class:`bool` specifying whether to output the
            shape for the result of the forward Fourier transform.

        :returns: A 3-:class:`tuple` of the (per--MPI-rank) shape of the requested
            array (as specified by ``forward_output``).
        """

        raise NotImplementedError

    def forward_transform(self, fx, fk, **kwargs):
        raise NotImplementedError

    def backward_transform(self, fk, fx, **kwargs):
        raise NotImplementedError

    def _debug(self, *args, **kwargs):
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(*args, **kwargs)

    def _debug_barrier(self, *args, **kwargs):
        if logger.isEnabledFor(logging.DEBUG):
            self.decomp.Barrier()
            if self.decomp.rank == 0:
                logger.debug(*args, **kwargs)

    def dft(self, fx=None, fk=None, **kwargs):
        """
        Computes the forward Fourier transform.

        :arg fx: The array to be transformed.
            Can be a :class:`pyopencl.array.Array` with or without halo padding
            (which will be removed by
            :meth:`pystella.DomainDecomposition.remove_halos`
            if needed) or a :class:`numpy.ndarray` without halo padding.
            Arrays are copied as necessary.
            Defaults to *None*, in which case :attr:`fx` (attached
            to the transform) is used.

        :arg fk: The array in which to output the result of the transform.
            Can be a :class:`pyopencl.array.Array` or a :class:`numpy.ndarray`.
            Arrays are copied as necessary.
            Defaults to *None*, in which case :attr:`fk` (attached
            to the transform) is used.

        :returns: The forward Fourier transform of ``fx``.
            Either ``fk`` if supplied or :attr:`fk`.

        Any remaining keyword arguments are passed to :meth:`forward_transform`.

        .. note::
            If you need the result of multiple Fourier transforms at once, you must
            either supply an ``fk`` array or copy the output.
            Namely, without passing ``fk`` the same memory (attached to the
            transform object) will be used for output, overwriting any prior
            results.
        """

        if fx is not None:
            if fx.shape != self.shape(False):
                if isinstance(fx, cla.Array):
                    queue = fx.queue
                elif isinstance(self.fx, cla.Array):
                    queue = self.fx.queue
                else:
                    queue = None
                self.decomp.remove_halos(queue, fx, self.fx)
                _fx = self.fx
            elif not isinstance(fx, type(self.fx)):
                _fx = _transfer_array(self.fx, fx)
            else:
                _fx = fx
        else:
            _fx = self.fx

        if fk is not None:
            if not isinstance(fk, type(self.fk)):
                _fk = self.fk
            else:
                _fk = fk
        else:
            _fk = self.fk

        self._debug_barrier("all ranks initiating forward_transform")
        _fk = self.forward_transform(_fx, _fk, **kwargs)
        self._debug_barrier("all ranks finished forward_transform")

        if fk is not None:
            if not isinstance(fk, type(self.fk)):
                _fk = _transfer_array(fk, _fk)
            else:
                _fk = fk
        else:
            _fk = _fk

        return _fk

    def idft(self, fk=None, fx=None, **kwargs):
        """
        Computes the backward Fourier transform.

        :arg fk: The array to be transformed.
            Can be a :class:`pyopencl.array.Array` or a :class:`numpy.ndarray`.
            Arrays are copied as necessary.
            Defaults to *None*, in which case :attr:`fk` (attached
            to the transform) is used.

        :arg fx: The array in which to output the result of the transform.
            Can be a :class:`pyopencl.array.Array` with or without halo padding
            (which will be restored by
            :meth:`pystella.DomainDecomposition.restore_halos`
            if needed) or a :class:`numpy.ndarray` without halo padding.
            Arrays are copied as necessary.
            Defaults to *None*, in which case :attr:`fx` (attached
            to the transform) is used.

        :returns: The forward Fourier transform of ``fx``.
            Either ``fk`` if supplied or :attr:`fk`.

        Any remaining keyword arguments are passed to :meth:`backward_transform`.

        .. note::
            If you need the result of multiple Fourier transforms at once, you must
            either supply an ``fx`` array or copy the output.
            Namely, without passing ``fx`` the same memory (attached to the
            transform object) will be used for output, overwriting any prior
            results.
        """

        if fk is not None:
            if not isinstance(fk, type(self.fk)):
                _fk = _transfer_array(self.fk, fk)
            else:
                _fk = fk
        else:
            _fk = self.fk

        if fx is not None:
            if fx.shape == self.shape(False) and isinstance(fx, type(self.fx)):
                _fx = fx
            else:
                _fx = self.fx
        else:
            _fx = self.fx

        self._debug_barrier("all ranks initiating backward_transform")
        _fx = self.backward_transform(_fk, _fx, **kwargs)
        self._debug_barrier("all ranks finished backward_transform")

        if fx is not None:
            if fx.shape != self.shape(False):
                if isinstance(fx, cla.Array):
                    queue = fx.queue
                elif isinstance(self.fx, cla.Array):
                    queue = self.fx.queue
                else:
                    queue = None
                self.decomp.restore_halos(queue, _fx, fx)
                _fx = fx
            elif not isinstance(fx, type(self.fx)):
                _fx = _transfer_array(fx, _fx)
            else:
                _fx = _fx
        else:
            _fx = _fx

        return _fx

    def zero_corner_modes(self, array, only_imag=False):
        """
        Zeros the "corner" modes (modes where each component of its
        integral wavenumber is either zero or the Nyquist along
        that axis) of ``array`` (or just the imaginary part).

        :arg array: The array to operate on.
            May be a :class:`pyopencl.array.Array` or a :class:`numpy.ndarray`.

        :arg only_imag: A :class:`bool` determining whether to only
            set the imaginary part of the array to zero.
            Defaults to *False*, i.e., setting the mode to ``0+0j``.
        """

        sub_k = list(x.get().astype("int") for x in self.sub_k.values())
        shape = self.grid_shape

        where_to_zero = []
        for mu in range(3):
            kk = sub_k[mu]
            where_0 = np.argwhere(abs(kk) == 0).reshape(-1)
            where_N2 = np.argwhere(abs(kk) == shape[mu]//2).reshape(-1)
            where_to_zero.append(np.concatenate([where_0, where_N2]))

        from itertools import product
        for i, j, k in product(*where_to_zero):
            if only_imag:
                array[i, j, k] = array[i, j, k].real
            else:
                array[i, j, k] = 0.

        return array


def fftfreq(n):
    from numpy.fft import fftfreq
    freq = fftfreq(n, 1/n)
    if n % 2 == 0:
        freq[n//2] = np.abs(freq[n//2])
    return freq


def get_sliced_momenta(grid_shape, dtype, slc, queue):
    from pystella.fourier import get_real_dtype_with_matching_prec
    rdtype = get_real_dtype_with_matching_prec(dtype)

    k = [fftfreq(n).astype(rdtype) for n in grid_shape]

    if dtype.kind == "f":
        from numpy.fft import rfftfreq
        k[-1] = rfftfreq(grid_shape[-1], 1/grid_shape[-1]).astype(rdtype)

    names = ("momenta_x", "momenta_y", "momenta_z")
    sub_k = {direction: cla.to_device(queue, k_i[s_i])
             for direction, k_i, s_i in zip(names, k, slc)}

    return sub_k


class pDFT(BaseDFT):
    """
    A wrapper to :class:`mpi4py_fft.mpifft.PFFT` to compute distributed Fast Fourier
    transforms.

    See :class:`pystella.fourier.dft.BaseDFT`.

    :arg decomp: A :class:`pystella.DomainDecomposition`.
        The shape of the MPI processor grid is determined by
        the ``proc_shape`` attribute of this object.

    :arg queue: A :class:`pyopencl.CommandQueue`.

    :arg grid_shape: A 3-:class:`tuple` specifying the shape of position-space
        arrays to be transformed.

    :arg dtype: The datatype of position-space arrays to be transformed.
        The complex datatype for momentum-space arrays is chosen to have
        the same precision.

    Any keyword arguments are passed to :class:`mpi4py_fft.mpifft.PFFT`.

    .. versionchanged:: 2020.1

        Support for complex-to-complex transforms.
    """

    def __init__(self, decomp, queue, grid_shape, dtype, **kwargs):
        self.decomp = decomp
        self.grid_shape = grid_shape
        self.proc_shape = decomp.proc_shape
        self.dtype = np.dtype(dtype)
        self.is_real = self.dtype.kind == "f"

        from pystella.fourier import get_complex_dtype_with_matching_prec
        self.cdtype = get_complex_dtype_with_matching_prec(self.dtype)
        from pystella.fourier import get_real_dtype_with_matching_prec
        self.rdtype = get_real_dtype_with_matching_prec(self.dtype)

        if self.proc_shape[0] > 1 and self.proc_shape[1] == 1:
            slab = True
        else:
            slab = False

        from mpi4py_fft.pencil import Subcomm
        default_kwargs = dict(
            axes=([0], [1], [2]), threads=16, backend="fftw", collapse=True,
        )
        default_kwargs.update(kwargs)
        comm = decomp.comm if slab else Subcomm(decomp.comm, self.proc_shape)

        from mpi4py_fft import PFFT
        self.fft = PFFT(comm, grid_shape, dtype=dtype, slab=slab, **default_kwargs)

        self.fx = self.fft.forward.input_array
        self.fk = self.fft.forward.output_array

        slc = self.fft.local_slice(True)
        self.sub_k = get_sliced_momenta(grid_shape, self.dtype, slc, queue)

    @property
    def proc_permutation(self):
        axes = list(a for b in self.fft.axes for a in b)
        for t in self.fft.transfer:
            axes[t.axisA], axes[t.axisB] = axes[t.axisB], axes[t.axisA]
        return axes

    def shape(self, forward_output=True):
        return self.fft.shape(forward_output=forward_output)

    def forward_transform(self, fx, fk, normalize=False, **kwargs):
        return self.fft.forward(
            input_array=fx, output_array=fk, normalize=normalize, **kwargs)

    def backward_transform(self, fk, fx, **kwargs):
        return self.fft.backward(input_array=fk, output_array=fx, **kwargs)


class pyclDFT(BaseDFT):
    """
    A wrapper to :mod:`pycl_fft` to compute Fast Fourier transforms with
    :mod:`VkFFT` or :mod:`clFFT`.

    See :class:`pystella.fourier.dft.BaseDFT`.

    :arg decomp: A :class:`pystella.DomainDecomposition`.

    :arg context: A :class:`pyopencl.Context`.

    :arg queue: A :class:`pyopencl.CommandQueue`.

    :arg grid_shape: A 3-:class:`tuple` specifying the shape of position-space
        arrays to be transformed.

    :arg dtype: The datatype of position-space arrays to be transformed.
        The complex datatype for momentum-space arrays is chosen to have
        the same precision.

    :arg backend: Which :mod:`pycl_fft` backend to use.
        One of ``"vkfft"`` or ``"clfft"``.

    .. versionadded:: 2021.1

        Supereseded the now-deprecated :class:`~pystella.fourier.dft.gDFT`.

    .. versionchanged:: 2020.1

        Support for complex-to-complex transforms.
    """

    def __init__(self, decomp, context, queue, grid_shape, dtype, backend):
        self.decomp = decomp
        self.grid_shape = grid_shape
        self.dtype = np.dtype(dtype)
        self.is_real = is_real = self.dtype.kind == "f"
        self.backend = backend

        from pystella.fourier import get_complex_dtype_with_matching_prec
        self.cdtype = cdtype = get_complex_dtype_with_matching_prec(self.dtype)
        from pystella.fourier import get_real_dtype_with_matching_prec
        self.rdtype = get_real_dtype_with_matching_prec(self.dtype)

        self.fx = cla.empty(queue, grid_shape, dtype)
        self.fk = cla.empty(queue, self.shape(is_real), cdtype)
        self.temp = None if not is_real else cla.empty_like(self.fk)

        from pycl_fft import get_transform_class
        Transform = get_transform_class(backend)
        self.ftransform = Transform(
            context, grid_shape, dtype, type="r2c" if is_real else "c2c",
            in_place=False)
        self.btransform = Transform(
            context, grid_shape, dtype, type="c2r" if is_real else "c2c",
            in_place=False)

        slc = ((), (), (),)
        self.sub_k = get_sliced_momenta(grid_shape, self.dtype, slc, queue)

    @property
    def proc_permutation(self):
        return tuple(range(len(self.grid_shape)))

    def shape(self, forward_output=True):
        if forward_output and self.is_real:
            shape = list(self.grid_shape)
            shape[-1] = shape[-1] // 2 + 1
            return tuple(shape)
        else:
            return self.grid_shape

    def forward_transform(self, fx, fk, **kwargs):
        return self.ftransform.forward(input=fx, output=fk)

    def backward_transform(self, fk, fx, **kwargs):
        return self.btransform.backward(input=fk, output=fx, temp=self.temp)


class gDFT(pyclDFT):
    def __init__(self, decomp, context, queue, grid_shape, dtype):
        from warnings import warn
        warn('gDFT is deprecated. Use pyclDFT with backend=="clfft" instead.',
             DeprecationWarning, stacklevel=2)
        super().__init__(decomp, context, queue, grid_shape, dtype, backend="clfft")
