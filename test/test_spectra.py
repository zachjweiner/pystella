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
import pyopencl as cl
import pyopencl.array as cla
import pystella as ps
import pytest

from pyopencl.tools import (  # noqa
        pytest_generate_tests_for_pyopencl as pytest_generate_tests)


def make_data(*shape):
    return np.random.rand(*shape) + 1j * np.random.rand(*shape)


def make_hermitian(data, fft):
    from pystella.fourier import gDFT
    if isinstance(fft, gDFT):
        from pystella.fourier.rayleigh import make_hermitian
        data = make_hermitian(data)
    data = fft.zero_corner_modes(data)
    return data


@pytest.mark.filterwarnings(
    "ignore::pyopencl.characterize.CLCharacterizationWarning")
@pytest.mark.filterwarnings("ignore::loopy.diagnostic.LoopyAdvisory")
@pytest.mark.parametrize("dtype", ['float64', 'complex128'])
@pytest.mark.parametrize("L", [(10,)*3, (10, 7, 8), (3, 8, 19), (13.2, 5.71, 9.4),
                               (11, 11, 4), (4, 11, 11), (11, 4, 11)])
def test_spectra(ctx_factory, grid_shape, proc_shape, dtype, L, timing=False):
    if ctx_factory:
        ctx = ctx_factory()
    else:
        ctx = ps.choose_device_and_make_context()

    queue = cl.CommandQueue(ctx)
    h = 1
    rank_shape = tuple(Ni // pi for Ni, pi in zip(grid_shape, proc_shape))
    mpi = ps.DomainDecomposition(proc_shape, h, rank_shape)

    fft = ps.DFT(mpi, ctx, queue, grid_shape, dtype)

    L = L or (3, 5, 7)
    dk = tuple(2 * np.pi / Li for Li in L)
    cdtype = fft.cdtype
    spec = ps.PowerSpectra(mpi, fft, dk, np.product(L), bin_width=min(dk)+.001)
    # FIXME: bin_width=min(dk) sometimes disagrees to O(.1%) with numpy...

    assert int(np.sum(spec.bin_counts)) == np.product(grid_shape), \
        "bin counts don't sum to total number of points/modes"

    k_power = 2.
    fk = make_data(*fft.shape(True)).astype(cdtype)

    fk_d = cla.to_device(queue, fk)
    spectrum = spec.bin_power(fk_d, k_power=k_power)
    bins = np.arange(-.5, spec.num_bins + .5) * spec.bin_width

    sub_k = list(x.get() for x in fft.sub_k.values())
    kvecs = np.meshgrid(*sub_k, indexing='ij', sparse=False)
    kmags = np.sqrt(sum((dki * ki)**2 for dki, ki in zip(dk, kvecs)))

    if fft.is_real:
        counts = 2. * np.ones_like(kmags)
        counts[kvecs[2] == 0] = 1
        counts[kvecs[2] == grid_shape[-1]//2] = 1
    else:
        counts = 1. * np.ones_like(kmags)

    if np.dtype(dtype) in (np.dtype('float64'), np.dtype('complex128')):
        max_rtol = 1.e-8
        avg_rtol = 1.e-11
    else:
        max_rtol = 2.e-2
        avg_rtol = 2.e-4

    bin_counts2 = spec.bin_power(np.ones_like(fk), queue=queue, k_power=0)
    assert np.max(np.abs(bin_counts2 - 1)) < max_rtol, \
        "bin counting disagrees between PowerSpectra and np.histogram"

    hist = np.histogram(kmags, bins=bins,
                        weights=np.abs(fk)**2 * counts * kmags**k_power)[0]
    hist = mpi.allreduce(hist) / spec.bin_counts

    # skip the Nyquist mode and the zero mode
    err = np.abs((spectrum[1:-2] - hist[1:-2]) / hist[1:-2])
    assert np.max(err) < max_rtol and np.average(err) < avg_rtol, \
           "power spectrum inaccurate for grid_shape=%s" % (grid_shape,)

    if timing:
        from common import timer
        t = timer(lambda: spec.bin_power(fk_d, k_power=k_power))
        print("power spectrum took %.3f ms for grid_shape=%s, dtype=%s"
              % (t, grid_shape, str(dtype)))


@pytest.mark.filterwarnings(
    "ignore::pyopencl.characterize.CLCharacterizationWarning")
@pytest.mark.filterwarnings("ignore::loopy.diagnostic.LoopyAdvisory")
@pytest.mark.parametrize("dtype", ['float64', 'float32'])
def test_pol_spectra(ctx_factory, grid_shape, proc_shape, dtype, timing=False):
    if ctx_factory:
        ctx = ctx_factory()
    else:
        ctx = ps.choose_device_and_make_context()

    if np.dtype(dtype).kind != 'f':
        dtype = 'float64'

    queue = cl.CommandQueue(ctx)
    h = 1
    rank_shape = tuple(Ni // pi for Ni, pi in zip(grid_shape, proc_shape))
    mpi = ps.DomainDecomposition(proc_shape, h, rank_shape)

    fft = ps.DFT(mpi, ctx, queue, grid_shape, dtype)

    L = (10, 8, 7)
    dk = tuple(2 * np.pi / Li for Li in L)
    cdtype = fft.cdtype
    spec = ps.PowerSpectra(mpi, fft, dk, np.product(L))

    k_power = 2.

    fk = make_data(*fft.shape(True)).astype(cdtype)
    fk = make_hermitian(fk, fft).astype(cdtype)
    plus = cla.to_device(queue, fk)

    fk = make_data(*fft.shape(True)).astype(cdtype)
    fk = make_hermitian(fk, fft).astype(cdtype)
    minus = cla.to_device(queue, fk)

    plus_ps_1 = spec.bin_power(plus, queue=queue, k_power=k_power)
    minus_ps_1 = spec.bin_power(minus, queue=queue, k_power=k_power)

    project = ps.Projector(fft, h)

    vector = cla.empty(queue, (3,)+fft.shape(True), cdtype)
    project.pol_to_vec(queue, plus, minus, vector)
    project.vec_to_pol(queue, plus, minus, vector)

    plus_ps_2 = spec.bin_power(plus, k_power=k_power)
    minus_ps_2 = spec.bin_power(minus, k_power=k_power)

    max_rtol = 1.e-8 if dtype == np.float64 else 1.e-2
    avg_rtol = 1.e-11 if dtype == np.float64 else 1.e-4

    # skip the Nyquist mode and the zero mode
    err = np.abs((plus_ps_1[1:-2] - plus_ps_2[1:-2]) / plus_ps_1[1:-2])
    assert np.max(err) < max_rtol and np.average(err) < avg_rtol, \
           "plus power spectrum inaccurate for grid_shape=%s" % (grid_shape,)

    err = np.abs((minus_ps_1[1:-2] - minus_ps_2[1:-2]) / minus_ps_1[1:-2])
    assert np.max(err) < max_rtol and np.average(err) < avg_rtol, \
           "minus power spectrum inaccurate for grid_shape=%s" % (grid_shape,)

    hij = cl.clrandom.rand(queue, (6,)+rank_shape, dtype)
    gw_spec = spec.gw(hij, project, 1.3)
    gw_pol_spec = spec.gw_polarization(hij, project, 1.3)

    max_rtol = 1.e-14 if dtype == np.float64 else 1.e-2
    avg_rtol = 1.e-11 if dtype == np.float64 else 1.e-4

    diff = gw_spec - gw_pol_spec[0] - gw_pol_spec[1]
    err = diff[1:-1] / gw_spec[1:-1]
    assert np.max(err) < max_rtol and np.average(err) < avg_rtol, \
           "gw pol don't add up to gw for grid_shape=%s" % (grid_shape,)


if __name__ == "__main__":
    args = {'grid_shape': (256,)*3, 'proc_shape': (1,)*3, 'dtype': 'float64'}
    from common import get_exec_arg_dict
    args.update(get_exec_arg_dict())
    test_spectra(None, **args, L=None, timing=True)
    test_pol_spectra(None, **args, timing=True)
