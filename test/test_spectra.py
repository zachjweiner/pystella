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
from common import get_errs

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
@pytest.mark.parametrize("dtype", ["float64", "complex128"])
@pytest.mark.parametrize("L", [(10,)*3, (10, 7, 8), (3, 8, 19), (13.2, 5.71, 9.4),
                               (11, 11, 4), (4, 11, 11), (11, 4, 11)])
def test_spectra(ctx_factory, grid_shape, proc_shape, dtype, L, timing=False):
    ctx = ctx_factory()

    queue = cl.CommandQueue(ctx)
    h = 1
    mpi = ps.DomainDecomposition(proc_shape, h, grid_shape=grid_shape)
    rank_shape, _ = mpi.get_rank_shape_start(grid_shape)

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
    kvecs = np.meshgrid(*sub_k, indexing="ij", sparse=False)
    kmags = np.sqrt(sum((dki * ki)**2 for dki, ki in zip(dk, kvecs)))

    if fft.is_real:
        counts = 2. * np.ones_like(kmags)
        counts[kvecs[2] == 0] = 1
        counts[kvecs[2] == grid_shape[-1]//2] = 1
    else:
        counts = 1. * np.ones_like(kmags)

    if np.dtype(dtype) in (np.dtype("float64"), np.dtype("complex128")):
        max_rtol = 1e-8
        avg_rtol = 1e-11
    else:
        max_rtol = 2e-2
        avg_rtol = 2e-4

    bin_counts2 = spec.bin_power(np.ones_like(fk), queue=queue, k_power=0)

    max_err, avg_err = get_errs(bin_counts2, np.ones_like(bin_counts2))
    assert max_err < max_rtol and avg_err < avg_rtol, \
        f"bin counting disagrees between PowerSpectra and np.histogram" \
        f" for {grid_shape=}: {max_err=}, {avg_err=}"

    hist = np.histogram(kmags, bins=bins,
                        weights=np.abs(fk)**2 * counts * kmags**k_power)[0]
    hist = mpi.allreduce(hist) / spec.bin_counts

    # skip the Nyquist mode and the zero mode
    max_err, avg_err = get_errs(spectrum[1:-2], hist[1:-2])
    assert max_err < max_rtol and avg_err < avg_rtol, \
        f"power spectrum inaccurate for {grid_shape=}: {max_err=}, {avg_err=}"

    if timing:
        from common import timer
        t = timer(lambda: spec.bin_power(fk_d, k_power=k_power))
        print(f"power spectrum took {t:.3f} ms for {grid_shape=}, {dtype=}")


@pytest.mark.filterwarnings(
    "ignore::pyopencl.characterize.CLCharacterizationWarning")
@pytest.mark.filterwarnings("ignore::loopy.diagnostic.LoopyAdvisory")
@pytest.mark.parametrize("dtype", ["float64", "float32"])
def test_pol_spectra(ctx_factory, grid_shape, proc_shape, dtype, timing=False):
    ctx = ctx_factory()

    if np.dtype(dtype).kind != "f":
        dtype = "float64"

    queue = cl.CommandQueue(ctx)
    h = 1
    mpi = ps.DomainDecomposition(proc_shape, h, grid_shape=grid_shape)
    rank_shape, _ = mpi.get_rank_shape_start(grid_shape)

    fft = ps.DFT(mpi, ctx, queue, grid_shape, dtype)

    L = (10, 8, 7)
    dk = tuple(2 * np.pi / Li for Li in L)
    dx = tuple(Li / Ni for Li, Ni in zip(L, grid_shape))
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

    project = ps.Projector(fft, h, dk, dx)

    vector = cla.empty(queue, (3,)+fft.shape(True), cdtype)
    project.pol_to_vec(queue, plus, minus, vector)
    project.vec_to_pol(queue, plus, minus, vector)

    plus_ps_2 = spec.bin_power(plus, k_power=k_power)
    minus_ps_2 = spec.bin_power(minus, k_power=k_power)

    max_rtol = 1e-8 if dtype == np.float64 else 1e-2
    avg_rtol = 1e-11 if dtype == np.float64 else 1e-4

    max_err, avg_err = get_errs(plus_ps_1[1:-2], plus_ps_2[1:-2])
    assert max_err < max_rtol and avg_err < avg_rtol, \
        f"plus power spectrum inaccurate for {grid_shape=}: {max_err=}, {avg_err=}"

    max_err, avg_err = get_errs(minus_ps_1[1:-2], minus_ps_2[1:-2])
    assert max_err < max_rtol and avg_err < avg_rtol, \
        f"minus power spectrum inaccurate for {grid_shape=}: {max_err=}, {avg_err=}"

    vec_sum = sum(spec.bin_power(vector[mu], k_power=k_power) for mu in range(3))
    pol_sum = plus_ps_1 + minus_ps_1

    max_err, avg_err = get_errs(vec_sum[1:-2], pol_sum[1:-2])
    assert max_err < max_rtol and avg_err < avg_rtol, \
        f"polarization power spectrum inaccurate for {grid_shape=}" \
        f": {max_err=}, {avg_err=}"

    # reset
    for mu in range(3):
        fk = make_data(*fft.shape(True)).astype(cdtype)
        fk = make_hermitian(fk, fft).astype(cdtype)
        vector[mu].set(fk)

    long = cla.zeros_like(plus)
    project.decompose_vector(queue, vector, plus, minus, long, times_abs_k=True)
    plus_ps = spec.bin_power(plus, k_power=k_power)
    minus_ps = spec.bin_power(minus, k_power=k_power)
    long_ps = spec.bin_power(long, k_power=k_power)

    vec_sum = sum(spec.bin_power(vector[mu], k_power=k_power) for mu in range(3))
    dec_sum = plus_ps + minus_ps + long_ps

    max_err, avg_err = get_errs(vec_sum[1:-2], dec_sum[1:-2])
    assert max_err < max_rtol and avg_err < avg_rtol, \
        f"decomp power spectrum inaccurate for {grid_shape=}: {max_err=}, {avg_err=}"

    hij = cl.clrandom.rand(queue, (6,)+rank_shape, dtype)
    gw_spec = spec.gw(hij, project, 1.3)
    gw_pol_spec = spec.gw_polarization(hij, project, 1.3)

    max_rtol = 1e-14 if dtype == np.float64 else 1e-2
    avg_rtol = 1e-11 if dtype == np.float64 else 1e-4

    pol_sum = gw_pol_spec[0] + gw_pol_spec[1]
    max_err, avg_err = get_errs(gw_spec[1:-2], pol_sum[1:-2])
    assert max_err < max_rtol and avg_err < avg_rtol, \
        f"gw pol don't add up to gw for {grid_shape=}: {max_err=}, {avg_err=}"


if __name__ == "__main__":
    from common import parser
    args = parser.parse_args()

    test_spectra(
        ps.choose_device_and_make_context,
        grid_shape=args.grid_shape, proc_shape=args.proc_shape,
        dtype=args.dtype, timing=args.timing, L=None
    )
    test_pol_spectra(
        ps.choose_device_and_make_context,
        grid_shape=args.grid_shape, proc_shape=args.proc_shape,
        dtype=args.dtype, timing=args.timing
    )
