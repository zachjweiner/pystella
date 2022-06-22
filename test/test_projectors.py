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
import pyopencl.clmath as clm
import pyopencl.array as cla
import pystella as ps
from pystella.derivs import FirstCenteredDifference, SecondCenteredDifference
from pystella.fourier import pyclDFT
import pytest
from common import get_errs

from test_rayleigh import is_hermitian

from pyopencl.tools import (  # noqa
    pytest_generate_tests_for_pyopencl as pytest_generate_tests)


@pytest.mark.parametrize("h", [1, 2, 3, 4])
@pytest.mark.parametrize("dtype", [np.float64])
def test_effective_momenta(ctx_factory, grid_shape, proc_shape, h, dtype):
    L = 10.
    N = 128
    dx = 10 / N
    dk = 2 * np.pi / L
    k = np.linspace(-N//2+1, N//2+1, 100)
    kmag = dk * k

    diff = 0
    stencil = FirstCenteredDifference(h)
    for i, coef in stencil.coefs.items():
        x = dx * i
        diff += coef * np.exp(1j * kmag * x)
        diff += - coef * np.exp(- 1j * kmag * x)

    k_diff = np.real(diff / dx / 1j)
    eff_k = stencil.get_eigenvalues(kmag, dx)

    assert np.max(np.abs(k_diff/eff_k - 1)) < 1e-14

    diff = 0
    stencil = SecondCenteredDifference(h)
    for i, coef in stencil.coefs.items():
        x = dx * i
        diff += coef * np.exp(1j * kmag * x)
        if i > 0:
            diff += coef * np.exp(- 1j * kmag * x)

    k_diff = np.real(diff / dx**2)
    eff_k = stencil.get_eigenvalues(kmag, dx)

    assert np.max(np.abs(k_diff/eff_k - 1)) < 1e-11


def make_data(queue, fft):
    kshape = fft.shape(True)
    data = np.random.rand(*kshape) + 1j * np.random.rand(*kshape)
    if isinstance(fft, pyclDFT):
        from pystella.fourier.rayleigh import make_hermitian
        data = make_hermitian(data).astype(np.complex128)

    data = fft.zero_corner_modes(data)
    return cla.to_device(queue, data)


@pytest.mark.parametrize("h", [0, 2])
@pytest.mark.parametrize("dtype", [np.float64])
def test_vector_projector(ctx_factory, grid_shape, proc_shape, h, dtype,
                          timing=False):
    ctx = ctx_factory()

    queue = cl.CommandQueue(ctx)
    mpi = ps.DomainDecomposition(proc_shape, h, grid_shape=grid_shape)
    rank_shape, _ = mpi.get_rank_shape_start(grid_shape)
    pencil_shape = tuple(ni+2*h for ni in rank_shape)

    L = (10, 8, 11.5)
    dx = tuple(Li / Ni for Li, Ni in zip(L, grid_shape))
    dk = tuple(2 * np.pi / Li for Li in L)

    fft = ps.DFT(mpi, ctx, queue, grid_shape, dtype)
    cdtype = fft.cdtype
    if h > 0:
        stencil = FirstCenteredDifference(h)
        project = ps.Projector(fft, stencil.get_eigenvalues, dk, dx)
        derivs = ps.FiniteDifferencer(mpi, h, dx)
    else:
        project = ps.Projector(fft, lambda k, dx: k, dk, dx)
        derivs = ps.SpectralCollocator(fft, dk)

    vector_x = cla.empty(queue, (3,)+pencil_shape, dtype)
    div = cla.empty(queue, rank_shape, dtype)
    pdx = cla.empty(queue, (3,)+rank_shape, dtype)

    def get_divergence_error(vector):
        for mu in range(3):
            fft.idft(vector[mu], vector_x[mu])

        derivs.divergence(queue, vector_x, div)

        derivs(queue, fx=vector_x[0], pdx=pdx[0])
        derivs(queue, fx=vector_x[1], pdy=pdx[1])
        derivs(queue, fx=vector_x[2], pdz=pdx[2])
        norm = sum([clm.fabs(pdx[mu]) for mu in range(3)])

        max_err = cla.max(clm.fabs(div)) / cla.max(norm)
        avg_err = cla.sum(clm.fabs(div)) / cla.sum(norm)
        return max_err, avg_err

    max_rtol = 1e-11 if dtype == np.float64 else 1e-4
    avg_rtol = 1e-13 if dtype == np.float64 else 1e-5

    k_shape = fft.shape(True)
    vector = cla.empty(queue, (3,)+k_shape, cdtype)

    for mu in range(3):
        vector[mu] = make_data(queue, fft).astype(cdtype)

    project.transversify(queue, vector)

    max_err, avg_err = get_divergence_error(vector)
    assert max_err < max_rtol and avg_err < avg_rtol, \
        f"transversify failed for {grid_shape=}, {h=}: {max_err=}, {avg_err=}"

    plus = make_data(queue, fft).astype(cdtype)
    minus = make_data(queue, fft).astype(cdtype)
    project.pol_to_vec(queue, plus, minus, vector)

    if isinstance(fft, pyclDFT):
        assert all(is_hermitian(vector[i]) for i in range(3)), \
            f"pol->vec is non-hermitian for {grid_shape=}, {h=}"

    max_err, avg_err = get_divergence_error(vector)
    assert max_err < max_rtol and avg_err < avg_rtol, \
        f"pol_to_vec result not transverse for {grid_shape=}, {h=}" \
        f": {max_err=}, {avg_err=}"

    vector_h = vector.get()
    vector_2 = cla.zeros_like(vector)
    project.transversify(queue, vector, vector_2)
    vector_2_h = vector_2.get()

    max_err, avg_err = get_errs(vector_h, vector_2_h)
    assert max_err < max_rtol and avg_err < avg_rtol, \
        f"pol->vector != its own transverse proj. for {grid_shape=}, {h=}" \
        f": {max_err=}, {avg_err=}"

    plus1 = cla.zeros_like(plus)
    minus1 = cla.zeros_like(minus)
    project.vec_to_pol(queue, plus1, minus1, vector)

    if isinstance(fft, pyclDFT):
        assert is_hermitian(plus1) and is_hermitian(minus1), \
            f"polarizations aren't hermitian for {grid_shape=}, {h=}"

    max_err, avg_err = get_errs(plus1.get(), plus.get())
    assert max_err < max_rtol and avg_err < avg_rtol, \
        f"pol->vec->pol (plus) is not identity for {grid_shape=}, {h=}" \
        f": {max_err=}, {avg_err=}"

    max_err, avg_err = get_errs(minus1.get(), minus.get())
    assert max_err < max_rtol and avg_err < avg_rtol, \
        f"pol->vec->pol (minus) is not identity for {grid_shape=}, {h=}" \
        f": {max_err=}, {avg_err=}"

    project.vec_to_pol(queue, vector[0], vector[1], vector)

    max_err, avg_err = get_errs(plus1.get(), vector[0].get())
    assert max_err < max_rtol and avg_err < avg_rtol, \
        f"in-place pol->vec->pol (plus) not identity for {grid_shape=}, {h=}" \
        f": {max_err=}, {avg_err=}"

    max_err, avg_err = get_errs(minus1.get(), vector[1].get())
    assert max_err < max_rtol and avg_err < avg_rtol, \
        f"in-place pol->vec->pol (minus) not identity for {grid_shape=}, {h=}" \
        f": {max_err=}, {avg_err=}"

    # reset and test longitudinal component
    for mu in range(3):
        vector[mu] = make_data(queue, fft).astype(cdtype)
        fft.idft(vector[mu], vector_x[mu])

    long = cla.zeros_like(minus)
    project.decompose_vector(queue, vector, plus1, minus1, long)

    long_x = cla.empty(queue, pencil_shape, dtype)
    fft.idft(long, long_x)

    div_true = cla.empty(queue, rank_shape, dtype)
    derivs.divergence(queue, vector_x, div_true)

    derivs(queue, fx=long_x, grd=pdx)
    div_long = cla.empty(queue, rank_shape, dtype)
    if h != 0:
        pdx_h = cla.empty(queue, (3,)+pencil_shape, dtype)
        for mu in range(3):
            mpi.restore_halos(queue, pdx[mu], pdx_h[mu])
        derivs.divergence(queue, pdx_h, div_long)
    else:
        derivs.divergence(queue, pdx, div_long)

    max_err, avg_err = get_errs(div_true.get(), div_long.get())
    assert max_err < 1e-6 and avg_err < 1e-11, \
        f"lap(longitudinal) != div vector for {grid_shape=}, {h=}" \
        f": {max_err=}, {avg_err=}"

    vector[...] = 0.
    project.decomp_to_vec(queue, plus1, minus1, long, vector, times_abs_k=True)
    for mu in range(3):
        fft.idft(vector[mu], vector_x[mu])

    div_test = cla.empty_like(div_true)
    derivs.divergence(queue, vector_x, div_test)

    max_err, avg_err = get_errs(div_test.get(), div_true.get())
    assert max_err < 1e-6 and avg_err < 1e-11, \
        f"decomp_to_vec: lap(longitudinal) != div vector for {grid_shape=}, {h=}" \
        f": {max_err=}, {avg_err=}"

    if timing:
        from common import timer
        ntime = 10
        t = timer(lambda: project.transversify(queue, vector), ntime=ntime)
        print(f"transversify took {t:.3f} ms for {grid_shape=}")
        t = timer(lambda: project.pol_to_vec(queue, plus, minus, vector),
                  ntime=ntime)
        print(f"pol_to_vec took {t:.3f} ms for {grid_shape=}")
        t = timer(lambda: project.vec_to_pol(queue, plus, minus, vector),
                  ntime=ntime)
        print(f"vec_to_pol took {t:.3f} ms for {grid_shape=}")
        t = timer(
            lambda: project.decompose_vector(queue, vector, plus, minus, long),
            ntime=ntime
        )
        print(f"decompose_vector took {t:.3f} ms for {grid_shape=}")


def tensor_id(i, j):
    a = i if i <= j else j
    b = j if i <= j else i
    return (7 - a) * a // 2 - 4 + b


@pytest.mark.parametrize("h", [0, 2])
@pytest.mark.parametrize("dtype", [np.float64])
def test_tensor_projector(ctx_factory, grid_shape, proc_shape, h, dtype,
                          timing=False):
    ctx = ctx_factory()

    queue = cl.CommandQueue(ctx)
    mpi = ps.DomainDecomposition(proc_shape, h, grid_shape=grid_shape)
    rank_shape, _ = mpi.get_rank_shape_start(grid_shape)

    L = (10, 8, 11.5)
    dx = tuple(Li / Ni for Li, Ni in zip(L, grid_shape))
    dk = tuple(2 * np.pi / Li for Li in L)

    fft = ps.DFT(mpi, ctx, queue, grid_shape, dtype)
    cdtype = fft.cdtype
    if h > 0:
        stencil = FirstCenteredDifference(h)
        project = ps.Projector(fft, stencil.get_eigenvalues, dk, dx)
        derivs = ps.FiniteDifferencer(mpi, h, dx)
    else:
        project = ps.Projector(fft, lambda k, dx: k, dk, dx)
        derivs = ps.SpectralCollocator(fft, dk)

    vector_x = cla.empty(queue, (3,)+tuple(ni+2*h for ni in rank_shape), dtype)
    div = cla.empty(queue, rank_shape, dtype)
    pdx = cla.empty(queue, (3,)+rank_shape, dtype)

    def get_divergence_errors(hij):
        max_errors = []
        avg_errors = []
        for i in range(1, 4):
            for mu in range(3):
                fft.idft(hij[tensor_id(i, mu+1)], vector_x[mu])

            derivs.divergence(queue, vector_x, div)

            derivs(queue, fx=vector_x[0], pdx=pdx[0])
            derivs(queue, fx=vector_x[1], pdy=pdx[1])
            derivs(queue, fx=vector_x[2], pdz=pdx[2])
            norm = sum([clm.fabs(pdx[mu]) for mu in range(3)])

            max_errors.append(cla.max(clm.fabs(div)) / cla.max(norm))
            avg_errors.append(cla.sum(clm.fabs(div)) / cla.sum(norm))

        return np.array(max_errors), np.array(avg_errors)

    max_rtol = 1e-11 if dtype == np.float64 else 1e-4
    avg_rtol = 1e-13 if dtype == np.float64 else 1e-5

    def get_trace_errors(hij_h):
        trace = sum([hij_h[tensor_id(i, i)] for i in range(1, 4)])
        norm = np.sqrt(sum(np.abs(hij_h[tensor_id(i, i)])**2 for i in range(1, 4)))

        trace = np.abs(trace[norm != 0]) / norm[norm != 0]
        trace = trace[trace < .9]
        return np.max(trace), np.sum(trace) / trace.size

    k_shape = fft.shape(True)
    hij = cla.empty(queue, shape=(6,)+k_shape, dtype=cdtype)

    for mu in range(6):
        hij[mu] = make_data(queue, fft).astype(cdtype)

    project.transverse_traceless(queue, hij)
    hij_h = hij.get()

    if isinstance(fft, pyclDFT):
        assert all(is_hermitian(hij_h[i]) for i in range(6)), \
            f"TT projection is non-hermitian for {grid_shape=}, {h=}"

    max_err, avg_err = get_divergence_errors(hij)
    assert all(max_err < max_rtol) and all(avg_err < avg_rtol), \
        f"TT projection not transverse for {grid_shape=}, {h=}" \
        f": {max_err=}, {avg_err=}"

    max_err, avg_err = get_trace_errors(hij_h)
    assert max_err < max_rtol and avg_err < avg_rtol, \
        f"TT projected tensor isn't traceless for {grid_shape=}, {h=}" \
        f": {max_err=}, {avg_err=}"

    plus = make_data(queue, fft).astype(cdtype)
    minus = make_data(queue, fft).astype(cdtype)
    project.pol_to_tensor(queue, plus, minus, hij)

    if isinstance(fft, pyclDFT):
        assert all(is_hermitian(hij[i]) for i in range(6)), \
            f"pol->tensor is non-hermitian for {grid_shape=}, {h=}"

    max_err, avg_err = get_divergence_errors(hij)
    assert all(max_err < max_rtol) and all(avg_err < avg_rtol), \
        f"pol->tensor not transverse for {grid_shape=}, {h=}" \
        f": {max_err=}, {avg_err=}"

    hij_h = hij.get()
    max_err, avg_err = get_trace_errors(hij_h)
    assert max_err < max_rtol and avg_err < avg_rtol, \
        f"pol->tensor isn't traceless for {grid_shape=}, {h=}" \
        f": {max_err=}, {avg_err=}"

    hij_2 = cla.zeros_like(hij)
    project.transverse_traceless(queue, hij, hij_2)
    hij_h_2 = hij_2.get()

    max_err, avg_err = get_errs(hij_h, hij_h_2)
    assert max_err < max_rtol and avg_err < avg_rtol, \
        f"pol->tensor != its own TT projection for {grid_shape=}, {h=}" \
        f": {max_err=}, {avg_err=}"

    plus1 = cla.zeros_like(plus)
    minus1 = cla.zeros_like(minus)
    project.tensor_to_pol(queue, plus1, minus1, hij)

    if isinstance(fft, pyclDFT):
        assert is_hermitian(plus1) and is_hermitian(minus1), \
            f"polarizations aren't hermitian for {grid_shape=}, {h=}"

    max_err, avg_err = get_errs(plus1.get(), plus.get())
    assert max_err < max_rtol and avg_err < avg_rtol, \
        f"pol->tensor->pol (plus) is not identity for {grid_shape=}, {h=}" \
        f": {max_err=}, {avg_err=}"

    max_err, avg_err = get_errs(minus1.get(), minus.get())
    assert max_err < max_rtol and avg_err < avg_rtol, \
        f"pol->tensor->pol (minus) is not identity for {grid_shape=}, {h=}" \
        f": {max_err=}, {avg_err=}"

    project.tensor_to_pol(queue, hij[0], hij[1], hij)

    max_err, avg_err = get_errs(plus1.get(), hij[0].get())
    assert max_err < max_rtol and avg_err < avg_rtol, \
        f"in-place pol->tensor->pol (plus) not identity for {grid_shape=}, {h=}" \
        f": {max_err=}, {avg_err=}"

    max_err, avg_err = get_errs(minus1.get(), hij[1].get())
    assert max_err < max_rtol and avg_err < avg_rtol, \
        f"in-place pol->tensor->pol (minus) not identity for {grid_shape=}, {h=}" \
        f": {max_err=}, {avg_err=}"

    if timing:
        from common import timer
        ntime = 10
        t = timer(lambda: project.transverse_traceless(queue, hij), ntime=ntime)
        print(f"TT projection took {t:.3f} ms for {grid_shape=}")
        t = timer(lambda: project.pol_to_tensor(queue, plus, minus, hij),
                  ntime=ntime)
        print(f"pol->tensor took {t:.3f} ms for {grid_shape=}")
        t = timer(lambda: project.tensor_to_pol(queue, plus, minus, hij),
                  ntime=ntime)
        print(f"tensor->pol took {t:.3f} ms for {grid_shape=}")


if __name__ == "__main__":
    from common import parser
    args = parser.parse_args()

    for h in range(1, 5):
        test_effective_momenta(
            ps.choose_device_and_make_context,
            grid_shape=args.grid_shape, proc_shape=args.proc_shape,
            h=h, dtype=args.dtype,
        )

    test_vector_projector(
        ps.choose_device_and_make_context,
        grid_shape=args.grid_shape, proc_shape=args.proc_shape,
        h=args.h, dtype=args.dtype, timing=args.timing
    )
    test_tensor_projector(
        ps.choose_device_and_make_context,
        grid_shape=args.grid_shape, proc_shape=args.proc_shape,
        h=args.h, dtype=args.dtype, timing=args.timing
    )
