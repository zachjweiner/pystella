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
from pystella.fourier import gDFT
import pytest

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

    assert np.max(np.abs(k_diff/eff_k - 1)) < 1.e-14

    diff = 0
    stencil = SecondCenteredDifference(h)
    for i, coef in stencil.coefs.items():
        x = dx * i
        diff += coef * np.exp(1j * kmag * x)
        if i > 0:
            diff += coef * np.exp(- 1j * kmag * x)

    k_diff = np.real(diff / dx**2)
    eff_k = stencil.get_eigenvalues(kmag, dx)

    assert np.max(np.abs(k_diff/eff_k - 1)) < 1.e-11


def make_data(queue, fft):
    kshape = fft.shape(True)
    data = np.random.rand(*kshape) + 1j * np.random.rand(*kshape)
    if isinstance(fft, gDFT):
        from pystella.fourier.rayleigh import make_hermitian
        data = make_hermitian(data).astype(np.complex128)

    data = fft.zero_corner_modes(data)
    return cla.to_device(queue, data)


@pytest.mark.filterwarnings(
    "ignore::pyopencl.characterize.CLCharacterizationWarning")
@pytest.mark.filterwarnings("ignore::loopy.diagnostic.LoopyAdvisory")
@pytest.mark.parametrize("h", [0, 2])
@pytest.mark.parametrize("pol", [False, True])
@pytest.mark.parametrize("dtype", [np.float64])
def test_vector_projector(ctx_factory, grid_shape, proc_shape, h, dtype, pol,
                          timing=False):
    if proc_shape[0] * proc_shape[1] * proc_shape[2] > 1 and h == 0:
        pytest.skip("can't test continuum projectors on multiple ranks yet")

    if ctx_factory:
        ctx = ctx_factory()
    else:
        ctx = ps.choose_device_and_make_context()

    queue = cl.CommandQueue(ctx)
    rank_shape = tuple(Ni // pi for Ni, pi in zip(grid_shape, proc_shape))
    mpi = ps.DomainDecomposition(proc_shape, h, rank_shape)

    L = (10,)*3
    dx = tuple(Li / Ni for Li, Ni in zip(L, grid_shape))
    dk = tuple(2 * np.pi / Li for Li in L)

    fft = ps.DFT(mpi, ctx, queue, grid_shape, dtype)
    cdtype = fft.cdtype
    if h > 0:
        stencil = FirstCenteredDifference(h)
        project = ps.Projector(fft, stencil.get_eigenvalues)
    else:
        project = ps.Projector(fft, lambda k, dx: k)

    k_shape = fft.shape(True)
    vector = cla.empty(queue, (3,)+k_shape, cdtype)

    if pol:
        plus = make_data(queue, fft).astype(cdtype)
        minus = make_data(queue, fft).astype(cdtype)

        vector = cla.empty(queue, (3,)+k_shape, cdtype)
        project.pol_to_vec(queue, plus, minus, vector)

        if isinstance(fft, gDFT):
            for i in range(3):
                assert is_hermitian(vector[i]), \
                    "pol->vec is non-hermitian for grid_shape=%s, halo_shape=%s" \
                    % (grid_shape, h)

        plus1 = cla.zeros_like(plus)
        minus1 = cla.zeros_like(minus)

        project.vec_to_pol(queue, plus1, minus1, vector)

        if isinstance(fft, gDFT):
            assert is_hermitian(plus1), \
                "plus pol is not hermitian for grid_shape=%s, halo_shape=%s" %  \
                (grid_shape, h)
            assert is_hermitian(minus1), \
                "minus pol is not hermitian for grid_shape=%s, halo_shape=%s" %  \
                (grid_shape, h)

        assert np.allclose(plus1.get(), plus.get(), atol=0., rtol=1.e-11) and \
            np.allclose(minus1.get(), minus.get(), atol=0., rtol=1.e-11), \
            "pol->vec->pol is not identity for grid_shape=%s, halo_shape=%s" % \
            (grid_shape, h)

    else:
        for mu in range(3):
            vector[mu] = make_data(queue, fft).astype(cdtype)

        # apply twice to ensure smallness
        project.transversify(queue, vector)
        project.transversify(queue, vector)

    if h == 0:
        derivs = ps.SpectralCollocator(fft, dk)
    else:
        derivs = ps.FiniteDifferencer(mpi, h, dx)

    vector_x = cla.empty(queue, (3,)+tuple(ni+2*h for ni in rank_shape), dtype)
    for mu in range(3):
        fft.idft(vector[mu], vector_x[mu])

    div = cla.empty(queue, rank_shape, dtype)
    derivs.divergence(queue, vector_x, div)

    pdx = cla.empty(queue, (3,)+rank_shape, dtype)
    derivs(queue, fx=vector_x[0], pdx=pdx[0])
    derivs(queue, fx=vector_x[1], pdy=pdx[1])
    derivs(queue, fx=vector_x[2], pdz=pdx[2])
    norm = sum([clm.fabs(pdx[mu]) for mu in range(3)])

    max_err = cla.max(clm.fabs(div)) / cla.max(norm)
    avg_err = cla.sum(clm.fabs(div)) / cla.sum(norm)
    print(max_err, avg_err)

    max_rtol = 1.e-14 if dtype == np.float64 else 1.e-4
    avg_rtol = 1.e-15 if dtype == np.float64 else 1.e-6

    assert max_err < max_rtol and avg_err < avg_rtol, \
            "%s is not transverse for grid_shape=%s, halo_shape=%s" % \
            ("pol_to_vec" if pol else "transversify", grid_shape, h)

    if timing:
        from common import timer
        ntime = 10
        if pol:
            t = timer(lambda: project.pol_to_vec(queue, plus, minus, vector),
                      ntime=ntime)
            print("pol_to_vec took %.3f ms for grid_shape=%s" % (t, grid_shape))
            t = timer(lambda: project.vec_to_pol(queue, plus, minus, vector),
                      ntime=ntime)
            print("vec_to_pol took %.3f ms for grid_shape=%s" % (t, grid_shape))
        else:
            t = timer(lambda: project.transversify(queue, vector), ntime=ntime)
            print("transversify took %.3f ms for grid_shape=%s" % (t, grid_shape))


def tensor_id(i, j):
    a = i if i <= j else j
    b = j if i <= j else i
    return (7 - a) * a // 2 - 4 + b


@pytest.mark.filterwarnings(
    "ignore::pyopencl.characterize.CLCharacterizationWarning")
@pytest.mark.filterwarnings("ignore::loopy.diagnostic.LoopyAdvisory")
@pytest.mark.parametrize("h", [0, 2])
@pytest.mark.parametrize("pol", [False, True])
@pytest.mark.parametrize("dtype", [np.float64])
def test_tensor_projector(ctx_factory, grid_shape, proc_shape, h, dtype, pol,
                          timing=False):
    if proc_shape[0] * proc_shape[1] * proc_shape[2] > 1 and h == 0:
        pytest.skip("can't test continuum projectors on multiple ranks yet")
    if pol:
        pytest.skip("No tensor polarization projector yet")

    if ctx_factory:
        ctx = ctx_factory()
    else:
        ctx = ps.choose_device_and_make_context()

    queue = cl.CommandQueue(ctx)
    rank_shape = tuple(Ni // pi for Ni, pi in zip(grid_shape, proc_shape))
    mpi = ps.DomainDecomposition(proc_shape, h, rank_shape)

    L = (10,)*3
    dx = tuple(Li / Ni for Li, Ni in zip(L, grid_shape))
    dk = tuple(2 * np.pi / Li for Li in L)

    fft = ps.DFT(mpi, ctx, queue, grid_shape, dtype)
    cdtype = fft.cdtype
    if h > 0:
        stencil = FirstCenteredDifference(h)
        project = ps.Projector(fft, stencil.get_eigenvalues)
    else:
        project = ps.Projector(fft, lambda k, dx: k)

    k_shape = fft.shape(True)
    hij = cla.empty(queue, shape=(6,)+k_shape, dtype=cdtype)

    if pol:
        pass
    else:
        for mu in range(6):
            hij[mu] = make_data(queue, fft).astype(cdtype)

        project.transverse_traceless(queue, hij)

    hij_h = hij.get()

    if isinstance(fft, gDFT):
        for i in range(6):
            assert is_hermitian(hij_h[i]), \
                "TT projection is non-hermitian for grid_shape=%s, halo_shape=%s" \
                % (grid_shape, h)

    trace = sum([hij_h[tensor_id(i, i)] for i in range(1, 4)])
    tracenorm = np.sqrt(sum([np.abs(hij_h[tensor_id(i, i)])**2
                             for i in range(1, 4)]))

    trace = np.abs(trace[tracenorm != 0]) / tracenorm[tracenorm != 0]
    trace = trace[trace < .9]
    max_err = np.max(trace)
    l2_err = np.sum(trace) / trace.size

    max_rtol = 1.e-9 if dtype == np.float64 else 1.e-4
    l2_rtol = 1.e-14 if dtype == np.float64 else 1.e-6

    assert max_err < max_rtol and l2_err < l2_rtol, \
        "TT projected tensor isn't traceless for grid_shape=%s, halo_shape=%s" \
        % (grid_shape, h)

    if h == 0:
        derivs = ps.SpectralCollocator(fft, dk)
    else:
        derivs = ps.FiniteDifferencer(mpi, h, dx)

    vector_x = cla.empty(queue, (3,)+tuple(ni+2*h for ni in rank_shape), dtype)
    div = cla.empty(queue, rank_shape, dtype)
    pdx = cla.empty(queue, (3,)+rank_shape, dtype)

    for i in range(1, 4):
        for mu in range(3):
            fft.idft(hij[tensor_id(i, mu+1)], vector_x[mu])

        derivs.divergence(queue, vector_x, div)

        derivs(queue, fx=vector_x[0], pdx=pdx[0])
        derivs(queue, fx=vector_x[1], pdy=pdx[1])
        derivs(queue, fx=vector_x[2], pdz=pdx[2])
        norm = sum([clm.fabs(pdx[mu]) for mu in range(3)])

        max_err = cla.max(clm.fabs(div)) / cla.max(norm)
        avg_err = cla.sum(clm.fabs(div)) / cla.sum(norm)
        print(max_err, avg_err)

        max_rtol = 1.e-14 if dtype == np.float64 else 1.e-4
        avg_rtol = 1.e-15 if dtype == np.float64 else 1.e-6

        assert max_err < max_rtol and avg_err < avg_rtol, \
            "TT projection not transverse for grid_shape=%s, halo_shape=%s, i=%d" \
            % (grid_shape, h, i)

    if timing:
        from common import timer
        ntime = 10
        t = timer(lambda: project.transverse_traceless(queue, hij), ntime=ntime)
        print("TT projection took %.3f ms for grid_shape=%s" % (t, grid_shape))


if __name__ == "__main__":
    args = {'grid_shape': (256,)*3, 'proc_shape': (1,)*3, 'dtype': np.float64}
    from common import get_exec_arg_dict
    args.update(get_exec_arg_dict())
    for h in range(1, 5):
        test_effective_momenta(None, **args, h=h)
    for h in range(0, 5):
        test_vector_projector(None, **args, h=h, pol=False, timing=True)
        test_vector_projector(None, **args, h=h, pol=True, timing=True)
    for h in range(0, 5):
        test_tensor_projector(None, **args, h=h, pol=False, timing=True)
