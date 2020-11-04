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
import pyopencl.clmath as clm
import pystella as ps
import pytest

from pyopencl.tools import (  # noqa
    pytest_generate_tests_for_pyopencl as pytest_generate_tests)


@pytest.mark.filterwarnings(
    "ignore::pyopencl.characterize.CLCharacterizationWarning")
@pytest.mark.filterwarnings("ignore::loopy.diagnostic.LoopyAdvisory")
@pytest.mark.parametrize("h", [0, 1, 2, 3, 4])
@pytest.mark.parametrize("dtype", [np.float64])
@pytest.mark.parametrize("stream", [True, False])
def test_gradient_laplacian(ctx_factory, grid_shape, proc_shape, h, dtype,
                            stream, timing=False):
    if h == 0 and stream is True:
        pytest.skip('no streaming spectral')

    if ctx_factory:
        ctx = ctx_factory()
    else:
        ctx = ps.choose_device_and_make_context()

    queue = cl.CommandQueue(ctx)
    mpi = ps.DomainDecomposition(proc_shape, h, grid_shape=grid_shape)
    rank_shape, start = mpi.get_rank_shape_start(grid_shape)

    L = (3, 5, 7)
    dx = tuple(Li / Ni for Li, Ni in zip(L, grid_shape))
    dk = tuple(2 * np.pi / Li for Li in L)

    if h == 0:
        def get_evals_1(k, dx):
            return k

        def get_evals_2(k, dx):
            return - k**2

        fft = ps.DFT(mpi, ctx, queue, grid_shape, dtype)
        derivs = ps.SpectralCollocator(fft, dk)
    else:
        from pystella.derivs import FirstCenteredDifference, SecondCenteredDifference
        get_evals_1 = FirstCenteredDifference(h).get_eigenvalues
        get_evals_2 = SecondCenteredDifference(h).get_eigenvalues
        if stream:
            try:
                derivs = ps.FiniteDifferencer(mpi, h, dx, rank_shape=rank_shape,
                                              stream=stream)
            except:  # noqa
                pytest.skip("StreamingStencil unavailable")
        else:
            derivs = ps.FiniteDifferencer(mpi, h, dx, rank_shape=rank_shape)

    pencil_shape = tuple(ni + 2*h for ni in rank_shape)

    # set up test data
    fx_h = np.empty(pencil_shape, dtype)
    kvec = np.array(dk) * np.array([-5, 4, -3]).astype(dtype)
    xvec = np.meshgrid(*[dxi * np.arange(si, si + ni)
                         for dxi, si, ni in zip(dx, start, rank_shape)],
                       indexing='ij')

    phases = sum(ki * xi for ki, xi in zip(kvec, xvec))
    if h > 0:
        fx_h[h:-h, h:-h, h:-h] = np.sin(phases)
    else:
        fx_h[:] = np.sin(phases)
    fx_cos = np.cos(phases)

    fx = cla.empty(queue, pencil_shape, dtype)
    fx.set(fx_h)

    lap = cla.empty(queue, rank_shape, dtype)
    grd = cla.empty(queue, (3,)+rank_shape, dtype)
    derivs(queue, fx=fx, lap=lap, grd=grd)

    eff_kmag_sq = sum(get_evals_2(kvec_i, dxi) for dxi, kvec_i in zip(dx, kvec))

    lap_true = cla.to_device(queue, eff_kmag_sq * np.sin(phases))
    diff = clm.fabs(lap - lap_true)
    max_err = cla.max(diff) / cla.max(clm.fabs(lap_true))
    avg_err = cla.sum(diff) / cla.sum(clm.fabs(lap_true))

    max_rtol = 1.e-11 if dtype == np.float64 else 3.e-4
    avg_rtol = 1.e-12 if dtype == np.float64 else 5.e-5
    assert max_err < max_rtol and avg_err < avg_rtol, \
        f"lap inaccurate for {h=}, {grid_shape=}, {proc_shape=}"

    for i in range(3):
        eff_k = get_evals_1(kvec[i], dx[i])

        pdi_true = cla.to_device(queue, eff_k * fx_cos)
        diff = clm.fabs(grd[i] - pdi_true)
        max_err = cla.max(diff) / cla.max(clm.fabs(pdi_true))
        avg_err = cla.sum(diff) / cla.sum(clm.fabs(pdi_true))

        max_rtol = 1.e-12 if dtype == np.float64 else 1.e-5
        avg_rtol = 1.e-13 if dtype == np.float64 else 3.e-6
        assert max_err < max_rtol and avg_err < avg_rtol, \
            f"pd{i} inaccurate for {h=}, {grid_shape=}, {proc_shape=}"

    vec = cla.empty(queue, (3,)+pencil_shape, dtype)
    for mu in range(3):
        vec[mu] = fx

    div = cla.empty(queue, rank_shape, dtype)
    derivs.divergence(queue, vec, div)

    div_true = sum(grd[i] for i in range(3))
    diff = clm.fabs(div - div_true)
    max_err = cla.max(diff) / cla.max(clm.fabs(div_true))
    avg_err = cla.sum(diff) / cla.sum(clm.fabs(div_true))

    max_rtol = 1.e-14 if dtype == np.float64 else 3.e-4
    avg_rtol = 1.e-15 if dtype == np.float64 else 5.e-5
    assert max_err < max_rtol and avg_err < avg_rtol, \
        f"div inaccurate for {h=}, {grid_shape=}, {proc_shape=}"

    if timing:
        from common import timer

        base_args = dict(queue=queue, fx=fx)
        div_args = dict(queue=queue, vec=vec, div=div)
        if h == 0:
            import pyopencl.tools as clt
            pool = clt.MemoryPool(clt.ImmediateAllocator(queue))
            base_args['allocator'] = pool
            div_args['allocator'] = pool

        times = {}
        times['gradient and laplacian'] = timer(
            lambda: derivs(lap=lap, grd=grd, **base_args)
        )
        times['gradient'] = timer(lambda: derivs(grd=grd, **base_args))
        times['laplacian'] = timer(lambda: derivs(lap=lap, **base_args))
        times['pdx'] = timer(lambda: derivs(pdx=grd[0], **base_args))
        times['pdy'] = timer(lambda: derivs(pdy=grd[1], **base_args))
        times['pdz'] = timer(lambda: derivs(pdz=grd[2], **base_args))
        times['divergence'] = timer(lambda: derivs.divergence(**div_args))

        if mpi.rank == 0:
            print(f"{grid_shape=}, {h=}, {proc_shape=}")
            for key, val in times.items():
                print(f"{key} took {val:.3f} ms")


if __name__ == "__main__":
    from common import parser
    args = parser.parse_args()

    for stream in [True, False]:
        test_gradient_laplacian(
            None, grid_shape=args.grid_shape, proc_shape=args.proc_shape,
            h=args.h, dtype=args.dtype, timing=args.timing, stream=stream
        )

    test_gradient_laplacian(
        None, grid_shape=args.grid_shape, proc_shape=args.proc_shape,
        h=0, dtype=args.dtype, timing=args.timing, stream=False
    )
