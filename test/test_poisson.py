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
import pyopencl.clrandom as clr
import pyopencl.clmath as clm
import pystella as ps
import pytest

from pyopencl.tools import (  # noqa
    pytest_generate_tests_for_pyopencl as pytest_generate_tests)


@pytest.mark.parametrize("h", [0, 2])
@pytest.mark.parametrize("dtype", [np.float64])
def test_spectral_poisson(ctx_factory, grid_shape, proc_shape, h, dtype,
                          timing=False):
    ctx = ctx_factory()

    queue = cl.CommandQueue(ctx)
    mpi = ps.DomainDecomposition(proc_shape, h, grid_shape=grid_shape)
    rank_shape, _ = mpi.get_rank_shape_start(grid_shape)
    fft = ps.DFT(mpi, ctx, queue, grid_shape, dtype)

    L = (3, 5, 7)
    dx = tuple(Li / Ni for Li, Ni in zip(L, grid_shape))
    dk = tuple(2 * np.pi / Li for Li in L)

    if h == 0:
        def get_evals_2(k, dx):
            return - k**2

        derivs = ps.SpectralCollocator(fft, dk)
    else:
        from pystella.derivs import SecondCenteredDifference
        get_evals_2 = SecondCenteredDifference(h).get_eigenvalues
        derivs = ps.FiniteDifferencer(mpi, h, dx, stream=False)

    solver = ps.SpectralPoissonSolver(fft, dk, dx, get_evals_2)

    pencil_shape = tuple(ni + 2*h for ni in rank_shape)

    statistics = ps.FieldStatistics(mpi, 0, rank_shape=rank_shape,
                                    grid_size=np.prod(grid_shape))

    fx = cla.empty(queue, pencil_shape, dtype)
    rho = clr.rand(queue, rank_shape, dtype)
    rho -= statistics(rho)["mean"].item()
    lap = cla.empty(queue, rank_shape, dtype)
    rho_h = rho.get()

    for m_squared in (0, 1.2, 19.2):
        solver(queue, fx, rho, m_squared=m_squared)
        fx_h = fx.get()
        if h > 0:
            fx_h = fx_h[h:-h, h:-h, h:-h]

        derivs(queue, fx=fx, lap=lap)

        diff = np.fabs(lap.get() - rho_h - m_squared * fx_h)
        max_err = np.max(diff) / cla.max(clm.fabs(rho))
        avg_err = np.sum(diff) / cla.sum(clm.fabs(rho))

        max_rtol = 1e-12 if dtype == np.float64 else 1e-4
        avg_rtol = 1e-13 if dtype == np.float64 else 1e-5

        assert max_err < max_rtol and avg_err < avg_rtol, \
            f"solution inaccurate for {h=}, {grid_shape=}, {proc_shape=}"

    if timing:
        from common import timer
        time = timer(lambda: solver(queue, fx, rho, m_squared=m_squared), ntime=10)

        if mpi.rank == 0:
            print(f"poisson took {time:.3f} ms for {grid_shape=}, {proc_shape=}")


if __name__ == "__main__":
    from common import parser
    args = parser.parse_args()

    test_spectral_poisson(
        ps.choose_device_and_make_context,
        grid_shape=args.grid_shape, proc_shape=args.proc_shape,
        h=args.h, dtype=args.dtype, timing=args.timing
    )
