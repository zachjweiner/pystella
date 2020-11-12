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
import pyopencl.clrandom as clr
import pyopencl.array as cla
import pystella as ps
import pytest

from pyopencl.tools import (  # noqa
    pytest_generate_tests_for_pyopencl as pytest_generate_tests)


@pytest.mark.parametrize("dtype", ["float64", "complex128"])
def test_dft(ctx_factory, grid_shape, proc_shape, dtype, timing=False):
    if ctx_factory:
        ctx = ctx_factory()
    else:
        ctx = ps.choose_device_and_make_context()

    queue = cl.CommandQueue(ctx)
    h = 1
    mpi = ps.DomainDecomposition(proc_shape, h, grid_shape=grid_shape)
    rank_shape, _ = mpi.get_rank_shape_start(grid_shape)

    fft = ps.DFT(mpi, ctx, queue, grid_shape, dtype)
    grid_size = np.product(grid_shape)
    rdtype = fft.rdtype

    if fft.is_real:
        np_dft = np.fft.rfftn
        np_idft = np.fft.irfftn
    else:
        np_dft = np.fft.fftn
        np_idft = np.fft.ifftn

    if mpi.nranks == 1:
        rng = clr.ThreefryGenerator(ctx, seed=12321)
        fx = rng.uniform(queue, grid_shape, rdtype) + 1.e-2
        if not fft.is_real:
            fx = fx + 1j * rng.uniform(queue, grid_shape, rdtype)

        fx1 = fx.get()

        fk = fft.dft(fx)
        fk1 = fk.get()
        fk_np = np_dft(fx1)

        fx2 = fft.idft(fk).get()
        fx_np = np_idft(fk1)

        rtol = 1.e-11 if dtype in ("float64", "complex128") else 2.e-3
        assert np.allclose(fx1, fx2 / grid_size, rtol=rtol, atol=0), \
            f"IDFT(DFT(f)) != f for grid_shape={grid_shape}"

        assert np.allclose(fk_np, fk1, rtol=rtol, atol=0), \
            f"DFT disagrees with numpy for grid_shape={grid_shape}"

        assert np.allclose(fx_np, fx2 / grid_size, rtol=rtol, atol=0), \
            f"IDFT disagrees with numpy for grid_shape={grid_shape}"
    else:
        mpi0 = ps.DomainDecomposition(proc_shape, 0, rank_shape)
        if mpi0.rank == 0:
            rng = clr.ThreefryGenerator(ctx, seed=12321)
            f = rng.uniform(queue, grid_shape, rdtype) + 1.e-2
            if not fft.is_real:
                f = f + 1j * rng.uniform(queue, grid_shape, rdtype)
        else:
            f = None

        fx = cla.zeros(queue, rank_shape, dtype)
        mpi0.scatter_array(queue, f, fx, root=0)
        fx1 = fx.get()

        fk = fft.dft(fx)
        fx2 = fft.idft(fk)

        # FIXME: not currently testing individual transforms against numpy

        if mpi.rank == 0:
            rtol = 1.e-11 if dtype in ("float64", "complex128") else 2.e-3
            assert np.allclose(fx1, fx2 / grid_size, rtol=rtol, atol=0), \
                f"IDFT(DFT(f)) != f for grid_shape={grid_shape}"

            # assert np.allclose(fk_np, fk1, rtol=rtol, atol=0), \
            #     f"DFT disagrees with numpy for {grid_shape=}"

            # assert np.allclose(fx_np, fx2 / grid_size, rtol=rtol, atol=0), \
            #     f"IDFT disagrees with numpy for {grid_shape=}"

    fx_cl = cla.empty(queue, rank_shape, dtype)
    pencil_shape = tuple(ni + 2*h for ni in rank_shape)
    fx_cl_halo = cla.empty(queue, pencil_shape, dtype)
    fx_np = np.empty(rank_shape, dtype)
    fx_np_halo = np.empty(pencil_shape, dtype)
    fk_cl = cla.empty(queue, fft.shape(True), fft.fk.dtype)
    fk_np = np.empty(fft.shape(True), fft.fk.dtype)

    # FIXME: check that these actually produce the correct result
    fx_types = {"cl": fx_cl, "cl halo": fx_cl_halo,
                "np": fx_np, "np halo": fx_np_halo,
                "None": None}

    fk_types = {"cl": fk_cl, "np": fk_np, "None": None}

    # run all of these to ensure no runtime errors even if no timing
    if timing:
        ntime = 20
    else:
        ntime = 1

    from common import timer

    if mpi.rank == 0:
        print(f"N = {grid_shape}, ",
              "complex" if np.dtype(dtype).kind == "c" else "real")

    from itertools import product
    for (a, input_), (b, output) in product(fx_types.items(), fk_types.items()):
        t = timer(lambda: fft.dft(input_, output), ntime=ntime)
        if mpi.rank == 0:
            print(f"dft({a}, {b}) took {t:.3f} ms")

    for (a, input_), (b, output) in product(fk_types.items(), fx_types.items()):
        t = timer(lambda: fft.idft(input_, output), ntime=ntime)
        if mpi.rank == 0:
            print(f"idft({a}, {b}) took {t:.3f} ms")


if __name__ == "__main__":
    from common import parser
    args = parser.parse_args()

    test_dft(
        None, grid_shape=args.grid_shape, proc_shape=args.proc_shape,
        dtype=args.dtype, timing=args.timing,
    )
