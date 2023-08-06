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
from common import get_errs

from pyopencl.tools import (  # noqa
    pytest_generate_tests_for_pyopencl as pytest_generate_tests)


@pytest.mark.parametrize("dtype", ["float64", "complex128"])
@pytest.mark.parametrize("backend", ["fftw", "clfft", "vkfft"])
def test_dft(ctx_factory, grid_shape, proc_shape, dtype, backend, timing=False):
    if backend != "fftw" and np.prod(proc_shape) > 1:
        pytest.skip("Must use mpi4py-fft on more than one rank.")

    ctx = ctx_factory()

    queue = cl.CommandQueue(ctx)
    h = 1
    mpi = ps.DomainDecomposition(proc_shape, h, grid_shape=grid_shape)
    mpi0 = ps.DomainDecomposition(proc_shape, 0, grid_shape=grid_shape)
    rank_shape, _ = mpi.get_rank_shape_start(grid_shape)

    fft = ps.DFT(mpi, ctx, queue, grid_shape, dtype, backend=backend)
    grid_size = np.prod(grid_shape)
    rdtype = fft.rdtype

    if fft.is_real:
        np_dft = np.fft.rfftn
        np_idft = np.fft.irfftn
    else:
        np_dft = np.fft.fftn
        np_idft = np.fft.ifftn

    rtol = 1e-11 if dtype in ("float64", "complex128") else 2e-3

    rng = clr.ThreefryGenerator(ctx, seed=12321*(mpi.rank+1))
    fx = rng.uniform(queue, rank_shape, rdtype) + 1e-2
    if not fft.is_real:
        fx = fx + 1j * rng.uniform(queue, rank_shape, rdtype)

    fx = fx.get()

    fk = fft.dft(fx)
    if isinstance(fk, cla.Array):
        fk = fk.get()
    fk, _fk = fk.copy(), fk  # hang on to one that fftw won't overwrite

    fx2 = fft.idft(_fk)
    if isinstance(fx2, cla.Array):
        fx2 = fx2.get()

    fx_glb = np.empty(shape=grid_shape, dtype=dtype)
    for root in range(mpi.nranks):
        mpi0.gather_array(queue, fx, fx_glb, root=root)

    fk_glb_np = np.ascontiguousarray(np_dft(fx_glb))
    fx2_glb_np = np.ascontiguousarray(np_idft(fk_glb_np))

    if backend == "fftw":
        fk_np = fk_glb_np[fft.fft.local_slice(True)]
        fx2_np = fx2_glb_np[fft.fft.local_slice(False)]
    else:
        fk_np = fk_glb_np
        fx2_np = fx2_glb_np

    max_err, avg_err = get_errs(fx, fx2 / grid_size)
    assert max_err < rtol, \
        f"IDFT(DFT(f)) != f for {grid_shape=}, {proc_shape=}: {max_err=}, {avg_err=}"

    max_err, avg_err = get_errs(fk_np, fk)
    assert max_err < rtol, \
        f"DFT disagrees with numpy for {grid_shape=}, {proc_shape=}:"\
        f" {max_err=}, {avg_err=}"

    max_err, avg_err = get_errs(fx2_np, fx2 / grid_size)
    assert max_err < rtol, \
        f"IDFT disagrees with numpy for {grid_shape=}, {proc_shape=}:"\
        f" {max_err=}, {avg_err=}"

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

    from common import timer

    if mpi.rank == 0:
        print(f"N = {grid_shape}, ",
              "complex" if np.dtype(dtype).kind == "c" else "real")

    from itertools import product
    # run all of these to ensure no runtime errors even if no timing
    for (a, input_), (b, output) in product(fx_types.items(), fk_types.items()):
        ntime = 200 if ("cl" in a and "cl" in b) else 20 if timing else 1
        t = timer(lambda: fft.dft(input_, output), ntime=ntime)  # noqa: B023
        if mpi.rank == 0:
            print(f"dft({a}, {b}) took {t:.3f} ms")

    for (a, input_), (b, output) in product(fk_types.items(), fx_types.items()):
        ntime = 200 if ("cl" in a and "cl" in b) else 20 if timing else 1
        if "cl" in a and "cl" in b:
            ntime = 200
        t = timer(lambda: fft.idft(input_, output), ntime=ntime)  # noqa: B023
        if mpi.rank == 0:
            print(f"idft({a}, {b}) took {t:.3f} ms")


if __name__ == "__main__":
    from common import parser
    parser.add_argument("--backend", type=str, default="vkfft")
    args = parser.parse_args()
    if np.prod(args.proc_shape) > 1:
        args.backend = "fftw"

    test_dft(
        ps.choose_device_and_make_context,
        grid_shape=args.grid_shape, proc_shape=args.proc_shape,
        dtype=args.dtype, backend=args.backend, timing=args.timing,
    )
