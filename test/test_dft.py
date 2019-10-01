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


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_dft(ctx_factory, grid_shape, proc_shape, dtype, timing=False):
    if ctx_factory:
        ctx = ctx_factory()
    else:
        ctx = ps.choose_device_and_make_context()

    queue = cl.CommandQueue(ctx)
    h = 1
    rank_shape = tuple(Ni // pi for Ni, pi in zip(grid_shape, proc_shape))
    mpi = ps.DomainDecomposition(proc_shape, h, rank_shape)

    fft = ps.DFT(mpi, ctx, queue, grid_shape, dtype)
    grid_size = np.product(grid_shape)

    if proc_shape[0] * proc_shape[1] * proc_shape[2] == 1:
        rng = clr.ThreefryGenerator(ctx, seed=12321)
        fx = rng.uniform(queue, grid_shape, dtype) + 1.e-2
        fx1 = fx.get()

        fk = fft.dft(fx)
        fk1 = fk.get()
        fk_np = np.fft.rfftn(fx1)

        fx2 = fft.idft(fk).get()
        fx_np = np.fft.irfftn(fk1)

        rtol = 1.e-11 if dtype == np.float64 else 2.e-3
        assert np.allclose(fx1, fx2 / grid_size, rtol=rtol, atol=0), \
                "IDFT(DFT(f)) != f for grid_shape=%s" % (grid_shape,)

        assert np.allclose(fk_np, fk1, rtol=rtol, atol=0), \
                "DFT disagrees with numpy for grid_shape=%s" % (grid_shape,)

        assert np.allclose(fx_np, fx2 / grid_size, rtol=rtol, atol=0), \
                "IDFT disagrees with numpy for grid_shape=%s" % (grid_shape,)

    fx_cl = cla.empty(queue, rank_shape, dtype)
    pencil_shape = tuple(ni + 2*h for ni in rank_shape)
    fx_cl_halo = cla.empty(queue, pencil_shape, dtype)
    fx_np = np.empty(rank_shape, dtype)
    fx_np_halo = np.empty(pencil_shape, dtype)
    fk_cl = cla.empty(queue, fft.shape(True), fft.fk.dtype)
    fk_np = np.empty(fft.shape(True), fft.fk.dtype)

    # FIXME: check that these actually produce the correct result
    fx_types = {'cl': fx_cl, 'cl halo': fx_cl_halo,
                'np': fx_np, 'np halo': fx_np_halo,
                'None': None}

    fk_types = {'cl': fk_cl, 'np': fk_np, 'None': None}

    # run all of these to ensure no runtime errors even if no timing
    if timing:
        ntime = 20
    else:
        ntime = 1

    from common import timer

    if mpi.rank == 0:
        print("N = %s" % (grid_shape,))

    from itertools import product
    for (a, input_), (b, output) in product(fx_types.items(), fk_types.items()):
        t = timer(lambda: fft.dft(input_, output), ntime=ntime)
        if mpi.rank == 0:
            print("dft(%s, %s) took %.3f ms" % (a, b, t))

    for (a, input_), (b, output) in product(fk_types.items(), fx_types.items()):
        t = timer(lambda: fft.idft(input_, output), ntime=ntime)
        if mpi.rank == 0:
            print("idft(%s, %s) took %.3f ms" % (a, b, t))


if __name__ == "__main__":
    args = {'grid_shape': (256,)*3, 'proc_shape': (1,)*3, 'dtype': np.float64}
    from common import get_exec_arg_dict
    args.update(get_exec_arg_dict())
    test_dft(None, **args, timing=True)
