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
import pystella as ps
import pytest

from pyopencl.tools import (  # noqa
    pytest_generate_tests_for_pyopencl as pytest_generate_tests)


@pytest.mark.filterwarnings(
    "ignore::pyopencl.characterize.CLCharacterizationWarning")
@pytest.mark.filterwarnings("ignore::loopy.diagnostic.LoopyAdvisory")
@pytest.mark.parametrize("dtype", [np.float64, np.float32])
@pytest.mark.parametrize("stream", [True, False])
def test_stencil(ctx_factory, grid_shape, proc_shape, dtype, stream, h=1,
                 timing=False):
    if ctx_factory:
        ctx = ctx_factory()
    else:
        ctx = ps.choose_device_and_make_context()

    queue = cl.CommandQueue(ctx)
    rank_shape = tuple(Ni // pi for Ni, pi in zip(grid_shape, proc_shape))

    from pymbolic import var
    x = var("x")
    y = var("y")
    i, j, k = var("i"), var("j"), var("k")

    map_dict = {}
    map_dict[y[i, j, k]] = (
        x[i + h + h, j + h, k + h]
        + x[i + h, j + h + h, k + h]
        + x[i + h, j + h, k + h + h]
        + x[i - h + h, j + h, k + h]
        + x[i + h, j - h + h, k + h]
        + x[i + h, j + h, k - h + h]
    )

    if stream:
        try:
            stencil_map = ps.StreamingStencil(
                map_dict, prefetch_args=["x"], halo_shape=h
            )
        except:  # noqa
            pytest.skip("StreamingStencil unavailable")
    else:
        stencil_map = ps.Stencil(map_dict, h, prefetch_args=["x"])

    x = clr.rand(queue, tuple(ni + 2*h for ni in rank_shape), dtype)
    y = clr.rand(queue, rank_shape, dtype)

    x_h = x.get()
    y_true = (
        x_h[2*h:, h:-h, h:-h]
        + x_h[h:-h, 2*h:, h:-h]
        + x_h[h:-h, h:-h, 2*h:]
        + x_h[:-2*h, h:-h, h:-h]
        + x_h[h:-h, :-2*h, h:-h]
        + x_h[h:-h, h:-h, :-2*h]
    )

    stencil_map(queue, x=x, y=y)

    rtol = 5.e-14 if dtype == np.float64 else 1.e-5

    assert np.allclose(y.get(), y_true, rtol=rtol, atol=0), \
        f"average innaccurate for {grid_shape=}, {h=}, {proc_shape=}"

    if timing:
        from common import timer
        t = timer(lambda: stencil_map(queue, x=x, y=y)[0])
        print(f"stencil took {t:.3f} ms for {grid_shape=}, {h=}, {proc_shape=}")
        bandwidth = (x.nbytes + y.nbytes) / 1024**3 / t * 1000
        print(f"Bandwidth = {bandwidth} GB/s")


if __name__ == "__main__":
    from common import parser
    args = parser.parse_args()

    for h in range(1, 4):
        for stream in [True, False]:
            test_stencil(
                None, grid_shape=args.grid_shape, proc_shape=args.proc_shape,
                dtype=args.dtype, timing=args.timing,
                stream=stream, h=h
            )
