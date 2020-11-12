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


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_elementwise(ctx_factory, grid_shape, proc_shape, dtype, timing=False):
    if ctx_factory:
        ctx = ctx_factory()
    else:
        ctx = ps.choose_device_and_make_context()

    queue = cl.CommandQueue(ctx)
    rank_shape = tuple(Ni // pi for Ni, pi in zip(grid_shape, proc_shape))

    from pymbolic import var
    a = var("a")
    b = var("b")

    from pystella.field import Field
    x = Field("x")
    y = Field("y")
    z = Field("z")

    tmp_dict = {a[0]: x + 2,
                a[1]: 2 + x * y,
                b: x + y / 2}
    map_dict = {x: a[0] * y**2 * x + a[1] * b,
                z: z + a[1] * b}
    single_insn = {x: y + z}

    ew_map = ps.ElementWiseMap(map_dict, tmp_instructions=tmp_dict)

    x = clr.rand(queue, rank_shape, dtype=dtype)
    y = clr.rand(queue, rank_shape, dtype=dtype)
    z = clr.rand(queue, rank_shape, dtype=dtype)

    a0 = x + 2
    a1 = 2 + x * y
    b = x + y / 2
    x_true = a0 * y**2 * x + a1 * b
    z_true = z + a1 * b

    ew_map(queue, x=x, y=y, z=z)

    rtol = 5.e-14 if dtype == np.float64 else 1.e-5

    assert np.allclose(x.get(), x_true.get(), rtol=rtol, atol=0), \
        f"x innaccurate for {grid_shape=}, {proc_shape=}"

    assert np.allclose(z.get(), z_true.get(), rtol=rtol, atol=0), \
        f"z innaccurate for {grid_shape=}, {proc_shape=}"

    # test success of single instruction
    ew_map_single = ps.ElementWiseMap(single_insn)
    ew_map_single(queue, x=x, y=y, z=z)

    assert np.allclose(x.get(), y.get() + z.get(), rtol=rtol, atol=0), \
        f"x innaccurate for {grid_shape=}, {proc_shape=}"

    if timing:
        from common import timer
        t = timer(lambda: ew_map(queue, x=x, y=y, z=z)[0])
        print(f"elementwise map took {t:.3f} ms for {grid_shape=}, {proc_shape=}")
        bandwidth = 5 * x.nbytes/1024**3 / t * 1000
        print(f"Bandwidth = {bandwidth:.1f} GB/s")


if __name__ == "__main__":
    from common import parser
    args = parser.parse_args()

    test_elementwise(
        None, grid_shape=args.grid_shape, proc_shape=args.proc_shape,
        dtype=args.dtype, timing=args.timing
    )
