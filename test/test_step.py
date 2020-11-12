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
import loopy as lp
import pystella as ps
import pytest

from pyopencl.tools import (  # noqa
    pytest_generate_tests_for_pyopencl as pytest_generate_tests)

# this only tests Stepper's correctness as an ODE solver
from pystella.step import all_steppers


@pytest.mark.parametrize("dtype", [np.float64])
@pytest.mark.parametrize("Stepper", all_steppers)
def test_step(ctx_factory, proc_shape, dtype, Stepper):
    if proc_shape != (1, 1, 1):
        pytest.skip("test step only on one rank")

    if ctx_factory:
        ctx = ctx_factory()
    else:
        ctx = ps.choose_device_and_make_context()

    queue = cl.CommandQueue(ctx)

    from pystella.step import LowStorageRKStepper
    is_low_storage = LowStorageRKStepper in Stepper.__bases__

    rank_shape = (1, 1, 8)
    init_vals = np.linspace(1, 3, 8)
    if is_low_storage:
        y = cla.zeros(queue, rank_shape, dtype)
        y[0, 0, :] = init_vals
        y0 = y.copy()
    else:
        num_copies = Stepper.num_copies
        y = cla.zeros(queue, (num_copies,)+rank_shape, dtype)
        y[0, 0, 0, :] = init_vals
        y0 = y[0].copy()

    dtlist = [.1, .05, .025]

    for n in [-1., -2., -3., -4.]:
        max_errs = {}
        for dt in dtlist:
            def sol(y0, t):
                return ((-1 + n)*(-t + y0**(1 - n)/(-1 + n)))**(1/(1 - n))

            _y = ps.Field("y")
            rhs_dict = {_y: _y**n}

            stepper = Stepper(rhs_dict, dt=dt, halo_shape=0, rank_shape=rank_shape)

            if is_low_storage:
                y[0, 0, :] = init_vals
            else:
                y[0, 0, 0, :] = init_vals

            t = 0
            errs = []
            while t < .1:
                for s in range(stepper.num_stages):
                    stepper(s, queue=queue, y=y, filter_args=True)
                t += dt

                if is_low_storage:
                    errs.append(cla.max(clm.fabs(1. - sol(y0, t)/y)).get())
                else:
                    errs.append(cla.max(clm.fabs(1. - sol(y0, t)/y[0])).get())

            max_errs[dt] = np.max(errs)

        order = stepper.expected_order
        print(f"{order=}, {n=}")
        print(max_errs)
        print([max_errs[a] / max_errs[b] for a, b in zip(dtlist[:-1], dtlist[1:])])

        order = stepper.expected_order
        rtol = dtlist[-1]**order if dtype == np.float64 else 1.e-1
        assert list(max_errs.values())[-1] < rtol, \
            f"Stepper solution inaccurate for {n=}"

        for a, b in zip(dtlist[:-1], dtlist[1:]):
            assert max_errs[a] / max_errs[b] > .9 * (a/b)**order, \
                f"Stepper convergence failing for {n=}"


def test_low_storage_edge_codegen_and_tmp_alloc(ctx_factory, proc_shape, dtype=None):
    if proc_shape != (1, 1, 1):
        pytest.skip("test step only on one rank")

    if ctx_factory:
        ctx = ctx_factory()
    else:
        ctx = ps.choose_device_and_make_context()

    queue = cl.CommandQueue(ctx)

    from pystella import LowStorageRK54
    from pymbolic import parse

    rhs_dict = {
        parse("y[i, j, k]"): 1,
    }
    stepper = LowStorageRK54(rhs_dict, dt=.1, halo_shape=0)
    y = cla.zeros(queue, (8, 8, 8), "complex128")
    tmp_arrays = stepper.get_tmp_arrays_like(y=y)
    assert tmp_arrays["_y_tmp"].shape == y.shape
    assert tmp_arrays["_y_tmp"].dtype == y.dtype
    stepper(0, queue=queue, y=y)
    tmp_for_check = stepper.tmp_arrays["_y_tmp"]
    stepper(1, queue=queue, y=y)
    assert tmp_for_check is stepper.tmp_arrays["_y_tmp"]

    rhs_dict = {
        parse("y"): 1,
    }
    stepper = LowStorageRK54(rhs_dict, args=[lp.GlobalArg("y", shape=tuple())],
                             dt=.1, halo_shape=0)
    y = np.zeros(1)
    tmp_arrays = stepper.get_tmp_arrays_like(y=y)
    assert tmp_arrays["_y_tmp"].shape == y.shape
    assert tmp_arrays["_y_tmp"].dtype == y.dtype
    # kernel won't work
    # stepper(0, queue=queue, y=y)
    # tmp_for_check = stepper.tmp_arrays["_y_tmp"]
    # stepper(1, queue=queue, y=y)
    # assert tmp_for_check is stepper.tmp_arrays["_y_tmp"]

    rhs_dict = {
        ps.Field(parse("y[0, 0]"), shape=(2, 2)): 1,
        ps.Field(parse("y[0, 1]"), shape=(2, 2)): 1,
        ps.Field(parse("y[1, 0]"), shape=(2, 2)): 1,
        ps.Field(parse("y[1, 1]"), shape=(2, 2)): 1,
    }
    stepper = LowStorageRK54(rhs_dict, dt=.1, halo_shape=0)
    y = cla.zeros(queue, (2, 2, 12, 12, 12), "float64")
    tmp_arrays = stepper.get_tmp_arrays_like(y=y)
    assert tmp_arrays["_y_tmp"].shape == y.shape
    assert tmp_arrays["_y_tmp"].dtype == y.dtype
    stepper(0, queue=queue, y=y)
    tmp_for_check = stepper.tmp_arrays["_y_tmp"]
    stepper(1, queue=queue, y=y)
    assert tmp_for_check is stepper.tmp_arrays["_y_tmp"]

    rhs_dict = {
        ps.Field("y", shape=(1, 2))[0, 1]: 1,
    }
    stepper = LowStorageRK54(rhs_dict, dt=.1, halo_shape=0)
    y = cla.zeros(queue, (1, 2, 12, 12, 12), "float64")
    tmp_arrays = stepper.get_tmp_arrays_like(y=y)
    assert tmp_arrays["_y_tmp"].shape == y.shape
    assert tmp_arrays["_y_tmp"].dtype == y.dtype
    stepper(0, queue=queue, y=y)
    tmp_for_check = stepper.tmp_arrays["_y_tmp"]
    stepper(1, queue=queue, y=y)
    assert tmp_for_check is stepper.tmp_arrays["_y_tmp"]

    rhs_dict = {
        ps.Field("y"): 1,
        ps.Field("z"): 1,
    }
    stepper = LowStorageRK54(rhs_dict, dt=.1, halo_shape=0)
    y = cla.zeros(queue, (12, 12, 12), "float64")
    z = cla.zeros(queue, (12, 12, 12), "complex128")
    tmp_arrays = stepper.get_tmp_arrays_like(y=y, z=z)
    assert tmp_arrays["_y_tmp"].shape == y.shape
    assert tmp_arrays["_y_tmp"].dtype == y.dtype
    assert tmp_arrays["_z_tmp"].shape == z.shape
    assert tmp_arrays["_z_tmp"].dtype == z.dtype


if __name__ == "__main__":
    test_low_storage_edge_codegen_and_tmp_alloc(None, (1, 1, 1), np.float64)
    for stepper in all_steppers:
        test_step(None, (1, 1, 1), np.float64, stepper)
