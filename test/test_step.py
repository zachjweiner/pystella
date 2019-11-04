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
        k_tmp = cla.zeros(queue, (1,)+rank_shape, dtype=dtype)
        y0 = y.copy()
    else:
        num_copies = Stepper.num_copies
        y = cla.zeros(queue, (num_copies,)+rank_shape, dtype)
        y[0, 0, 0, :] = init_vals
        k_tmp = None
        y0 = y[0].copy()

    dtlist = [.1, .05, .025]

    for n in [-1., -2., -3., -4.]:
        max_errs = {}
        for dt in dtlist:
            def sol(y0, t):
                return ((-1 + n)*(-t + y0**(1 - n)/(-1 + n)))**(1/(1 - n))

            _y = ps.Field('y')
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
                    stepper(s, queue=queue, y=y, k_tmp=k_tmp, filter_args=True)
                t += dt

                if is_low_storage:
                    errs.append(cla.max(clm.fabs(1. - sol(y0, t)/y)).get())
                else:
                    errs.append(cla.max(clm.fabs(1. - sol(y0, t)/y[0])).get())

            max_errs[dt] = np.max(errs)

        order = stepper.expected_order
        print('order = %d' % order, 'n = %d' % n)
        print(max_errs)
        print([max_errs[a] / max_errs[b] for a, b in zip(dtlist[:-1], dtlist[1:])])

        order = stepper.expected_order
        rtol = dtlist[-1]**order if dtype == np.float64 else 1.e-1
        assert list(max_errs.values())[-1] < rtol, \
               "Stepper solution inaccurate for n=%f" % (n)

        for a, b in zip(dtlist[:-1], dtlist[1:]):
            assert max_errs[a] / max_errs[b] > .9 * (a/b)**order, \
                "Stepper convergence failing for n=%f" % (n)


if __name__ == "__main__":
    for stepper in all_steppers:
        test_step(None, (1, 1, 1), np.float64, stepper)
