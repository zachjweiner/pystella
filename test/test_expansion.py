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
import pystella as ps
import pytest

from pyopencl.tools import (  # noqa
    pytest_generate_tests_for_pyopencl as pytest_generate_tests)


# for deprecated call to logger.warn in loopy c_execution
@pytest.mark.filterwarnings("ignore::DeprecationWarning")
@pytest.mark.parametrize("dtype", [np.float64])
@pytest.mark.parametrize("Stepper", [ps.RungeKutta4, ps.LowStorageRK54])
def test_expansion(ctx_factory, proc_shape, dtype, Stepper, timing=False):
    if proc_shape != (1, 1, 1):
        pytest.skip("test expansion only on one rank")

    def sol(w, t):
        x = (1 + 3*w)
        return (x*(t/np.sqrt(3) + 2/x))**(2/x)/2**(2/x)

    from pystella.step import LowStorageRKStepper
    is_low_storage = LowStorageRKStepper in Stepper.__bases__

    for w in [0, 1/3, 1/2, 1, -1/4]:
        def energy(a):
            return a**(-3-3*w)  # noqa: B023

        def pressure(a):
            return w * energy(a)  # noqa: B023

        t = 0
        dt = .005
        expand = ps.Expansion(energy(1.), Stepper, mpl=np.sqrt(8.*np.pi))

        while t <= 10. - dt:
            for s in range(expand.stepper.num_stages):
                slc = (0) if is_low_storage else (0 if s == 0 else 1)
                expand.step(s, energy(expand.a[slc]), pressure(expand.a[slc]), dt)
            t += dt

        slc = () if is_low_storage else (0)

        order = expand.stepper.expected_order
        rtol = dt**order

        print(order,
              w,
              expand.a[slc]/sol(w, t) - 1,
              expand.constraint(energy(expand.a[slc])))

        assert np.allclose(expand.a[slc], sol(w, t), rtol=rtol, atol=0), \
            f"FLRW solution inaccurate for {w=}"

        assert expand.constraint(energy(expand.a[slc])) < rtol, \
            f"FLRW solution disobeying constraint for {w=}"


if __name__ == "__main__":
    from common import parser
    args = parser.parse_args()

    from pystella.step import all_steppers
    for stepper in all_steppers[-5:]:
        test_expansion(
            None, proc_shape=args.proc_shape, dtype=args.dtype, timing=args.timing,
            Stepper=stepper,
        )
