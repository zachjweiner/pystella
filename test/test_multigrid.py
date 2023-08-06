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

from pystella.multigrid import (FullApproximationScheme, MultiGridSolver,
                                NewtonIterator)


@pytest.mark.filterwarnings("ignore::loopy.diagnostic.ParameterFinderWarning")
@pytest.mark.parametrize("h", [1])
@pytest.mark.parametrize("dtype", [np.float64])
@pytest.mark.parametrize("Solver", [NewtonIterator])
@pytest.mark.parametrize("MG", [FullApproximationScheme, MultiGridSolver])
def test_multigrid(ctx_factory, grid_shape, proc_shape, h, dtype, Solver, MG,
                   timing=False):
    ctx = ctx_factory()

    queue = cl.CommandQueue(ctx)
    rank_shape = tuple(Ni // pi for Ni, pi in zip(grid_shape, proc_shape))
    mpi = ps.DomainDecomposition(proc_shape, h, rank_shape)

    L = 10
    dx = L / grid_shape[0]

    statistics = ps.FieldStatistics(mpi, h, rank_shape=rank_shape,
                                    grid_size=np.prod(grid_shape))

    def get_laplacian(f):
        from pystella.derivs import _lap_coefs, centered_diff
        lap_coefs = _lap_coefs[h]
        from pymbolic import var
        return sum([centered_diff(f, lap_coefs, direction=mu, order=2)
                    for mu in range(1, 4)]) / var("dx")**2

    test_problems = {}

    from pystella import Field
    f = Field("f", offset="h")
    rho = Field("rho", offset="h")
    test_problems[f] = (get_laplacian(f), rho)

    f = Field("f2", offset="h")
    rho = Field("rho2", offset="h")
    test_problems[f] = (get_laplacian(f) - f, rho)

    solver = Solver(mpi, queue, test_problems, halo_shape=h, dtype=dtype,
                    fixed_parameters=dict(omega=1/2))
    mg = MG(solver=solver, halo_shape=h, dtype=dtype)

    def zero_mean_array():
        f0 = clr.rand(queue, grid_shape, dtype)
        f = clr.rand(queue, tuple(ni + 2*h for ni in rank_shape), dtype)
        mpi.scatter_array(queue, f0, f, root=0)
        avg = statistics(f)["mean"].item()
        f = f - avg
        mpi.share_halos(queue, f)
        return f

    f = zero_mean_array()
    rho = zero_mean_array()

    f2 = zero_mean_array()
    rho2 = zero_mean_array()

    poisson_errs = []
    helmholtz_errs = []
    num_v_cycles = 15 if MG == MultiGridSolver else 10
    for _ in range(num_v_cycles):
        errs = mg(mpi, queue, dx0=dx, f=f, rho=rho, f2=f2, rho2=rho2)
        poisson_errs.append(errs[-1][-1]["f"])
        helmholtz_errs.append(errs[-1][-1]["f2"])

    for name, cycle_errs in zip(["poisson", "helmholtz"],
                                [poisson_errs, helmholtz_errs]):
        tol = 1e-6 if MG == MultiGridSolver else 5e-14
        assert cycle_errs[-1][1] < tol and cycle_errs[-2][1] < 10*tol, \
            f"multigrid solution to {name} eqn is inaccurate for " \
            f"{grid_shape=}, {h=}, {proc_shape=}\n{cycle_errs=}"


if __name__ == "__main__":
    from common import parser
    parser.set_defaults(grid_shape=(128,)*3)
    args = parser.parse_args()

    test_multigrid(
        ps.choose_device_and_make_context,
        grid_shape=args.grid_shape, proc_shape=args.proc_shape,
        h=args.h, dtype=args.dtype, timing=args.timing,
        Solver=NewtonIterator, MG=FullApproximationScheme
    )
