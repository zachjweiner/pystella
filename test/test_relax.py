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
import pystella as ps
import pytest

from pyopencl.tools import (  # noqa
        pytest_generate_tests_for_pyopencl as pytest_generate_tests)

from pystella.multigrid import JacobiIterator, NewtonIterator


@pytest.mark.parametrize("h", [1])
@pytest.mark.parametrize("dtype", [np.float64])
@pytest.mark.parametrize("Solver", [JacobiIterator, NewtonIterator])
def test_relax(ctx_factory, grid_shape, proc_shape, h, dtype, Solver, timing=False):
    if min(grid_shape) < 128:
        pytest.skip("test_relax needs larger grids, for now")

    if ctx_factory:
        ctx = ctx_factory()
    else:
        ctx = ps.choose_device_and_make_context()

    queue = cl.CommandQueue(ctx)
    rank_shape = tuple(Ni // pi for Ni, pi in zip(grid_shape, proc_shape))
    mpi = ps.DomainDecomposition(proc_shape, h, rank_shape)

    L = 10
    dx = L / grid_shape[0]
    dk = 2 * np.pi / L

    fft = ps.DFT(mpi, ctx, queue, grid_shape, dtype)
    spectra = ps.PowerSpectra(mpi, fft, (dk,)*3, L**3)
    statistics = ps.FieldStatistics(mpi, h, rank_shape=rank_shape,
                                    grid_size=np.product(grid_shape))

    def get_laplacian(f):
        from pystella.derivs import _lap_coefs, centered_diff
        lap_coefs = _lap_coefs[h]
        from pymbolic import var
        return sum([centered_diff(f, lap_coefs, direction=mu, order=2)
                    for mu in range(1, 4)]) / var('dx')**2

    test_problems = {}

    from pystella import Field
    f = Field('f', offset='h')
    rho = Field('rho', offset='h')
    test_problems[f] = (get_laplacian(f), rho)

    f = Field('f2', offset='h')
    rho = Field('rho2', offset='h')
    test_problems[f] = (get_laplacian(f) - f, rho)

    solver = Solver(mpi, queue, test_problems, h=h, dtype=dtype,
                    fixed_parameters=dict(omega=1/2))

    def zero_mean_array():
        f0 = clr.rand(queue, grid_shape, dtype)
        f = clr.rand(queue, tuple(ni + 2*h for ni in rank_shape), dtype)
        mpi.scatter_array(queue, f0, f, root=0)
        avg = statistics(f)['mean']
        f = f - avg
        mpi.share_halos(queue, f)
        return f

    f = zero_mean_array()
    rho = zero_mean_array()
    tmp = cla.zeros_like(f)

    f2 = zero_mean_array()
    rho2 = zero_mean_array()
    tmp2 = cla.zeros_like(f)

    num_iterations = 1000
    errors = {'f': [], 'f2': []}
    first_mode_zeroed = {'f': [], 'f2': []}
    for i in range(0, num_iterations, 2):
        solver(mpi, queue, iterations=2, dx=np.array(dx),
               f=f, tmp_f=tmp, rho=rho,
               f2=f2, tmp_f2=tmp2, rho2=rho2)

        err = solver.get_error(queue,
                               f=f, r_f=tmp, rho=rho,
                               f2=f2, r_f2=tmp2, rho2=rho2, dx=np.array(dx))
        for k, v in err.items():
            errors[k].append(v)

        for key, resid in zip(['f', 'f2'], [tmp, tmp2]):
            spectrum = spectra(resid, k_power=0)
            if mpi.rank == 0:
                max_amp = np.max(spectrum)
                first_zero = np.argmax(spectrum[1:] < 1.e-30 * max_amp)
                first_mode_zeroed[key].append(first_zero)

    for k, errs in errors.items():
        errs = np.array(errs)
        iters = np.arange(1, errs.shape[0]+1)  # pylint: disable=E1136
        assert (errs[10:, 0] * iters[10:] / errs[0, 0] < 1.).all(), \
            "relaxation not converging at least linearly for " \
            "grid_shape=%s, h=%d, proc_shape=%s" \
            % (grid_shape, h, proc_shape)

    first_mode_zeroed = mpi.bcast(first_mode_zeroed, root=0)
    for k, x in first_mode_zeroed.items():
        x = np.array(list(x))[2:]
        assert (x[1:] <= x[:-1]).all() and np.min(x) < np.max(x) / 5, \
            "relaxation not smoothing error grid_shape=%s, h=%d, proc_shape=%s" \
            % (grid_shape, h, proc_shape)


if __name__ == "__main__":
    args = {'grid_shape': (128,)*3, 'proc_shape': (1,)*3,
            'dtype': np.float64, 'h': 1}
    from common import get_exec_arg_dict
    args.update(get_exec_arg_dict())
    test_relax(None, **args, Solver=NewtonIterator, timing=True)
