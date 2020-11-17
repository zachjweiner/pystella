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
from common import get_errs

from pyopencl.tools import (  # noqa
    pytest_generate_tests_for_pyopencl as pytest_generate_tests)


@pytest.mark.parametrize("h", [1, 2])
@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_scalar_energy(ctx_factory, grid_shape, proc_shape, h, dtype, timing=False):
    if ctx_factory:
        ctx = ctx_factory()
    else:
        ctx = ps.choose_device_and_make_context()

    queue = cl.CommandQueue(ctx)
    mpi = ps.DomainDecomposition(proc_shape, h, grid_shape=grid_shape)
    rank_shape, _ = mpi.get_rank_shape_start(grid_shape)

    grid_size = np.product(grid_shape)

    nscalars = 2

    def potential(f):
        phi, chi = f[0], f[1]
        return 1/2 * phi**2 + 1/2 * chi**2 + 1/2 * phi**2 * chi**2

    scalar_sector = ps.ScalarSector(nscalars, potential=potential)
    scalar_energy = ps.Reduction(mpi, scalar_sector, rank_shape=rank_shape,
                                 grid_size=grid_size, halo_shape=h)

    pencil_shape = tuple(ni+2*h for ni in rank_shape)
    f = clr.rand(queue, (nscalars,)+pencil_shape, dtype)
    dfdt = clr.rand(queue, (nscalars,)+pencil_shape, dtype)
    lap = clr.rand(queue, (nscalars,)+rank_shape, dtype)

    energy = scalar_energy(queue, f=f, dfdt=dfdt, lap_f=lap, a=np.array(1.))

    kin_test = []
    grad_test = []
    for fld in range(nscalars):
        df_h = dfdt[fld].get()
        rank_sum = np.sum(df_h[h:-h, h:-h, h:-h]**2)
        kin_test.append(1/2 * mpi.allreduce(rank_sum) / grid_size)

        f_h = f[fld].get()
        lap_h = lap[fld].get()

        rank_sum = np.sum(- f_h[h:-h, h:-h, h:-h] * lap_h)
        grad_test.append(1/2 * mpi.allreduce(rank_sum) / grid_size)

    energy_test = {}
    energy_test["kinetic"] = np.array(kin_test)
    energy_test["gradient"] = np.array(grad_test)

    phi = f[0].get()[h:-h, h:-h, h:-h]
    chi = f[1].get()[h:-h, h:-h, h:-h]
    pot_rank = np.sum(potential([phi, chi]))
    energy_test["potential"] = np.array(mpi.allreduce(pot_rank) / grid_size)

    max_rtol = 1e-14 if dtype == np.float64 else 1e-5
    avg_rtol = 1e-14 if dtype == np.float64 else 1e-5

    for key, value in energy.items():
        max_err, avg_err = get_errs(value, energy_test[key])
        assert max_err < max_rtol and avg_err < avg_rtol, \
            f"{key} inaccurate for {nscalars=}, {grid_shape=}, {proc_shape=}" \
            f": {max_err=}, {avg_err=}"

    if timing:
        from common import timer
        t = timer(lambda: scalar_energy(queue, a=np.array(1.),
                                        f=f, dfdt=dfdt, lap_f=lap))
        if mpi.rank == 0:
            print(f"scalar energy took {t:.3f} "
                  f"ms for {nscalars=}, {grid_shape=}, {proc_shape=}")


if __name__ == "__main__":
    from common import parser
    args = parser.parse_args()

    test_scalar_energy(
        None, grid_shape=args.grid_shape, proc_shape=args.proc_shape,
        h=args.h, dtype=args.dtype, timing=args.timing
    )
