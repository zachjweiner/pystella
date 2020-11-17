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
import pystella as ps
import pytest

from pyopencl.tools import (  # noqa
    pytest_generate_tests_for_pyopencl as pytest_generate_tests)


@pytest.mark.filterwarnings(
    "ignore::pyopencl.characterize.CLCharacterizationWarning")
@pytest.mark.filterwarnings("ignore::loopy.diagnostic.LoopyAdvisory")
@pytest.mark.filterwarnings("ignore::loopy.diagnostic.ParameterFinderWarning")
@pytest.mark.parametrize("h", [2])
@pytest.mark.parametrize("dtype", [np.float64])
def test_transfer(ctx_factory, grid_shape, proc_shape, h, dtype, timing=False):
    if ctx_factory:
        ctx = ctx_factory()
    else:
        ctx = ps.choose_device_and_make_context()

    queue = cl.CommandQueue(ctx)
    rank_shape = tuple(Ni // pi for Ni, pi in zip(grid_shape, proc_shape))
    mpi = ps.DomainDecomposition(proc_shape, h, rank_shape)
    grid_shape_2 = tuple(Ni // 2 for Ni in grid_shape)
    rank_shape_2 = tuple(ni // 2 for ni in rank_shape)
    mpi2 = ps.DomainDecomposition(proc_shape, h, rank_shape_2)

    from pystella.multigrid import (Injection,
                                    FullWeighting,
                                    LinearInterpolation,
                                    # CubicInterpolation
                                    )

    inject = Injection(halo_shape=h, dtype=dtype)
    full_weighting = FullWeighting(halo_shape=h, dtype=dtype)

    def relerr(a, b):
        return np.max(np.abs(a-b))

    for restrict in [inject, full_weighting]:
        f1h = cla.zeros(queue, tuple(ni + 2*h for ni in rank_shape), dtype)
        f2h = cla.zeros(queue, tuple(ni + 2*h for ni in rank_shape_2), dtype)

        kvec = 2 * np.pi * np.array([1, 1, 1]).astype(dtype)

        xvecs = np.meshgrid(np.linspace(0, 1, grid_shape[0]+1)[:-1],
                            np.linspace(0, 1, grid_shape[1]+1)[:-1],
                            np.linspace(0, 1, grid_shape[2]+1)[:-1], indexing="ij")

        phases = kvec[0] * xvecs[0] + kvec[1] * xvecs[1] + kvec[2] * xvecs[2]
        mpi.scatter_array(queue, np.sin(phases), f1h, root=0)
        mpi.share_halos(queue, f1h)

        restrict(queue, f1=f1h, f2=f2h)

        restrict_error = relerr(f1h.get()[h:-h:2, h:-h:2, h:-h:2],
                                f2h.get()[h:-h, h:-h, h:-h])

        if restrict == inject:
            expected_error_bound = 1e-15
        else:
            expected_error_bound = .05 / (grid_shape[0]/32)**2

        assert restrict_error < expected_error_bound, \
            f"restrict innaccurate for {grid_shape=}, {h=}, {proc_shape=}"

    linear_interp = LinearInterpolation(halo_shape=h, dtype=dtype)
    # cubic_interp = CubicInterpolation(halo_shape=h, dtype=dtype)

    for interp in [linear_interp]:
        kvec = 2 * np.pi * np.array([1, 1, 1]).astype(dtype)

        xvecs = np.meshgrid(np.linspace(0, 1, grid_shape_2[0]+1)[:-1],
                            np.linspace(0, 1, grid_shape_2[1]+1)[:-1],
                            np.linspace(0, 1, grid_shape_2[2]+1)[:-1], indexing="ij")

        phases = kvec[0] * xvecs[0] + kvec[1] * xvecs[1] + kvec[2] * xvecs[2]
        mpi2.scatter_array(queue, np.sin(phases), f2h, root=0)
        mpi2.share_halos(queue, f2h)

        f1h_new = cla.zeros_like(f1h)
        interp(queue, f1=f1h_new, f2=f2h)
        mpi.share_halos(queue, f1h_new)

        interp_error = relerr(f1h_new.get(), f1h.get())

        # if interp == cubic_interp:
        #     expected_error_bound = .005 / (grid_shape[0]/32)**4
        # else:
        #     expected_error_bound = .1 / (grid_shape[0]/32)**2
        expected_error_bound = .1 / (grid_shape[0]/32)**2

        assert interp_error < expected_error_bound, \
            f"interp innaccurate for {grid_shape=}, {h=}, {proc_shape=}"


if __name__ == "__main__":
    from common import parser
    parser.set_defaults(grid_shape=(128,)*3)
    args = parser.parse_args()

    test_transfer(
        None, grid_shape=args.grid_shape, proc_shape=args.proc_shape,
        h=args.h, dtype=args.dtype, timing=args.timing
    )
