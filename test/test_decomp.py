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


params = [
    (1, None, True),
    (1, (128, 64, 32), True),
    (1, (109, 67, 32), True),
    (1, None, False),
    ((2, 0, 3), None, True),
    ((0, 2, 1), None, True),
]


@pytest.mark.parametrize("h, _grid_shape, pass_grid_shape", params)
@pytest.mark.parametrize("dtype", [np.float64])
def test_share_halos(ctx_factory, grid_shape, proc_shape, h, dtype,
                     _grid_shape, pass_grid_shape, timing=False):
    if ctx_factory:
        ctx = ctx_factory()
    else:
        ctx = ps.choose_device_and_make_context()

    if isinstance(h, int):
        h = (h,)*3

    queue = cl.CommandQueue(ctx)
    grid_shape = _grid_shape or grid_shape
    mpi = ps.DomainDecomposition(
        proc_shape, h, grid_shape=(grid_shape if pass_grid_shape else None)
    )
    rank_shape, substart = mpi.get_rank_shape_start(grid_shape)

    # data will be same on each rank
    rng = clr.ThreefryGenerator(ctx, seed=12321)
    data = rng.uniform(queue,
                       tuple(Ni + 2*hi for Ni, hi in zip(grid_shape, h)),
                       dtype).get()
    if h[0] > 0:
        data[:h[0], :, :] = data[-2*h[0]:-h[0], :, :]
        data[-h[0]:, :, :] = data[h[0]:2*h[0], :, :]
    if h[1] > 0:
        data[:, :h[1], :] = data[:, -2*h[1]:-h[1], :]
        data[:, -h[1]:, :] = data[:, h[1]:2*h[1], :]
    if h[2] > 0:
        data[:, :, :h[2]] = data[:, :, -2*h[2]:-h[2]]
        data[:, :, -h[2]:] = data[:, :, h[2]:2*h[2]]

    subdata = np.empty(tuple(ni + 2*hi for ni, hi in zip(rank_shape, h)), dtype)
    rank_slice = tuple(slice(si + hi, si + ni + hi)
                       for ni, si, hi in zip(rank_shape, substart, h))
    unpadded_slc = tuple(slice(hi, -hi) if hi > 0 else slice(None) for hi in h)
    subdata[unpadded_slc] = data[rank_slice]

    subdata_device = cla.to_device(queue, subdata)
    mpi.share_halos(queue, subdata_device)
    subdata2 = subdata_device.get()

    pencil_slice = tuple(slice(si, si + ri + 2*hi)
                         for ri, si, hi in zip(rank_shape, substart, h))
    assert (subdata2 == data[pencil_slice]).all(), \
        f"rank {mpi.rank} {mpi.rank_tuple} has incorrect halo data"

    # test that can call with different-shaped input
    if not pass_grid_shape:
        subdata_device_new = clr.rand(
            queue, tuple(ni//2 + 2*hi for ni, hi in zip(rank_shape, h)), dtype
        )
        mpi.share_halos(queue, subdata_device_new)

    if timing:
        from common import timer
        t = timer(lambda: mpi.share_halos(queue, fx=subdata_device))
        if mpi.rank == 0:
            print(f"share_halos took {t:.3f} ms for "
                  f"{grid_shape=}, {h=}, {proc_shape=}")


@pytest.mark.parametrize("h, _grid_shape, pass_grid_shape", params)
@pytest.mark.parametrize("dtype", [np.float64])
def test_gather_scatter(ctx_factory, grid_shape, proc_shape, h, dtype,
                        _grid_shape, pass_grid_shape, timing=False):
    if ctx_factory:
        ctx = ctx_factory()
    else:
        ctx = ps.choose_device_and_make_context()

    if isinstance(h, int):
        h = (h,)*3

    queue = cl.CommandQueue(ctx)
    grid_shape = _grid_shape or grid_shape
    mpi = ps.DomainDecomposition(proc_shape, h)
    rank_shape, substart = mpi.get_rank_shape_start(grid_shape)

    rank_slice = tuple(slice(si, si + ri)
                       for ri, si, hi in zip(rank_shape, substart, h))
    pencil_shape = tuple(ni + 2*hi for ni, hi in zip(rank_shape, h))

    unpadded_slc = tuple(slice(hi, -hi) if hi > 0 else slice(None) for hi in h)

    # create random data with same seed on all ranks
    rng = clr.ThreefryGenerator(ctx, seed=12321)
    data = rng.uniform(queue, grid_shape, dtype)

    # cl.Array -> cl.Array
    subdata = cla.zeros(queue, pencil_shape, dtype)
    mpi.scatter_array(queue, data if mpi.rank == 0 else None, subdata, 0)
    sub_h = subdata.get()
    data_h = data.get()
    assert (sub_h[unpadded_slc] == data_h[rank_slice]).all()

    data_test = cla.zeros_like(data)
    mpi.gather_array(queue, subdata, data_test if mpi.rank == 0 else None, 0)
    data_test_h = data_test.get()
    if mpi.rank == 0:
        assert (data_test_h == data_h).all()

    # np.ndarray -> np.ndarray
    mpi.scatter_array(queue, data_h if mpi.rank == 0 else None, sub_h, 0)
    assert (sub_h[unpadded_slc] == data_h[rank_slice]).all()

    mpi.gather_array(queue, sub_h, data_test_h if mpi.rank == 0 else None, 0)
    if mpi.rank == 0:
        assert (data_test_h == data_h).all()

    # scatter cl.Array -> np.ndarray
    sub_h[:] = 0
    mpi.scatter_array(queue, data if mpi.rank == 0 else None, sub_h, 0)
    assert (sub_h[unpadded_slc] == data_h[rank_slice]).all()

    # gather np.ndarray -> cl.Array
    data_test[:] = 0
    mpi.gather_array(queue, sub_h, data_test if mpi.rank == 0 else None, 0)
    data_test_h = data_test.get()
    if mpi.rank == 0:
        assert (data_test_h == data_h).all()

    # scatter np.ndarray -> cl.Array
    subdata[:] = 0
    mpi.scatter_array(queue, data_h if mpi.rank == 0 else None, subdata, 0)
    sub_h = subdata.get()
    assert (sub_h[unpadded_slc] == data_h[rank_slice]).all()

    # gather cl.Array -> np.ndarray
    data_test_h[:] = 0
    mpi.gather_array(queue, subdata, data_test_h if mpi.rank == 0 else None, 0)
    if mpi.rank == 0:
        assert (data_test_h == data_h).all()

    if timing:
        from common import timer
        ntime = 25
        times = {}

        times['scatter cl.Array -> cl.Array'] = \
            timer(lambda: mpi.scatter_array(queue, data, subdata, 0), ntime=ntime)
        times['scatter cl.Array -> np.ndarray'] = \
            timer(lambda: mpi.scatter_array(queue, data, sub_h, 0), ntime=ntime)
        times['scatter np.ndarray -> cl.Array'] = \
            timer(lambda: mpi.scatter_array(queue, data_h, subdata, 0), ntime=ntime)
        times['scatter np.ndarray -> np.ndarray'] = \
            timer(lambda: mpi.scatter_array(queue, data_h, sub_h, 0), ntime=ntime)

        times['gather cl.Array -> cl.Array'] = \
            timer(lambda: mpi.gather_array(queue, subdata, data, 0), ntime=ntime)
        times['gather cl.Array -> np.ndarray'] = \
            timer(lambda: mpi.gather_array(queue, subdata, data_h, 0), ntime=ntime)
        times['gather np.ndarray -> cl.Array'] = \
            timer(lambda: mpi.gather_array(queue, sub_h, data, 0), ntime=ntime)
        times['gather np.ndarray -> np.ndarray'] = \
            timer(lambda: mpi.gather_array(queue, sub_h, data_h, 0), ntime=ntime)

        if mpi.rank == 0:
            print(f"{grid_shape=}, {h=}, {proc_shape=}")
            for key, val in times.items():
                print(f"{key} took {val:.3f} ms")


if __name__ == "__main__":
    from common import parser
    parser.add_argument('--pass_grid_shape', type=bool, default=True)
    args = parser.parse_args()

    test_share_halos(
        None, grid_shape=args.grid_shape, proc_shape=args.proc_shape,
        h=args.h, dtype=args.dtype, timing=args.timing,
        pass_grid_shape=args.pass_grid_shape, _grid_shape=None
    )
    test_gather_scatter(
        None, grid_shape=args.grid_shape, proc_shape=args.proc_shape,
        h=args.h, dtype=args.dtype, timing=args.timing,
        pass_grid_shape=args.pass_grid_shape, _grid_shape=None
    )
