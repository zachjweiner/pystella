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
@pytest.mark.parametrize("op", ['avg', 'sum', 'max'])
@pytest.mark.parametrize("_grid_shape", [None, (128, 64, 32)])
@pytest.mark.parametrize("pass_grid_dims", [True, False])
def test_reduction(ctx_factory, grid_shape, proc_shape, dtype, op,
                   _grid_shape, pass_grid_dims, timing=False):
    if ctx_factory:
        ctx = ctx_factory()
    else:
        ctx = ps.choose_device_and_make_context()

    queue = cl.CommandQueue(ctx)
    h = 1
    grid_shape = _grid_shape or grid_shape
    rank_shape = tuple(Ni // pi for Ni, pi in zip(grid_shape, proc_shape))
    mpi = ps.DomainDecomposition(proc_shape, h, rank_shape)

    from pymbolic import var
    from pystella import Field
    tmp_insns = [(var('x'), Field('f') / 2 + .31)]

    reducers = {}
    reducers['avg'] = [(var('x'), op)]

    if pass_grid_dims:
        reducer = ps.Reduction(mpi, reducers, rank_shape=rank_shape,
                               tmp_instructions=tmp_insns,
                               grid_size=np.product(grid_shape))
    else:
        reducer = ps.Reduction(mpi, reducers, tmp_instructions=tmp_insns)

    f = clr.rand(queue, rank_shape, dtype=dtype)

    import pyopencl.tools as clt
    pool = clt.MemoryPool(clt.ImmediateAllocator(queue))

    result = reducer(queue, f=f, allocator=pool)
    avg = result['avg']

    avg_test = reducer.reduce_array(f / 2 + .31, op)
    if op == 'avg':
        avg_test /= np.product(grid_shape)

    rtol = 5.e-14 if dtype == np.float64 else 1.e-5
    assert np.allclose(avg, avg_test, rtol=rtol, atol=0), \
        "%s reduction innaccurate for grid_shape=%s, proc_shape=%s" \
        % (op, grid_shape, proc_shape)

    if timing:
        from common import timer
        t = timer(lambda: reducer(queue, f=f, allocator=pool), ntime=1000)
        if mpi.rank == 0:
            print("reduction took %.3f ms for grid_shape=%s, proc_shape=%s"
                  % (t, grid_shape, proc_shape))
            print("Bandwidth = %.1f GB/s"
                  % ((f.nbytes)/1024**3 / t * 1000))


@pytest.mark.parametrize("dtype", [np.float64])
@pytest.mark.parametrize("op", ['avg'])
@pytest.mark.parametrize("_grid_shape", [None, (128, 64, 32)])
def test_reduction_with_new_shape(ctx_factory, grid_shape, proc_shape, dtype, op,
                                  _grid_shape, timing=False):
    if ctx_factory:
        ctx = ctx_factory()
    else:
        ctx = ps.choose_device_and_make_context()

    queue = cl.CommandQueue(ctx)
    h = 1
    grid_shape = _grid_shape or grid_shape
    rank_shape = tuple(Ni // pi for Ni, pi in zip(grid_shape, proc_shape))
    mpi = ps.DomainDecomposition(proc_shape, h, rank_shape)

    from pystella import Field
    reducers = {}
    reducers['avg'] = [(Field('f'), op)]

    reducer = ps.Reduction(mpi, reducers)

    f = clr.rand(queue, rank_shape, dtype=dtype)
    result = reducer(queue, f=f)
    avg = result['avg']

    avg_test = reducer.reduce_array(f, op)
    if op == 'avg':
        avg_test /= np.product(grid_shape)

    rtol = 5.e-14 if dtype == np.float64 else 1.e-5
    assert np.allclose(avg, avg_test, rtol=rtol, atol=0), \
        "%s reduction innaccurate for grid_shape=%s, proc_shape=%s" \
        % (op, grid_shape, proc_shape)

    # test call to reducer with new shape
    grid_shape = tuple(Ni // 2 for Ni in grid_shape)
    rank_shape = tuple(Ni // pi for Ni, pi in zip(grid_shape, proc_shape))
    f = clr.rand(queue, rank_shape, dtype=dtype)
    result = reducer(queue, f=f)
    avg = result['avg']

    avg_test = reducer.reduce_array(f, op)
    if op == 'avg':
        avg_test /= np.product(grid_shape)

    rtol = 5.e-14 if dtype == np.float64 else 1.e-5
    assert np.allclose(avg, avg_test, rtol=rtol, atol=0), \
        "%s reduction w/new shape innaccurate for grid_shape=%s, proc_shape=%s" \
        % (op, grid_shape, proc_shape)


@pytest.mark.parametrize("dtype", [np.float64])
@pytest.mark.parametrize("_grid_shape", [None, (128, 64, 32)])
@pytest.mark.parametrize("pass_grid_dims", [True, False])
def test_field_statistics(ctx_factory, grid_shape, proc_shape, dtype, _grid_shape,
                          pass_grid_dims, timing=False):
    if ctx_factory:
        ctx = ctx_factory()
    else:
        ctx = ps.choose_device_and_make_context()

    queue = cl.CommandQueue(ctx)
    h = 1
    grid_shape = _grid_shape or grid_shape
    rank_shape = tuple(Ni // pi for Ni, pi in zip(grid_shape, proc_shape))
    mpi = ps.DomainDecomposition(proc_shape, h, rank_shape)

    # make select parameters local for convenience
    h = 2
    f = clr.rand(queue, (2, 1)+tuple(ni + 2*h for ni in rank_shape), dtype=dtype)

    if pass_grid_dims:
        statistics = ps.FieldStatistics(mpi, h, rank_shape=rank_shape,
                                        grid_size=np.product(grid_shape))
    else:
        statistics = ps.FieldStatistics(mpi, h)

    import pyopencl.tools as clt
    pool = clt.MemoryPool(clt.ImmediateAllocator(queue))

    stats = statistics(f, allocator=pool)
    avg = stats['mean']
    var = stats['variance']

    f_h = f.get()
    rank_sum = np.sum(f_h[..., h:-h, h:-h, h:-h], axis=(-3, -2, -1))
    avg_test = mpi.allreduce(rank_sum) / np.product(grid_shape)

    rank_sum = np.sum(f_h[..., h:-h, h:-h, h:-h]**2, axis=(-3, -2, -1))
    var_test = mpi.allreduce(rank_sum) / np.product(grid_shape) - avg_test**2

    rtol = 5.e-14 if dtype == np.float64 else 1.e-5

    assert np.allclose(avg, avg_test, rtol=rtol, atol=0), \
        "average innaccurate for grid_shape=%s, proc_shape=%s" \
        % (grid_shape, proc_shape)

    assert np.allclose(var, var_test, rtol=rtol, atol=0), \
        "variance innaccurate for grid_shape=%s, proc_shape=%s" \
        % (grid_shape, proc_shape)

    if timing:
        from common import timer
        t = timer(lambda: statistics(f, allocator=pool))
        if mpi.rank == 0:
            print("field stats took "
                  "%.3f ms for outer shape %s, grid_shape=%s, proc_shape=%s"
                  % (t, f.shape[:-3], grid_shape, proc_shape))


if __name__ == "__main__":
    args = {'grid_shape': (256,)*3, 'proc_shape': (1,)*3,
            'dtype': np.float64, '_grid_shape': None}
    from common import get_exec_arg_dict
    args.update(get_exec_arg_dict())
    for op in ['avg', 'sum', 'max']:
        test_reduction(None, **args, op=op, pass_grid_dims=True, timing=True)
    test_reduction_with_new_shape(None, **args, op='avg')
    test_field_statistics(None, **args, pass_grid_dims=True, timing=True)
