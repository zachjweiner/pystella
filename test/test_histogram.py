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


@pytest.mark.filterwarnings(
    "ignore::pyopencl.characterize.CLCharacterizationWarning")
@pytest.mark.filterwarnings("ignore::loopy.diagnostic.LoopyAdvisory")
@pytest.mark.parametrize("dtype", ['float64', 'float32'])
def test_histogram(ctx_factory, grid_shape, proc_shape, dtype, timing=False):
    if ctx_factory:
        ctx = ctx_factory()
    else:
        ctx = ps.choose_device_and_make_context()

    queue = cl.CommandQueue(ctx)
    h = 1
    mpi = ps.DomainDecomposition(proc_shape, h, grid_shape=grid_shape)
    rank_shape, _ = mpi.get_rank_shape_start(grid_shape)

    num_bins = 193

    if np.dtype(dtype) in (np.dtype('float64'), np.dtype('complex128')):
        max_rtol, avg_rtol = 1e-12, 1e-13
    else:
        max_rtol, avg_rtol = 5e-4, 5e-5

    from pystella.histogram import Histogrammer
    from pymbolic import var

    _fx = ps.Field('fx')
    histograms = {
        'count': (var('abs')(_fx) * num_bins, 1),
        'squared': (var('abs')(_fx) * num_bins, _fx**2),
    }
    hist = Histogrammer(mpi, histograms, num_bins, rank_shape, dtype)

    rng = clr.ThreefryGenerator(ctx, seed=12321)
    fx = rng.uniform(queue, rank_shape, dtype)
    fx_h = fx.get()

    result = hist(queue, fx=fx)

    res = result['count']
    np_res = np.histogram(fx_h, bins=np.linspace(0, 1, num_bins+1))[0]
    np_res = mpi.allreduce(np_res)
    err = np.abs((res - np_res) / np.maximum(np.abs(res), np.abs(np_res)))

    assert np.max(err) < max_rtol and np.average(err) < avg_rtol, \
        f"Histogrammer inaccurate for grid_shape={grid_shape}"

    res = result['squared']
    np_res = np.histogram(fx_h, bins=np.linspace(0, 1, num_bins+1),
                          weights=fx_h**2)[0]
    np_res = mpi.allreduce(np_res)
    err = np.abs((res - np_res) / np.maximum(np.abs(res), np.abs(np_res)))

    assert np.max(err) < max_rtol and np.average(err) < avg_rtol, \
        f"Histogrammer with weights inaccurate for grid_shape={grid_shape}"

    if timing:
        from common import timer
        t = timer(lambda: hist(queue, fx=fx))
        print(f"histogram took {t:.3f} ms for {grid_shape=}, {dtype=}")


@pytest.mark.filterwarnings(
    "ignore::pyopencl.characterize.CLCharacterizationWarning")
@pytest.mark.filterwarnings("ignore::loopy.diagnostic.LoopyAdvisory")
@pytest.mark.parametrize("dtype", ['float64', 'float32'])
def test_field_histogram(ctx_factory, grid_shape, proc_shape, dtype, timing=False):
    if ctx_factory:
        ctx = ctx_factory()
    else:
        ctx = ps.choose_device_and_make_context()

    queue = cl.CommandQueue(ctx)
    h = 1
    mpi = ps.DomainDecomposition(proc_shape, h, grid_shape=grid_shape)
    rank_shape, _ = mpi.get_rank_shape_start(grid_shape)
    pencil_shape = tuple(Ni + 2 * h for Ni in rank_shape)

    num_bins = 432

    if np.dtype(dtype) in (np.dtype('float64'), np.dtype('complex128')):
        max_rtol, avg_rtol = 1e-12, 1e-13
    else:
        max_rtol, avg_rtol = 5e-4, 5e-5

    from pystella.histogram import FieldHistogrammer
    hist = FieldHistogrammer(mpi, num_bins, rank_shape, dtype, halo_shape=h)

    rng = clr.ThreefryGenerator(ctx, seed=12321)
    fx = rng.uniform(queue, (2, 2)+pencil_shape, dtype, a=-1.2, b=3.)
    fx_h = fx.get()[..., h:-h, h:-h, h:-h]

    result = hist(fx)

    outer_shape = fx.shape[:-3]
    from itertools import product
    slices = list(product(*[range(n) for n in outer_shape]))

    for slc in slices:
        res = result['linear'][slc]
        np_res = np.histogram(fx_h[slc], bins=result['linear_bins'][slc])[0]
        np_res = mpi.allreduce(np_res)
        err = np.abs((res - np_res) / np.maximum(np.abs(res), np.abs(np_res)))

        assert np.max(err) < max_rtol and np.average(err) < avg_rtol, \
            f"Histogrammer inaccurate for grid_shape={grid_shape}"

        res = result['log'][slc]
        bins = result['log_bins'][slc]

        # avoid FPA comparison issues
        # numpy sometimes doesn't count the actual maximum/minimum
        eps = 1e-14 if np.dtype(dtype) == np.dtype('float64') else 1.e-4
        bins[0] *= (1 - eps)
        bins[-1] *= (1 + eps)

        np_res = np.histogram(np.abs(fx_h[slc]), bins=bins)[0]
        np_res = mpi.allreduce(np_res)
        norm = np.maximum(np.abs(res), np.abs(np_res))
        norm[norm == 0.] = 1.
        err = np.abs((res - np_res) / norm)

        assert np.max(err) < max_rtol and np.average(err) < avg_rtol, \
            f"log Histogrammer inaccurate for grid_shape={grid_shape}"

    if timing:
        from common import timer
        t = timer(lambda: hist(fx[0, 0]))
        print(f"field histogram took {t:.3f} ms for {grid_shape=}, {dtype=}")


if __name__ == "__main__":
    args = {'grid_shape': (256,)*3, 'proc_shape': (1,)*3, 'dtype': 'float64'}
    from common import get_exec_arg_dict
    args.update(get_exec_arg_dict())
    test_histogram(None, **args, timing=True)
    test_field_histogram(None, **args, timing=True)
