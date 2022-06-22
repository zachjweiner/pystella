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


@pytest.mark.parametrize("dtype", ["float64"])
@pytest.mark.parametrize("num_bins", [123, 1024, 1493])
@pytest.mark.parametrize("_N", [256, 1200])
def test_trivial_histogram(ctx_factory, grid_shape, proc_shape, dtype,
                           num_bins, _N, timing=False):
    ctx = ctx_factory()

    grid_shape = (_N,)*3
    queue = cl.CommandQueue(ctx)
    h = 1
    mpi = ps.DomainDecomposition(proc_shape, h, grid_shape=grid_shape)
    rank_shape, _ = mpi.get_rank_shape_start(grid_shape)

    histograms = {
        "a": (13., 1),
        "b": (10.3, 2),
        "c": (100.9, 3),
    }
    hist = ps.Histogrammer(mpi, histograms, num_bins, dtype, rank_shape=rank_shape)

    result = hist(queue)

    for key, (_b, weight) in histograms.items():
        res = result[key]
        b = int(np.floor(_b))
        expected = weight * np.product(grid_shape)
        assert res[b] == expected, \
            f"{key}: result={res[b]}, {expected=}, ratio={res[b]/expected}"
        assert np.all(res[res != res[b]] == 0.)


@pytest.mark.parametrize("dtype", ["float64", "float32"])
@pytest.mark.parametrize("num_bins", [123, 1024, 1493])
def test_histogram(ctx_factory, grid_shape, proc_shape, dtype, num_bins,
                   timing=False):
    ctx = ctx_factory()

    queue = cl.CommandQueue(ctx)
    h = 1
    mpi = ps.DomainDecomposition(proc_shape, h, grid_shape=grid_shape)
    rank_shape, _ = mpi.get_rank_shape_start(grid_shape)

    if np.dtype(dtype) in (np.dtype("float64"), np.dtype("complex128")):
        max_rtol, avg_rtol = 1e-10, 1e-11
    else:
        max_rtol, avg_rtol = 5e-4, 5e-5

    from pymbolic import var
    _fx = ps.Field("fx")
    histograms = {
        "count": (var("abs")(_fx) * num_bins, 1),
        "squared": (var("abs")(_fx) * num_bins, _fx**2),
    }
    hist = ps.Histogrammer(mpi, histograms, num_bins, dtype, rank_shape=rank_shape)

    rng = clr.ThreefryGenerator(ctx, seed=12321)
    fx = rng.uniform(queue, rank_shape, dtype)
    fx_h = fx.get()

    result = hist(queue, fx=fx)

    res = result["count"]
    assert np.sum(res.astype("int64")) == np.product(grid_shape), \
        f"Count histogram doesn't sum to grid_size ({np.sum(res)})"

    bins = np.linspace(0, 1, num_bins+1).astype(dtype)
    weights = np.ones_like(fx_h)
    np_res = np.histogram(fx_h, bins=bins, weights=weights)[0]
    np_res = mpi.allreduce(np_res)

    max_err, avg_err = get_errs(res, np_res)
    assert max_err < max_rtol and avg_err < avg_rtol, \
        f"Histogrammer inaccurate for grid_shape={grid_shape}" \
        f": {max_err=}, {avg_err=}"

    res = result["squared"]
    np_res = np.histogram(fx_h, bins=bins, weights=fx_h**2)[0]
    np_res = mpi.allreduce(np_res)

    max_err, avg_err = get_errs(res, np_res)
    assert max_err < max_rtol and avg_err < avg_rtol, \
        f"Histogrammer with weights inaccurate for grid_shape={grid_shape}" \
        f": {max_err=}, {avg_err=}"

    if timing:
        from common import timer
        t = timer(lambda: hist(queue, fx=fx))
        print(f"histogram took {t:.3f} ms for {grid_shape=}, {dtype=}")


@pytest.mark.parametrize("dtype", ["float64", "float32"])
def test_field_histogram(ctx_factory, grid_shape, proc_shape, dtype, timing=False):
    ctx = ctx_factory()

    queue = cl.CommandQueue(ctx)
    h = 1
    mpi = ps.DomainDecomposition(proc_shape, h, grid_shape=grid_shape)
    rank_shape, _ = mpi.get_rank_shape_start(grid_shape)
    pencil_shape = tuple(Ni + 2 * h for Ni in rank_shape)

    num_bins = 432

    if np.dtype(dtype) in (np.dtype("float64"), np.dtype("complex128")):
        max_rtol, avg_rtol = 1e-10, 1e-11
    else:
        max_rtol, avg_rtol = 5e-4, 5e-5

    hist = ps.FieldHistogrammer(mpi, num_bins, dtype,
                                rank_shape=rank_shape, halo_shape=h)

    rng = clr.ThreefryGenerator(ctx, seed=12321)
    fx = rng.uniform(queue, (2, 2)+pencil_shape, dtype, a=-1.2, b=3.)
    fx_h = fx.get()[..., h:-h, h:-h, h:-h]

    result = hist(fx)

    outer_shape = fx.shape[:-3]
    from itertools import product
    slices = list(product(*[range(n) for n in outer_shape]))

    for slc in slices:
        res = result["linear"][slc]
        np_res = np.histogram(fx_h[slc], bins=result["linear_bins"][slc])[0]
        np_res = mpi.allreduce(np_res)

        max_err, avg_err = get_errs(res, np_res)
        assert max_err < max_rtol and avg_err < avg_rtol, \
            f"linear Histogrammer inaccurate for grid_shape={grid_shape}" \
            f": {max_err=}, {avg_err=}"

        res = result["log"][slc]
        bins = result["log_bins"][slc]

        # avoid FPA comparison issues
        # numpy sometimes doesn't count the actual maximum/minimum
        eps = 1e-14 if np.dtype(dtype) == np.dtype("float64") else 1e-4
        bins[0] *= (1 - eps)
        bins[-1] *= (1 + eps)

        np_res = np.histogram(np.abs(fx_h[slc]), bins=bins)[0]
        np_res = mpi.allreduce(np_res)
        norm = np.maximum(np.abs(res), np.abs(np_res))
        norm[norm == 0.] = 1.

        max_err, avg_err = get_errs(res, np_res)
        assert max_err < max_rtol and avg_err < avg_rtol, \
            f"log Histogrammer inaccurate for grid_shape={grid_shape}" \
            f": {max_err=}, {avg_err=}"

    if timing:
        from common import timer
        t = timer(lambda: hist(fx[0, 0]))
        print(f"field histogram took {t:.3f} ms for {grid_shape=}, {dtype=}")


if __name__ == "__main__":
    from common import parser
    args = parser.parse_args()

    test_trivial_histogram(
        ps.choose_device_and_make_context,
        grid_shape=args.grid_shape, proc_shape=args.proc_shape,
        dtype="float64", timing=args.timing, num_bins=1493, _N=1200,
    )
    test_histogram(
        ps.choose_device_and_make_context,
        grid_shape=args.grid_shape, proc_shape=args.proc_shape,
        dtype=args.dtype, timing=args.timing, num_bins=1001,
    )
    test_field_histogram(
        ps.choose_device_and_make_context,
        grid_shape=args.grid_shape, proc_shape=args.proc_shape,
        dtype=args.dtype, timing=args.timing
    )
