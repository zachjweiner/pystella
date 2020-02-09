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


@pytest.mark.parametrize("dtype", ['float64', 'complex128'])
@pytest.mark.parametrize("random", [True, False])
def test_generate_WKB(ctx_factory, grid_shape, proc_shape, dtype, random,
                      timing=False):
    if ctx_factory:
        ctx = ctx_factory()
    else:
        ctx = ps.choose_device_and_make_context()

    queue = cl.CommandQueue(ctx)
    h = 1
    rank_shape = tuple(Ni // pi for Ni, pi in zip(grid_shape, proc_shape))
    mpi = ps.DomainDecomposition(proc_shape, h, rank_shape)

    fft = ps.DFT(mpi, ctx, queue, grid_shape, dtype)

    L = (10,)*3
    volume = np.product(L)
    dk = tuple(2 * np.pi / Li for Li in L)
    modes = ps.RayleighGenerator(ctx, fft, dk, volume)

    # only checking that this call is successful
    fk, dfk = modes.generate_WKB(queue, random=random)

    if timing:
        ntime = 10
        from common import timer
        t = timer(lambda: modes.generate_WKB(queue, random=random), ntime=ntime)
        print("%srandom, set_modes took %.3f ms for grid_shape=%s"
              % ('' if random else 'non-', t, grid_shape))


@pytest.mark.filterwarnings(
    "ignore::pyopencl.characterize.CLCharacterizationWarning")
@pytest.mark.filterwarnings("ignore::loopy.diagnostic.LoopyAdvisory")
@pytest.mark.parametrize("dtype", ['float64', 'complex128'])
@pytest.mark.parametrize("random", [True, False])
def test_generate(ctx_factory, grid_shape, proc_shape, dtype, random, timing=False):
    if ctx_factory:
        ctx = ctx_factory()
    else:
        ctx = ps.choose_device_and_make_context()

    queue = cl.CommandQueue(ctx)
    h = 1
    rank_shape = tuple(Ni // pi for Ni, pi in zip(grid_shape, proc_shape))
    mpi = ps.DomainDecomposition(proc_shape, h, rank_shape)

    fft = ps.DFT(mpi, ctx, queue, grid_shape, dtype)

    num_bins = int(sum(Ni**2 for Ni in grid_shape)**.5 / 2 + .5) + 1
    L = (10,)*3
    volume = np.product(L)
    dk = tuple(2 * np.pi / Li for Li in L)
    spectra = ps.PowerSpectra(mpi, fft, dk, volume)
    modes = ps.RayleighGenerator(ctx, fft, dk, volume, seed=5123)

    kbins = min(dk) * np.arange(0, num_bins)
    test_norm = 1 / 2 / np.pi**2 / np.product(grid_shape)**2

    for exp in [-1, -2, -3]:
        def power(k):
            return k**exp

        fk = modes.generate(queue, random=random, norm=1, field_ps=power)

        spectrum = spectra.norm * spectra.bin_power(fk, queue=queue, k_power=3)[1:-1]
        true_spectrum = test_norm * kbins[1:-1]**3 * power(kbins[1:-1])
        err = np.abs(1 - spectrum / true_spectrum)

        tol = .1 if num_bins < 64 else .3
        assert (np.max(err[num_bins//3:-num_bins//3]) < tol
                and np.average(err[1:]) < tol), \
            "init power spectrum incorrect for %srandom k**%d" \
            % ('' if random else 'non-', exp)

        if random:
            fx = fft.idft(cla.to_device(queue, fk)).real
            if isinstance(fx, cla.Array):
                fx = fx.get()

            grid_size = np.product(grid_shape)

            avg = mpi.allreduce(np.sum(fx)) / grid_size
            var = mpi.allreduce(np.sum(fx**2)) / grid_size - avg**2
            skew = mpi.allreduce(np.sum(fx**3)) / grid_size - 3 * avg * var - avg**3
            skew /= var**1.5
            assert skew < tol, \
                "init power spectrum has large skewness for k**%d" % (exp)

    if timing:
        ntime = 10
        from common import timer
        t = timer(lambda: modes.generate(queue, random=random), ntime=ntime)
        print("%srandom, set_modes took %.3f ms for grid_shape=%s"
              % ('' if random else 'non-', t, grid_shape))


def is_hermitian(fk):
    if isinstance(fk, cla.Array):
        fk = fk.get()

    grid_shape = list(fk.shape)
    grid_shape[-1] = 2 * (grid_shape[-1] - 1)
    pos = [np.arange(0, Ni//2+1) for Ni in grid_shape]
    neg = [np.concatenate([np.array([0]), np.arange(Ni-1, Ni//2-1, -1)])
           for Ni in grid_shape]

    test = np.array([])
    for k in [0, grid_shape[-1]//2]:
        for n, p in zip(neg[0], pos[0]):
            test = np.append(test, np.allclose(fk[n, neg[1], k],
                                               np.conj(fk[p, pos[1], k]),
                                               atol=0, rtol=1.e-12))
            test = np.append(test, np.allclose(fk[p, neg[1], k],
                                               np.conj(fk[n, pos[1], k]),
                                               atol=0, rtol=1.e-12))
        for n, p in zip(neg[1], pos[1]):
            test = np.append(test, np.allclose(fk[neg[0], n, k],
                                               np.conj(fk[pos[0], p, k]),
                                               atol=0, rtol=1.e-12))
            test = np.append(test, np.allclose(fk[neg[0], p, k],
                                               np.conj(fk[pos[0], n, k]),
                                               atol=0, rtol=1.e-12))

    for i in [0, grid_shape[0]//2]:
        for j in [0, grid_shape[1]//2]:
            for k in [0, grid_shape[2]//2]:
                test = np.append(test, [np.abs(np.imag(fk[i, j, k])) < 1.e-15])
    return test.all()


@pytest.mark.parametrize("dtype", ['float64'])
def test_make_hermitian(ctx_factory, grid_shape, proc_shape, dtype):
    if proc_shape != (1, 1, 1):
        pytest.skip("test make_hermitian only on one rank")

    kshape = (grid_shape[0], grid_shape[1], grid_shape[2]//2 + 1)
    data = np.random.rand(*kshape) + 1j * np.random.rand(*kshape)

    from pystella.fourier.rayleigh import make_hermitian
    data = make_hermitian(data)
    assert is_hermitian(data), "data is not hermitian"


if __name__ == "__main__":
    args = {'grid_shape': (32,)*3, 'proc_shape': (1, 1, 1), 'dtype': 'float64'}
    from common import get_exec_arg_dict
    args.update(get_exec_arg_dict())
    if args['proc_shape'] == (1, 1, 1):
        test_make_hermitian(None, **args)
    for random in [True, False]:
        test_generate_WKB(None, **args, random=random, timing=True)
        test_generate(None, **args, random=random, timing=True)
