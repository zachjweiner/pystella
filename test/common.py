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
from time import time
import pyopencl as cl
import argparse


def get_errs(a, b):
    a = np.atleast_1d(a)
    b = np.atleast_1d(b)
    mask = (a != 0.) | (b != 0.)
    a = a[mask]
    b = b[mask]
    err = np.abs((a - b) / np.maximum(np.abs(a), np.abs(b)))

    return np.max(err), np.average(err)


def timer(kernel, ntime=200, nwarmup=2):
    for _ in range(nwarmup):
        kernel()

    start = time()
    for _ in range(ntime):
        res = kernel()

    if isinstance(res, cl.Event):
        res.wait()
    elif isinstance(res, cl.array.Array):
        res.finish()

    end = time()

    return (end - start) / ntime * 1e3


class ArgumentParser(argparse.ArgumentParser):
    def parse_args(self, *args, **kwargs):
        args = super().parse_args(*args, **kwargs)

        args.proc_shape = tuple(args.proc_shape)
        args.grid_shape = tuple(args.grid_shape)

        return args


parser = ArgumentParser(add_help=False)
parser.add_argument("--help", action="help", help="show this help message and exit")
parser.add_argument("-proc", "--proc_shape", type=int, nargs=3, default=(1, 1, 1))
parser.add_argument("-grid", "--grid_shape", type=int, nargs=3, default=(256,)*3)
parser.add_argument("--h", "-h", type=int, default=2, metavar="h")
parser.add_argument("--dtype", "-dtype", type=np.dtype, default=np.float64)
parser.add_argument("--no-timing", dest="timing", default=True, action="store_false")
