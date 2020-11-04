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


def timer(kernel, ntime=200, nwarmup=2):
    for i in range(nwarmup):
        kernel()

    start = time()
    for i in range(ntime):
        evt = kernel()

    if isinstance(evt, cl.Event):
        evt.wait()

    end = time()

    return (end - start) / ntime * 1e3


class ArgumentParser(argparse.ArgumentParser):
    def parse_args(self, *args, **kwargs):
        args = super().parse_args(*args, **kwargs)

        args.proc_shape = tuple(args.proc_shape)
        args.grid_shape = tuple(args.grid_shape)

        return args


parser = ArgumentParser(add_help=False)
parser.add_argument('--help', action='help', help='show this help message and exit')
parser.add_argument('-proc', '--proc_shape', type=int, nargs=3, default=(1, 1, 1))
parser.add_argument('-grid', '--grid_shape', type=int, nargs=3, default=(256,)*3)
parser.add_argument('--h', '-h', type=int, default=2, metavar='h')
parser.add_argument('--dtype', '-dtype', type=np.dtype, default=np.float64)
parser.add_argument('--timing', '-time', type=bool, default=True)
