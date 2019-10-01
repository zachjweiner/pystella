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


from time import time
import pyopencl as cl


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


def get_exec_arg_dict():
    """
    Interprets command line arguments (obtained from `sys.argv`) as key-value
    pairs. Entries corresponding to values are passed to :func:`eval` and stored
    as such, unless :func:`eval` raises an exception, in which case the string
    input itself is stored.

    :returns: A :class:`dict` of the command-line arguments.
    """

    def eval_unless_str(string):
        try:
            x = eval(string)
        except:  # noqa: E722
            x = string
        return x

    import sys
    ll = sys.argv[1:]
    return dict(zip(ll[::2], map(eval_unless_str, ll[1::2])))
