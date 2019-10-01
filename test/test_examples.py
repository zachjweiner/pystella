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


import os
import subprocess
import pytest

from pyopencl.tools import (  # noqa
        pytest_generate_tests_for_pyopencl as pytest_generate_tests)

examples = {
    'examples/wave_equation.py': None,
    'examples/scalar_preheating.py': 3.e-8,
}


@pytest.mark.parametrize("filename, expected", examples.items())
def test_examples(ctx_factory, grid_shape, proc_shape, filename, expected):
    if proc_shape[0] * proc_shape[1] * proc_shape[2] > 1:
        pytest.skip('run examples on only one rank')

    if os.environ.get('RUNNING_ON_AZURE', False):
        pytest.skip("can't run tests on Azure")

    result = subprocess.run(['python', filename, 'end_time', '1'],
                            stdout=subprocess.PIPE)

    assert result.returncode == 0, '%s failed' % filename

    if expected is not None:
        from glob import glob
        from h5py import File
        files = sorted(glob('20*.h5'))
        f = File(files[-1], 'r')
        constraint = f['energy/constraint'][-1]
        print(filename, constraint)
        f.close()
        os.remove(files[-1])

        assert constraint < expected, '%s constraint is wrong' % filename


if __name__ == "__main__":
    args = {'grid_shape': (256,)*3, 'proc_shape': (1,)*3}
    for example, expected in examples.items():
        test_examples(None, **args, filename=example, expected=expected)
