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
import h5py


def get_versions(dependencies):
    import importlib
    from pkg_resources import get_distribution, DistributionNotFound
    from pytools import find_module_git_revision
    versions = {}
    git_revs = {}
    for dep in dependencies:
        try:
            versions[dep] = get_distribution(dep).version
        except (ModuleNotFoundError, DistributionNotFound):
            versions[dep] = None
        try:
            file = importlib.import_module(dep.replace('.', '')).__file__
            git_revs[dep] = find_module_git_revision(file, n_levels_up=1)
        except ModuleNotFoundError:
            git_revs[dep] = None
    return versions, git_revs


def append(dset, data):
    dset.resize(dset.shape[0]+1, axis=0)
    dset[-1] = data


class OutputFile:
    """
    A wrapper to :class:`h5py:File` which collects and saves useful run
    information and provides functionality to append to datasets.

    No arguments are required, but the following keyword arguments are
    recognized:

    :arg context: A :class:`pyopencl.Context`. If not *None*, information
        about the device, driver, and platform is saved to the
        :attr:`attrs` dictionary.
        Defaults to *None*.

    :arg name: The name of the ``.h5`` (sans the extension) file to create.
        If *None*, a unique filename is chosen based on the current date and
        time.
        Defaults to *None*.

    :arg runfile: A file whose content will be saved as a string to
        ``attrs['runfile']``, if not *None*. Useful for attaching the run file
        of a simulation to its output.
        Defaults to *None*.

    Any remaining keyword arguments are saved to the :attr:`attrs` dictionary.
    If any value ``val`` is not of valid type to be saved, the ``val.__name__``
    attribute is saved if the value is a :class:`type` instance, or else
    ``str(val)`` is saved.

    Versions and git revisions (when available) of :mod:`pystella` and its
    dependencies are saved as ``'versions'`` and ``'git_revs'``
    :class:`h5py:Dataset`\\ s. The hostname is recorded in the ``'hostname'``
    key of the :attr:`attrs` dictionary.

    .. automethod:: output
    """

    def create_from_kwargs(self, f, group, **kwargs):
        f.create_group(group)
        for key, val in kwargs.items():
            if not isinstance(val, np.ndarray):
                val = np.array(val)
            shape = (0,) + val.shape
            maxshape = (None,) + val.shape
            f[group].create_dataset(key, shape=shape, dtype=val.dtype,
                                    maxshape=maxshape, chunks=True)

    def __init__(self, context=None, name=None, runfile=None, **kwargs):
        if name is None:
            import datetime
            name = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        while True:
            try:
                self.filename = name + '.h5'
                self.file = h5py.File(self.filename, 'x')
                self.file.close()
                break
            except OSError:
                import time
                time.sleep(1)
                name = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        with self.open() as f:
            if context is not None:
                device, = context.devices
                f.attrs['device'] = device.name
                f.attrs['driver_version'] = device.driver_version
                f.attrs['platform_version'] = device.platform.version

            import socket
            f.attrs['hostname'] = socket.getfqdn()

            for key, val in kwargs.items():
                try:
                    f.attrs[key] = val
                except:  # noqa
                    if isinstance(val, type):
                        f.attrs[key] = val.__name__
                    else:
                        f.attrs[key] = str(val)

            if runfile is not None:
                fp = open(runfile)
                content = fp.read()
                fp.close()
                f.attrs['runfile'] = content

            # output current dependency versions
            dependencies = {'pystella', 'numpy', 'scipy',
                            'pyopencl', 'loopy', 'pymbolic',
                            'mpi4py', 'gpyfft', 'mpi4py_fft', 'h5py'}
            dependencies |= set(kwargs.pop('dependencies', {}))
            versions, git_revs = get_versions(dependencies)

            f.create_group('versions')
            for k, v in versions.items():
                f['versions'][k] = v or ''

            f.create_group('git_revs')
            for k, v in git_revs.items():
                f['git_revs'][k] = v or ''

    def open(self, mode="a"):
        return h5py.File(self.filename, mode)

    def output(self, group, **kwargs):
        """
        Appends values to datasets within a :class:`h5py:Group` named ``group``.
        ``group`` is created if it does not exist, and the :class:`h5py:Dataset`'s
        of this :class:`h5py:Group` are determined by the keys of keyword arguments.
        If ``group`` already exists, iterates over each :class:`h5py:Dataset` and
        appends values from keyword arguments (matching :class:`h5py:Dataset`
        names to keys).

        :arg group: The :class:`h5py:Group` to append :class:`h5py:Dataset`
            values to.

        If ``group`` already exists, a keyword argument for each
        :class:`h5py:Dataset` in ``group`` must be provided.
        """

        # create group and datasets if they don't exist
        with self.open() as f:
            if group not in f:
                self.create_from_kwargs(f, group, **kwargs)

            # ensure that all fields are provided
            for key in f[group]:
                val = kwargs.pop(key)
                append(f[group][key], val)
