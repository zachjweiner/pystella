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
import pyopencl.array as cla
import loopy as lp

import logging
logger = logging.getLogger(__name__)


class DomainDecomposition:
    """
    Implements functions needed for the MPI domain decomposition of a 3D grid.

    If :mod:`mpi4py` is not installed, then only single-rank operation is supported.

    :arg proc_shape: A 3-:class:`tuple` specifying the shape of the MPI
        processor grid.

        .. note::

            Currently, ``proc_shape[2]`` must be ``1``, i.e., only
            two-dimensional domain decompositions are supported.

    :arg halo_shape: The number of halo layers on (both sides of) each axis of
        the computational grid (i.e., those used for grid synchronization or
        enforcing boundary conditions).
        May be a 3-:class:`tuple` or an :class:`int` (which is interpreted
        as that value for each axis).
        Note that zero is a valid value along any axis (in which case grid
        synchronization for such axes is skipped).
        For example, ``halo_shape=(2,0,1)`` specifies that there are two
        additional layers at each the beginning and end of the first axis, none
        for the second, and one for the third.
        For ``rank_shape=(Nx, Ny, Nz)``, an appropriately padded array would
        then have shape ``(Nx+4, Ny, Nz+2)``.

    The following keyword arguments are recognized:

    :arg rank_shape: A 3-:class:`tuple` specifying the shape of the computational
        sub-grid on the calling process.
        Defaults to *None*, in which case the global size is not fixed (and
        will be inferred when, e.g., :meth:`share_halos` is called, at a slight
        performance penalty).

    :arg grid_shape: A 3-:class:`tuple` specifying the shape of the global
        computational grid.
        If passed, :attr:`rank_shape` is determined for each rank.
        Note that :attr:`proc_shape` need not evenly divide ``grid_shape``, i.e.,
        different ranks may have different :attr:`rank_shape`\\ s.

    :raises NotImplementedError: if ``proc_shape[2] != 1``.

    :raises ValueError: if the size of the processor grid
        ``proc_shape[0] * proc_shape[1] * proc_shape[2]`` is not equal to the
        total number of ranks the application was launched with
        (i.e., that returned by :func:`mpi4py.MPI.COMM_WORLD.Get_size()`).

    :raises ValueError: if both ``rank_shape`` and ``grid_shape`` are passed.

    .. automethod:: share_halos
    .. automethod:: remove_halos
    .. automethod:: gather_array
    .. automethod:: restore_halos
    .. automethod:: scatter_array
    .. automethod:: rankID
    .. autoattribute:: rank_tuple
    .. automethod:: bcast
    .. automethod:: allreduce
    .. automethod:: get_rank_shape_start

    .. attribute:: comm

        An :class:`mpi4py.MPI.COMM_WORLD` if :mod:`mpi4py` is installed, else *None*.

    .. attribute:: rank

        The integral rank of the calling process, i.e., that returned by
        :meth:`mpi4py.MPI.COMM_WORLD.Get_rank`.

    .. attribute:: nranks

        The total number of ranks, i.e., that returned by
        :meth:`mpi4py.MPI.COMM_WORLD.Get_size`.

    .. attribute:: proc_shape

    .. attribute:: rank_shape
    """

    def __init__(self, proc_shape, halo_shape, rank_shape=None, grid_shape=None):
        self.proc_shape = tuple(proc_shape)
        self.halo_shape = ((halo_shape,)*3 if isinstance(halo_shape, int)
                           else halo_shape)
        self.buffer_arrays = {}
        self.rank_shape = rank_shape

        try:
            from mpi4py import MPI
            self.comm = MPI.COMM_WORLD
            self.rank = self.comm.Get_rank()
            self.nranks = self.comm.Get_size()
        except ModuleNotFoundError:
            self.comm = None
            self.rank = 0
            self.nranks = 1

        if proc_shape[2] != 1:
            raise NotImplementedError("decomposition in z not yet supported")

        if proc_shape[0] * proc_shape[1] * proc_shape[2] != self.nranks:
            raise ValueError(
                f"{proc_shape} is an invalid decomposition for {self.nranks} ranks"
            )

        self.rz = self.rank % proc_shape[2]
        self.ry = (self.rank - self.rz) // proc_shape[2] % proc_shape[1]
        self.rx = (self.rank - self.rz - proc_shape[2] * self.ry) // proc_shape[1]

        if grid_shape is not None:
            if self.rank_shape is not None:
                raise ValueError("Only pass one of rank_shape or grid_shape.")
            self.rank_shape, _ = self.get_rank_shape_start(grid_shape)

        params_to_fix = dict(hx=self.halo_shape[0],
                             hy=self.halo_shape[1],
                             hz=self.halo_shape[2])
        if self.rank_shape is not None:
            for k, v in zip(("Nx", "Ny", "Nz"), self.rank_shape):
                params_to_fix[k] = v

        pencil_shape_str = "(Nx+2*hx, Ny+2*hy, Nz+2*hz)"
        # so loopy is assured of no out-of-bounds accesses:
        assumptions = "Nx > 0 and Ny > 0 and Nz > 0"

        def x_comm_knl(instructions):
            knl = lp.make_kernel(
                "[Nx, Ny, Nz, hx, hy, hz] \
                -> { [i,j,k]: 0<=i<hx and 0<=j<Ny+2*hy and 0<=k<Nz+2*hz }",
                instructions,
                [
                    lp.GlobalArg("arr", shape=pencil_shape_str, offset=lp.auto),
                    lp.GlobalArg("buf", shape="(2, hx, Ny+2*hy, Nz+2*hz)"),
                    ...,
                ],
                default_offset=lp.auto,
                lang_version=(2018, 2),
                assumptions=assumptions,
            )
            knl = lp.remove_unused_arguments(knl)
            knl = lp.fix_parameters(knl, **params_to_fix)
            knl = lp.split_iname(knl, "k", 32, outer_tag="g.0", inner_tag="l.0")
            knl = lp.split_iname(knl, "j", 2, outer_tag="g.1", inner_tag="l.1")
            return knl

        self.pack_x_knl = x_comm_knl(["buf[1, i, j, k] = arr[Nx+i, j, k]",
                                      "buf[0, i, j, k] = arr[i+hx, j, k]"])
        self.unpack_x_knl = x_comm_knl(["arr[i, j, k] = buf[1, i, j, k]",
                                        "arr[Nx+hx+i, j, k] = buf[0, i, j, k]"])
        self.pack_unpack_x_knl = x_comm_knl(["arr[i, j, k] = arr[Nx+i, j, k]",
                                             "arr[Nx+hx+i, j, k] = arr[i+hx, j, k]"])

        def y_comm_knl(instructions):
            knl = lp.make_kernel(
                "[Nx, Ny, Nz, hx, hy, hz] \
                -> { [i,j,k]: 0<=i<Nx+2*hx and 0<=j<hy and 0<=k<Nz+2*hz }",
                instructions,
                [
                    lp.GlobalArg("arr", shape=pencil_shape_str, offset=lp.auto),
                    lp.GlobalArg("buf", shape="(2, Nx+2*hx, hy, Nz+2*hz)"),
                    ...,
                ],
                default_offset=lp.auto,
                lang_version=(2018, 2),
                assumptions=assumptions,
            )
            knl = lp.remove_unused_arguments(knl)
            knl = lp.fix_parameters(knl, **params_to_fix)
            knl = lp.split_iname(knl, "k", 32, outer_tag="g.0", inner_tag="l.0")
            knl = lp.split_iname(knl, "i", 2, outer_tag="g.1", inner_tag="l.1")
            return knl

        self.pack_y_knl = y_comm_knl(["buf[1, i, j, k] = arr[i, Ny+j, k]",
                                      "buf[0, i, j, k] = arr[i, j+hy, k]"])
        self.unpack_y_knl = y_comm_knl(["arr[i, j, k] = buf[1, i, j, k]",
                                        "arr[i, Ny+hy+j, k] = buf[0, i, j, k]"])
        self.pack_unpack_y_knl = y_comm_knl(["arr[i, j, k] = arr[i, Ny+j, k]",
                                             "arr[i, Ny+hy+j, k] = arr[i, j+hy, k]"])

        def z_comm_knl(instructions):
            knl = lp.make_kernel(
                "[Nx, Ny, Nz, hx, hy, hz] \
                 -> { [i,j,k]: 0<=i<Nx+2*hx and 0<=j<Ny+2*hy and 0<=k<hz }",
                instructions,
                [
                    lp.GlobalArg("arr", shape=pencil_shape_str, offset=lp.auto),
                    ...,
                ],
                default_offset=lp.auto,
                lang_version=(2018, 2),
                assumptions=assumptions,
            )
            knl = lp.remove_unused_arguments(knl)
            knl = lp.fix_parameters(knl, **params_to_fix)
            knl = lp.split_iname(knl, "k", self.halo_shape[2],
                                 outer_tag="g.0", inner_tag="l.0")
            knl = lp.split_iname(knl, "j", 8, outer_tag="g.1", inner_tag="l.1")
            knl = lp.split_iname(knl, "i", 1, outer_tag="g.2", inner_tag="l.2")
            return knl

        self.pack_unpack_z_knl = z_comm_knl(["arr[i, j, k] = arr[i, j, Nz+k]",
                                             "arr[i, j, Nz+hz+k] = arr[i, j, k+hz]"])

        def make_G_S_knl(instructions):
            knl = lp.make_kernel(
                "[Nx, Ny, Nz] -> { [i,j,k]: 0<=i<Nx and 0<=j<Ny and 0<=k<Nz }",
                instructions,
                [
                    lp.GlobalArg("subarr", shape=pencil_shape_str, offset=lp.auto),
                    lp.GlobalArg("arr", shape="(Nx, Ny, Nz)", offset=lp.auto),
                    ...,
                ],
                default_offset=lp.auto,
                lang_version=(2018, 2),
            )
            knl = lp.fix_parameters(knl, **params_to_fix)
            knl = lp.split_iname(knl, "k", 32, outer_tag="g.0", inner_tag="l.0")
            knl = lp.split_iname(knl, "j", 2, outer_tag="g.1", inner_tag="unr")
            knl = lp.split_iname(knl, "i", 1, outer_tag="g.2", inner_tag="unr")
            return knl

        self.gather_knl = make_G_S_knl("arr[i, j, k] = subarr[i+hx, j+hy, k+hz]")
        self.scatter_knl = make_G_S_knl("subarr[i+hx, j+hy, k+hz] = arr[i, j, k]")

        self._kernels_bound = False

    def _bind_kernels(self, queue_or_context):
        self.pack_x_knl = self.pack_x_knl.executor(queue_or_context)
        self.unpack_x_knl = self.unpack_x_knl.executor(queue_or_context)
        self.pack_unpack_x_knl = self.pack_unpack_x_knl.executor(queue_or_context)
        self.pack_y_knl = self.pack_y_knl.executor(queue_or_context)
        self.unpack_y_knl = self.unpack_y_knl.executor(queue_or_context)
        self.pack_unpack_y_knl = self.pack_unpack_y_knl.executor(queue_or_context)
        self.pack_unpack_z_knl = self.pack_unpack_z_knl.executor(queue_or_context)
        self.gather_knl = self.gather_knl.executor(queue_or_context)
        self.scatter_knl = self.scatter_knl.executor(queue_or_context)
        self._kernels_bound = True

    def get_displs_and_counts(self, full_shape, x_slice):
        NX, NY, NZ = full_shape

        displs = np.ones(np.product(self.proc_shape[:2]), dtype="int")
        counts = np.ones(np.product(self.proc_shape[:2]), dtype="int")
        for ri in range(self.proc_shape[0]):
            for rj in range(self.proc_shape[1]):
                (Nx, Ny, Nz), (sx, sy, sz) = self.get_rank_shape_start(
                    full_shape, (ri, rj, 0)
                )
                rid = self.rankID(ri, rj, 0)
                # repeat the last row when some ranks have smaller rank_shape[0]
                displs[rid] = NZ * (sy + NY * sx) + min(x_slice, Nx-1) * NY * NZ
                counts[rid] = Nz * Ny

        return displs, counts

    def rankID(self, rx, ry, rz):
        """
        :returns: The (integer) rank number corresponding to the processor grid
            site, ``(rx, ry, rz)``.
        """

        rx_ = (rx + self.proc_shape[0]) % self.proc_shape[0]
        ry_ = (ry + self.proc_shape[1]) % self.proc_shape[1]
        rz_ = (rz + self.proc_shape[2]) % self.proc_shape[2]
        return rz_ + self.proc_shape[2] * (ry_ + self.proc_shape[1] * rx_)

    @property
    def rank_tuple(self):
        """
        A 3-:class:`tuple` containing the MPI rank's location in the processor grid.
        """

        return (self.rx, self.ry, self.rz)

    def get_rank_shape_start(self, grid_shape, rank_tuple=None):
        """
        Computes the calling process's ``rank_shape`` and starting indices
        ``rank_start`` (as would be indexed into the global array)
        corresponding to a global array with shape ``grid_shape``.

        :arg grid_shape: A 3-:class:`tuple` specifying the shape of the
            corresponding global array.

        :arg rank_tuple: A 3-:class:`tuple` specifying for which MPI rank
            in the processor grid for which to compute outputs.
            Defaults to that of the calling process (i.e., :attr:`rank_tuple`).

        :returns: ``(rank_shape, rank_start)``.
        """

        # matches decomposition by mpi4py_fft.pencil
        def get_size_start(N, size, rank):
            q, r = divmod(N, size)
            size = q + (1 if r > rank else 0)  # increase size by 1 for first r ranks
            start = rank * q + min(rank, r)
            return (size, start)

        rank_shape = [None] * 3
        rank_start = [None] * 3
        rank_tuple = rank_tuple or self.rank_tuple
        for i, (Npi, ri) in enumerate(zip(self.proc_shape, rank_tuple)):
            n_i, s_i = get_size_start(grid_shape[i], Npi, ri)
            rank_shape[i] = n_i
            rank_start[i] = s_i

        return tuple(rank_shape), tuple(rank_start)

    def _get_or_create_halo_buffers(self, queue, shape, dtype):
        if (shape, dtype) not in self.buffer_arrays:
            # map_to_host on multiple ranks causes problems on pocl
            on_pocl = queue.device.platform.name == "Portable Computing Language"

            buf_dev = cla.zeros(queue, shape, dtype)
            buf_send = buf_dev.map_to_host() if not on_pocl else buf_dev.get()
            buf_recv = buf_dev.map_to_host() if not on_pocl else buf_dev.get()
            self.buffer_arrays[(shape, dtype)] = buf_dev, buf_send, buf_recv

        return self.buffer_arrays[(shape, dtype)]

    def Barrier(self):
        if self.comm is not None:
            self.comm.Barrier()

    def _debug_barrier(self, *args, **kwargs):
        if logger.isEnabledFor(logging.DEBUG):
            self.Barrier()
            if self.rank == 0:
                logger.debug(*args, **kwargs)

    def _debug(self, *args, **kwargs):
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(*args, **kwargs)

    def share_halos(self, queue, fx):
        """
        Communicates halo data across all axes, imposing periodic boundary
        conditions.

        :arg queue: The :class:`pyopencl.CommandQueue` to enqueue kernels and copies.

        :arg fx: The :class:`pyopencl.array.Array` whose halo elements are to be
            synchronized across ranks.
            The number of halo layers on each face of the grid is fixed by
            :attr:`halo_shape`, while the shape of ``fx``
            (i.e., subtracting halo padding specified by :attr:`halo_shape`)
            is only fixed if ``rank_shape`` or ``grid_shape`` were
            passed at object creation).
        """

        if not self._kernels_bound:
            self._bind_kernels(queue)

        h = self.halo_shape
        rank_shape = tuple(ni - 2 * hi for ni, hi in zip(fx.shape, h))

        if h[2] > 0:
            if self.proc_shape[2] == 1:
                evt, _ = self.pack_unpack_z_knl(queue, arr=fx)
            else:
                raise NotImplementedError("domain decomposition in z direction")

        if h[0] > 0:
            if self.proc_shape[0] == 1:
                evt, _ = self.pack_unpack_x_knl(queue, arr=fx)
            else:
                self._debug_barrier("all ranks beginning halo share in x")
                shape = (2, h[0], rank_shape[1] + 2*h[1], rank_shape[2] + 2*h[2])
                buf_dev, buf_send, buf_recv = self._get_or_create_halo_buffers(
                    queue, shape, fx.dtype)

                evt, _ = self.pack_x_knl(queue, arr=fx, buf=buf_dev)
                buf_dev.get(ary=buf_send)

                dest = self.rankID(self.rx-1, self.ry, self.rz)
                source = self.rankID(self.rx+1, self.ry, self.rz)
                self._debug(f"initiating first halo share in x: {dest=}, {source=}")
                self.comm.Sendrecv(sendbuf=buf_send[0], dest=dest,
                                   recvbuf=buf_recv[0], source=source)

                dest = self.rankID(self.rx+1, self.ry, self.rz)
                source = self.rankID(self.rx-1, self.ry, self.rz)
                self._debug(f"initiating second halo share in x: {dest=}, {source=}")
                self.comm.Sendrecv(sendbuf=buf_send[1], dest=dest,
                                   recvbuf=buf_recv[1], source=source)

                self._debug("finished halo share in x")
                buf_dev.set(buf_recv)
                evt, _ = self.unpack_x_knl(queue, arr=fx, buf=buf_dev)
                self._debug_barrier("all ranks finished halo share in x")

        if h[1] > 0:
            if self.proc_shape[1] == 1:
                evt, _ = self.pack_unpack_y_knl(queue, arr=fx)
            else:
                self._debug_barrier("all ranks beginning halo share in y")
                shape = (2, rank_shape[0] + 2*h[0], h[1], rank_shape[2] + 2*h[2])
                buf_dev, buf_send, buf_recv = self._get_or_create_halo_buffers(
                    queue, shape, fx.dtype)

                evt, _ = self.pack_y_knl(queue, arr=fx, buf=buf_dev)
                buf_dev.get(ary=buf_send)

                dest = self.rankID(self.rx, self.ry-1, self.rz)
                source = self.rankID(self.rx, self.ry+1, self.rz)
                self._debug(f"initiating first halo share in y: {dest=}, {source=}")
                self.comm.Sendrecv(sendbuf=buf_send[0], dest=dest,
                                   recvbuf=buf_recv[0], source=source)

                dest = self.rankID(self.rx, self.ry+1, self.rz)
                source = self.rankID(self.rx, self.ry-1, self.rz)
                self._debug(f"initiating second halo share in y: {dest=}, {source=}")
                self.comm.Sendrecv(sendbuf=buf_send[1], dest=dest,
                                   recvbuf=buf_recv[1], source=source)

                self._debug("finished halo share in y")
                buf_dev.set(buf_recv)
                evt, _ = self.unpack_y_knl(queue, arr=fx, buf=buf_dev)
                self._debug_barrier("all ranks finished halo share in y")

    def bcast(self, x, root):
        """
        A wrapper to :func:`MPI.COMM_WORLD.bcast` which broadcasts an arbitrary
        python object.

        :arg x: The value to broadcasted. Must be defined on all ranks (i.e.,
            set ``x = None`` on ranks other than ``root``).

        :arg root: The rank whose value of ``x`` should be broadcasted.

        :returns: The broadcasted value, on all ranks.
        """

        self._debug("initiating bcast")
        if self.comm is not None:
            x = self.comm.bcast(x, root=root)
        self._debug_barrier("all ranks finished bcast")
        return x

    def allreduce(self, rank_reduction, op=None):
        """
        A wrapper to :func:`MPI.COMM_WORLD.allreduce` which reduces and broadcasts
        a rank ``rank_reduction`` from the ``root`` rank.

        :arg rank_reduction: The rank's individual value to be reduced.

        :arg op: The MPI reduction operation to perform.
            Defaults to :class:`MPI.SUM`.

        :returns: The reduced value, on all ranks.
        """

        self._debug("initiating allreduce")
        if self.comm is not None:
            from mpi4py import MPI
            op = op or MPI.SUM
            result = self.comm.allreduce(rank_reduction, op=op)
        else:
            result = rank_reduction
        self._debug_barrier("all ranks finished allreduce")
        return result

    def remove_halos(self, queue, in_array, out_array):
        """
        Removes the halo padding from an array.

        The only restriction on the shapes of the three-dimensional input arrays
        is that the shape of ``in_array`` is larger than that of ``out_array``
        by ``2*halo_shape`` along each axis.

        :arg queue: The :class:`pyopencl.CommandQueue` to enqueue kernels and copies.

        :arg in_array: The subarray whose halos will be removed.
            May be either a :class:`pyopencl.array.Array` or a
            :class:`numpy.ndarray`.

        :arg out_array: The output array.
            May be either a :class:`pyopencl.array.Array` or a
            :class:`numpy.ndarray`.
        """

        if not self._kernels_bound:
            self._bind_kernels(queue)

        dtype = out_array.dtype
        if in_array.dtype != dtype:
            raise ValueError("in_array and out_array have different dtypes")

        cl_in = isinstance(in_array, cla.Array)
        cl_out = isinstance(out_array, cla.Array)
        np_in = isinstance(in_array, np.ndarray)
        np_out = isinstance(out_array, np.ndarray)

        slc = tuple(slice(hi, -hi) if hi > 0 else slice(None)
                    for hi in self.halo_shape)

        if cl_in and np_out:
            # FIXME: de-pad in on GPU?
            out_array[:] = in_array.get()[slc]
        elif cl_in or cl_out:
            evt, _ = self.gather_knl(queue, subarr=in_array, arr=out_array)
            evt.wait()  # FIXME: unnecessary?
        elif np_in and np_out:
            out_array[:] = in_array[slc]

    def gather_array(self, queue, in_array, out_array, root):
        """
        Gathers the subdomains of an array from each rank into a single array
        of the entire grid, removing halo padding from ``in_array``.

        :arg queue: The :class:`pyopencl.CommandQueue` to enqueue kernels and copies.

        :arg in_array: The subarrays to be gathered.
            May be either a :class:`pyopencl.array.Array` or a
            :class:`numpy.ndarray`.

        :arg out_array: The output array for the gathered grid.
            May be either a :class:`pyopencl.array.Array` or a
            :class:`numpy.ndarray` on rank ``root``, and *None* otherwise.

        :arg root: The rank to which ``in_array`` is gathered.
        """

        if not self._kernels_bound:
            self._bind_kernels(queue)

        h = self.halo_shape
        dtype = None if self.rank != root else out_array.dtype
        dtype = self.bcast(dtype, root=root)
        if in_array.dtype != dtype:
            raise ValueError("in_array and out_array have different dtypes")

        cl_in = self.bcast(isinstance(in_array, cla.Array), root)
        cl_out = self.bcast(isinstance(out_array, cla.Array), root)

        if self.nranks == 1:
            self.remove_halos(queue, in_array, out_array)
        else:
            temp_in = in_array.get() if cl_in else in_array
            if cl_out and self.rank == root:
                temp_out = np.zeros(shape=out_array.shape, dtype=dtype)
            elif self.rank == root:
                temp_out = out_array
            else:
                temp_out = None

            rank_shape = tuple(i - 2 * hi for i, hi in zip(in_array.shape, h))
            full_shape = None if self.rank != root else out_array.shape
            full_shape = self.bcast(full_shape, root)

            from mpi4py import MPI
            MPI_dtype = MPI._typedict[np.dtype(dtype).char]

            interior = tuple(slice(hi, -hi) if hi > 0 else slice(None) for hi in h)

            nx = self.allreduce(rank_shape[0], op=MPI.MAX)
            for i in range(nx):
                displs, counts = self.get_displs_and_counts(full_shape, i)
                # repeat the last row when some ranks have smaller rank_shape[0]
                x_slice = min(i + h[0], rank_shape[0] + h[0] - 1)
                tmp = temp_in[(x_slice,) + interior[1:]].copy()
                if self.rank == root:
                    recv_buf = [temp_out, counts, displs, MPI_dtype]
                else:
                    recv_buf = None
                self.comm.Gatherv(tmp, recv_buf, root=root)

            if cl_out and self.rank == root:
                out_array.set(temp_out)

    def restore_halos(self, queue, in_array, out_array):
        """
        Adds halo padding to an array.

        The only restriction on the shapes of the three-dimensional input arrays
        is that the shape of ``out_array`` is larger than that of ``in_array``
        by ``2*halo_shape`` along each axis.

        .. note::

            Since :meth:`share_halos` is not currently implemented for
            :class:`numpy.ndarray`\\ s, this method does not automatically
            share halos after they are restored.
            Thus, halos must be shared manually after the fact (for now).

        :arg queue: The :class:`pyopencl.CommandQueue` to enqueue kernels and copies.

        :arg in_array: The array to add halos to.
            May be either a :class:`pyopencl.array.Array` or a
            :class:`numpy.ndarray`.

        :arg out_array: The output array.
            May be either a :class:`pyopencl.array.Array` or a
            :class:`numpy.ndarray`.
        """

        if not self._kernels_bound:
            self._bind_kernels(queue)

        dtype = in_array.dtype
        if out_array.dtype != dtype:
            raise ValueError("in_array and out_array have different dtypes")

        cl_in = isinstance(in_array, cla.Array)
        cl_out = isinstance(out_array, cla.Array)
        np_in = isinstance(in_array, np.ndarray)
        np_out = isinstance(out_array, np.ndarray)

        slc = tuple(slice(hi, -hi) if hi > 0 else slice(None)
                    for hi in self.halo_shape)

        if cl_in and np_out:
            out_array[slc] = in_array.get()
        elif cl_out or cl_in:
            evt, _ = self.scatter_knl(queue, arr=in_array, subarr=out_array)
            evt.wait()  # FIXME: unnecessary?
        elif np_in and np_out:
            out_array[slc] = in_array

        # FIXME: share halos

    def scatter_array(self, queue, in_array, out_array, root):
        """
        Scatters the values of a single array of the entire grid to invidual
        subdomains on each rank.
        The per-rank ``out_array`` is padded by ``halo_shape``.

        .. note::

            Since :meth:`share_halos` is not currently implemented for
            :class:`numpy.ndarray`\\ s, this method does not automatically
            share halos after scattering.
            Thus, halos must be shared manually after the fact (for now).

        :arg queue: The :class:`pyopencl.CommandQueue` to enqueue kernels and copies.

        :arg in_array: The full array to be scattered.
            May be either a :class:`pyopencl.array.Array` or a
            :class:`numpy.ndarray` on rank ``root``, and *None* otherwise.

        :arg out_array: The output array for the scattered arrays.
            May be either a :class:`pyopencl.array.Array` or a
            :class:`numpy.ndarray`.

        :arg root: The rank from which ``in_array`` is scattered.
        """

        if not self._kernels_bound:
            self._bind_kernels(queue)

        h = self.halo_shape
        dtype = None if self.rank != root else in_array.dtype
        dtype = self.bcast(dtype, root=root)
        if out_array.dtype != dtype:
            raise ValueError("in_array and out_array have different dtypes")

        cl_in = self.bcast(isinstance(in_array, cla.Array), root)
        cl_out = self.bcast(isinstance(out_array, cla.Array), root)

        if self.nranks == 1:
            self.restore_halos(queue, in_array, out_array)
        else:
            rank_shape = tuple(i - 2 * hi for i, hi in zip(out_array.shape, h))

            full_shape = None if self.rank != root else in_array.shape
            full_shape = self.bcast(full_shape, root)

            tmp = np.zeros(shape=rank_shape[1:], dtype=dtype)
            if self.rank == root:
                temp_in = in_array.get() if cl_in else in_array
            else:
                temp_in = None
            if cl_out:
                temp_out = np.zeros(shape=out_array.shape, dtype=dtype)
            else:
                temp_out = out_array

            from mpi4py import MPI
            MPI_dtype = MPI._typedict[np.dtype(dtype).char]

            interior = tuple(slice(hi, -hi) if hi > 0 else slice(None) for hi in h)

            nx = self.allreduce(rank_shape[0], op=MPI.MAX)
            for i in range(nx):
                displs, counts = self.get_displs_and_counts(full_shape, i)
                if self.rank == root:
                    send_buf = [temp_in, counts, displs, MPI_dtype]
                else:
                    send_buf = None
                self.comm.Scatterv(send_buf, tmp, root=root)
                if i < rank_shape[0]:
                    temp_out[(i + h[0],) + interior[1:]] = tmp

            if cl_out:
                out_array.set(temp_out)
