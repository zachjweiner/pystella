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


import loopy as lp
import pymbolic.primitives as pp

__doc__ = """
.. currentmodule:: pystella
.. autoclass:: ElementWiseMap
"""


def append_new_args(old_args, new_args):
    all_args = old_args.copy()
    supplied_arg_names = set([arg.name for arg in old_args if hasattr(arg, 'name')])
    for arg in new_args:
        if arg.name not in supplied_arg_names:
            all_args.append(arg)
    return all_args


class ElementWiseMap:
    """
    An interface to :func:`loopy.make_kernel`, which creates a kernel
    with parallelization suitable for operations which are "local"---namely,
    element-wise maps where each workitem (thread) only accesses one element
    of global arrays.

    .. automethod:: __init__
    .. automethod:: __call__
    .. attribute:: knl

        The generated :class:`loopy.LoopKernel`.
    """

    def parallelize(self, knl, lsize):
        knl = lp.split_iname(knl, "k", lsize[0], outer_tag="g.0", inner_tag="l.0")
        knl = lp.split_iname(knl, "j", lsize[1], outer_tag="g.1", inner_tag="l.1")
        knl = lp.split_iname(knl, "i", lsize[2], outer_tag="g.2", inner_tag="unr")
        return knl

    def _assignment(self, assignee, expression, **kwargs):
        no_sync_with = kwargs.pop('no_sync_with', [('*', 'any')])
        return lp.Assignment(assignee, expression,
                             no_sync_with=no_sync_with,
                             **kwargs)

    def make_kernel(self, map_instructions, tmp_instructions, args, **kwargs):
        temp_statements = []
        temp_vars = []

        from pystella.field import index_fields
        indexed_tmp_insns = index_fields(tmp_instructions)
        indexed_map_insns = index_fields(map_instructions)

        for assignee, expression in indexed_tmp_insns:
            # only declare temporary variables once
            if isinstance(assignee, pp.Variable):
                current_tmp = assignee
            elif isinstance(assignee, pp.Subscript):
                current_tmp = assignee.aggregate
            else:
                current_tmp = None
            if current_tmp is not None and current_tmp not in temp_vars:
                temp_vars += [current_tmp]
                tvt = lp.Optional(None)
            else:
                tvt = lp.Optional()

            stmnt = self._assignment(assignee, expression, temp_var_type=tvt)
            temp_statements += [stmnt]

        output_statements = []
        for assignee, expression in indexed_map_insns:
            stmnt = self._assignment(assignee, expression)
            output_statements += [stmnt]

        options = kwargs.pop('options', lp.Options())
        # ignore lack of supposed dependency for single-instruction kernels
        if len(map_instructions) + len(tmp_instructions) == 1:
            setattr(options, 'check_dep_resolution', False)

        from pystella import get_field_args
        inferred_args = get_field_args([map_instructions, tmp_instructions])
        all_args = append_new_args(args, inferred_args)

        knl = lp.make_kernel(
            "[Nx, Ny, Nz] -> {[i,j,k]: 0<=i<Nx and 0<=j<Ny and 0<=k<Nz}",
            temp_statements + output_statements,
            all_args + [lp.ValueArg('Nx, Ny, Nz', dtype='int'), ...],
            options=options,
            **kwargs,
        )

        new_args = []
        for arg in knl.args:
            if isinstance(arg, lp.KernelArgument) and arg.dtype is None:
                new_arg = arg.copy(dtype=self.dtype)
                new_args.append(new_arg)
            else:
                new_args.append(arg)
        knl = knl.copy(args=new_args)
        knl = lp.remove_unused_arguments(knl)

        return knl

    def __init__(self, map_instructions, **kwargs):
        """
        :arg map_instructions: A :class:`list` of instructions which write to global
            arrays.
            Entries may be :class:`loopy.Assignment`'s or tuples
            ``(assignee, expression)`` of :mod:`pymbolic` expressions, the latter
            of which can include :class:`Field`'s.
            All entries will be processed with :func:`index_fields`.

        The following keyword-only arguments are recognized:

        :arg tmp_instructions: A :class:`list` of instructions
            which write to temporary variables (i.e., local or private memory).
            Entries may be :class:`loopy.Assignment`'s or tuples
            ``(assignee, expression)`` of :mod:`pymbolic` expressions, the latter
            of which can include :class:`Field`'s.
            The expressions will be processed with :func:`index_fields`.
            The statements produced from ``tmp_instructions`` will precede those of
            ``map_instructions``, and :class:`loopy.TemporaryVariable` arguments
            will be inferred as needed.

        :arg args: A list of :class:`loopy.KernelArgument`'s
            to be specified to :func:`loopy.make_kernel`.
            By default, all arguments (and their shapes) are inferred using
            :func:`get_field_args`, while any remaining (i.e., non-:class:`Field`)
            arguments are inferred by :func:`loopy.make_kernel`.
            Any arguments passed via ``args`` override those inferred by either
            of the above options.

        :arg dtype: The default datatype of arrays to assume.
            Will only be applied to all :class:`loopy.KernelArgument`'s
            whose datatypes were not already specified by any input ``args``.
            Defaults to *None*.

        :arg lsize: The size of local parallel workgroups. Defaults to
            ``(16, 4, 1)``, which should come close to saturating memory bandwidth
            in many cases.

        :arg rank_shape: A 3-:class:`tuple` specifying the shape of looped-over
            arrays.
            Defaults to *None*, in which case these values are not fixed (and
            will be inferred when the kernel is called at a slight performance
            penalty).

        :arg halo_shape: The number of halo layers on (both sides of) each axis of
            the computational grid.
            May either be an :class:`int`, interpreted as a value to fix the
            parameter ``h`` to, or a :class:`tuple`, interpreted as values for
            ``hx``, ``hy``, and ``hz``.
            Defaults to *None*, in which case no such values are fixed at kernel
            creation.

        Any remaining keyword arguments are passed to :func:`loopy.make_kernel`.
        """

        if 'map_dict' in kwargs:
            from warnings import warn
            warn("Passing map_dict is deprecated. Pass map_instructions instead.",
                 DeprecationWarning, stacklevel=2)
            map_instructions = kwargs.pop('map_dict')

        self.map_instructions = map_instructions
        if isinstance(self.map_instructions, dict):
            self.map_instructions = list(self.map_instructions.items())

        if 'tmp_dict' in kwargs:
            from warnings import warn
            warn("Passing tmp_dict is deprecated. Pass tmp_instructions instead.",
                 DeprecationWarning, stacklevel=2)
            tmp_instructions = kwargs.pop('tmp_dict')
        else:
            tmp_instructions = []

        self.tmp_instructions = kwargs.pop('tmp_instructions', tmp_instructions)
        if isinstance(self.tmp_instructions, dict):
            self.tmp_instructions = list(self.tmp_instructions.items())

        self.args = kwargs.pop('args', [...])
        self.dtype = kwargs.pop('dtype', None)

        # default local size which saturates memory bandwidth
        self.lsize = kwargs.pop('lsize', (16, 4, 1))
        rank_shape = kwargs.pop('rank_shape', None)
        halo_shape = kwargs.pop('halo_shape', None)

        kernel_kwargs = dict(
            seq_dependencies=True,
            default_offset=lp.auto,
            target=lp.PyOpenCLTarget(),
            lang_version=(2018, 2),
        )
        kernel_kwargs.update(kwargs)

        knl = self.make_kernel(self.map_instructions, self.tmp_instructions,
                               self.args, **kernel_kwargs)

        if rank_shape is not None:
            knl = lp.fix_parameters(
                knl, Nx=rank_shape[0], Ny=rank_shape[1], Nz=rank_shape[2]
            )
        if isinstance(halo_shape, int):
            knl = lp.fix_parameters(knl, h=halo_shape)
        elif isinstance(halo_shape, (tuple, list)):
            knl = lp.fix_parameters(
                knl, hx=halo_shape[0], hy=halo_shape[1], hz=halo_shape[2]
            )

        self.knl = self.parallelize(knl, self.lsize)

    def __call__(self, queue=None, filter_args=False, **kwargs):
        """
        Invokes the kernel, :attr:`knl`. All data arguments required by :attr:`knl`
        must be passed by keyword.

        The following keyword arguments are recognized:

        :arg queue: The :class:`pyopencl.CommandQueue` on which to enqueue the
            kernel.
            If *None* (the default), ``queue`` is not passed (i.e., for
            :class:`loopy.ExecutableCTarget`).

            .. note::

                For :class:`loopy.PyOpenCLTarget` (the default), a valid
                :class:`pyopencl.CommandQueue` is a required argument.

        :arg filter_args: Whether to filter ``kwargs`` such that only arguments to
            the ``knl`` are passed. Defaults to *False*.

        In addition, any arguments recognized by
        :meth:`loopy.PyOpenCLKernelExecutor.__call__` are also accepted.

        :returns: ``(evt, output)`` where ``evt`` is the :class:`pyopencl.Event`
            associated with the kernel invocation and ``output`` is any kernel
            output.
            See :mod:`loopy`'s tutorial for details.
        """

        input_args = kwargs.copy()
        if filter_args:
            kernel_args = [arg.name for arg in self.knl.args]
            for arg in kwargs:
                if arg not in kernel_args:
                    input_args.pop(arg)

            # add back PyOpenCLExecuter arguments
            if isinstance(self.knl.target, lp.PyOpenCLTarget):
                for arg in ['allocator', 'wait_for', 'out_host']:
                    if arg in kwargs:
                        input_args[arg] = kwargs.get(arg)

        if isinstance(self.knl.target, lp.PyOpenCLTarget):
            input_args['queue'] = queue

        knl_output = self.knl(**input_args)

        return knl_output

    def __str__(self):
        string = ''
        for key, value in self.tmp_instructions:
            string += str(key) + ' = ' + str(value) + '\n'
        for key, value in self.map_instructions:
            string += str(key) + ' = ' + str(value) + '\n'
        return string
