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
import loopy as lp
from pystella import Field, get_field_args, Stencil

__doc__ = """
.. currentmodule:: pystella.multigrid
.. autoclass:: pystella.multigrid.relax.RelaxationBase
.. autoclass:: JacobiIterator
.. autoclass:: NewtonIterator
"""


class RelaxationBase:
    """
    Base class for relaxation-based iterative solvers to solve
    boundary-value problems of the form

    .. math::

        L(f) = \\rho.

    Here :math:`\\rho` is not a function of :math:`f`, but :math:`L(f)`
    may in principle be an arbitrary (nonlinear differential) function
    of :math:`f` (assuming a subclass's implemented solver is appropriate
    for such an equation).

    .. automethod:: __init__
    .. automethod:: __call__
    .. automethod:: get_error

    A subclass implements a particular iterative solver by providing
    a :meth:`step_operator` method.

    .. automethod:: step_operator

    The below methods are documented for development's sake, but are not
    intended to be called by the user:

    .. automethod:: make_stepper
    .. automethod:: make_lhs_kernel
    .. automethod:: make_residual_kernel
    .. automethod:: make_resid_stats

    The following methods related to solving additional constraints on
    systems with periodic boundary conditions are incomplete:

    .. automethod:: make_shift_kernel
    .. automethod:: eval_constraint
    .. automethod:: solve_constraint
    """

    def __init__(self, decomp, queue, lhs_dict, MapKernel=Stencil, **kwargs):
        """
        :arg decomp: A :class:`~pystella.DomainDecomposition`.

        :arg queue: A :class:`pyopencl.CommandQueue`.

        :arg lhs_dict: A :class:`dict` representing the set of equations to be
            solved, whose keys must be :class:`~pystella.Field`\\ s representing the
            unknown degrees of freedom and values are :class:`tuple`\\ s
            ``(lhs, rho)`` representing the left-hand side :math:`L(f)`
            and right-hand side :math:`\\rho` of that unknown's equation.

        The following keyword arguments are recognized:

        :arg MapKernel: The kernel class which the required mapping kernels will
            be instances of---i.e., one of :class:`~pystella.ElementWiseMap` or its
            subclasses. Defaults to :class:`~pystella.Stencil`.

        :arg unknown_args: A list of :class:`loopy.ArrayArg`\\ s representing
            the unknown degrees of freedom.
            Defaults to *None*, in which case the correct arguments
            (in particular, their shapes) are (attempted to be) inferred
            from the keys of ``lhs_dict``.

        :arg rho_args: A list of :class:`loopy.ArrayArg`\\ s representing
            the static right-hand side arrays (i.e., those independent
            of the degrees of freedom).
            Defaults to *None*, in which case the correct arguments
            (in particular, their shapes) are (attempted to be) inferred
            from the values of ``lhs_dict``.

        Any remaining keyword arguments are passed to each of the kernel
        creation routines.
        """

        self.decomp = decomp
        self.lhs_dict = lhs_dict
        self.halo_shape = kwargs.get('halo_shape')

        # get GlobalArgs of unknowns, or infer from lhs_dict.keys()
        self.unknown_args = kwargs.pop('unknown_args', None)
        if self.unknown_args is None:
            self.unknown_args = get_field_args(list(lhs_dict.keys()))

        def array_args_like(args, prefix='', suffix=''):
            return [lp.GlobalArg(prefix+arg.name+suffix,
                                 shape=arg.shape, dtype=arg.dtype)
                    for arg in args]

        self.temp_args = array_args_like(self.unknown_args, prefix='tmp_')
        self.residual_args = array_args_like(self.unknown_args, prefix='r_')

        # get GlobalArgs of unknowns, or infer from lhs_dict.keys()
        self.rho_args = kwargs.pop('rho_args', None)
        if self.rho_args is None:
            rho_list = [lhs[1] for lhs in lhs_dict.values()]
            self.rho_args = get_field_args(rho_list)

        self.f_to_rho_dict = {}
        for f, (lhs, rho) in self.lhs_dict.items():
            self.f_to_rho_dict[f.child.name] = rho.child.name

        self.make_stepper(MapKernel, **kwargs)
        self.make_lhs_kernel(MapKernel, **kwargs)
        self.make_residual_kernel(MapKernel, **kwargs)
        self.make_resid_stats(decomp, queue, **kwargs)
        self.make_shift_kernel(**kwargs)

    def step_operator(self, f, lhs, rho):
        """
        :arg f: The unknown field for which a relaxation step instruction
            will be generated.

        :arg lhs: :math:`L(f)` for the unknown ``f``'s equation.

        :arg rho: :math:`\\rho` for the unknown ``f``'s equation.
        """

        raise NotImplementedError

    def make_stepper(self, MapKernel, **kwargs):
        self.step_dict = {}
        for f, (lhs, rho) in self.lhs_dict.items():
            tmp = Field('tmp_'+f.child.name, offset=f.offset)
            self.step_dict[tmp] = self.step_operator(f, lhs, rho)

        args = self.unknown_args + self.rho_args + self.temp_args
        self.stepper = MapKernel(self.step_dict, args=args, **kwargs)

    def step(self, queue, **kwargs):
        self.stepper(queue, **kwargs)

    def __call__(self, decomp, queue, iterations=100, **kwargs):
        """
        Executes a number of iterations of relaxation.

        :arg decomp: A :class:`~pystella.DomainDecomposition`.

            .. note::

                ``decomp`` is intended to (and should) be different from the
                :attr:`decomp` passed to :meth:`__init__`, as each multigrid level
                requires a different :class:`~pystella.DomainDecomposition`.

        :arg queue: A :class:`pyopencl.CommandQueue`.

        The following keyword arguments are recognized:

        :arg iterations: The number of iterations to execute.
            Defaults to ``100``.

        :arg solve_constraint:
            Defaults to *False*.

        All arrays required for the relaxation step must be passed by keyword.
        """

        solve_constraint = kwargs.pop('solve_constraint', False)

        even_iterations = iterations if iterations % 2 == 0 else iterations + 1
        for i in range(even_iterations):
            self.stepper(queue, **kwargs)
            for arg in self.unknown_args:
                f = arg.name
                kwargs[f], kwargs['tmp_'+f] = kwargs['tmp_'+f], kwargs[f]
                decomp.share_halos(queue, kwargs[f])

            if solve_constraint:
                self.solve_constraint(queue, **kwargs)

    def make_lhs_kernel(self, MapKernel, **kwargs):
        tmp_dict = {}
        lhs_dict = {}
        from pymbolic import var
        tmp_lhs = var('tmp_lhs')
        for i, (f, (lhs, rho)) in enumerate(self.lhs_dict.items()):
            tmp_dict[tmp_lhs[i]] = lhs
            resid = Field('r_'+f.child.name, offset='h')
            lhs_dict[rho] = resid + tmp_lhs[i]

        args = self.unknown_args + self.rho_args + self.residual_args
        self.lhs_correction = MapKernel(lhs_dict, tmp_instructions=tmp_dict,
                                        args=args, **kwargs)

    def make_residual_kernel(self, MapKernel, **kwargs):
        residual_dict = {}
        for f, (lhs, rho) in self.lhs_dict.items():
            resid = Field('r_'+f.child.name, offset='h')
            residual_dict[resid] = rho - lhs

        args = self.unknown_args + self.rho_args + self.residual_args
        self.residual = MapKernel(residual_dict, args=args, **kwargs)

    def make_resid_stats(self, decomp, queue, dtype, **kwargs):
        reducers = {}
        avg_reducers = {}
        # from pymbolic.functions import fabs
        from pymbolic import var
        fabs = var('fabs')
        for arg in self.unknown_args:
            f = arg.name
            resid = Field('r_'+f, offset='h')
            reducers[f] = [(fabs(resid), 'max'), (resid**2, 'avg')]
            avg_reducers[f] = [(resid, 'avg')]

        args = self.residual_args
        from pystella import Reduction
        self.resid_stats = Reduction(decomp, reducers, args=args, **kwargs)
        self.avg_resid = Reduction(decomp, avg_reducers, args=args, **kwargs)

    def get_error(self, queue, **kwargs):
        """
        Computes statistics of the current residual, :math:`L(f) - \\rho`.

        :arg queue: A :class:`pyopencl.CommandQueue`.

        All required arrays must be passed by keyword.

        :returns: A :class:`dict` whose values are :class:`list`\\ s of the
            :math:`L_\\infty` (maximum absolute) and :math:`L_2` (Euclidean)
            norms of the residual equation corresponding to the unknown denoted
            by the keys of the dictionary.
        """

        self.residual(queue, **kwargs, filter_args=True)

        padded_shape = kwargs.get(self.unknown_args[0].name).shape
        rank_shape = tuple(i - 2 * self.halo_shape for i in padded_shape)
        grid_size = np.product(self.decomp.proc_shape) * np.product(rank_shape)
        errs = self.resid_stats(queue, **kwargs, filter_args=True,
                                rank_shape=rank_shape, grid_size=grid_size)
        for k, v in errs.items():
            errs[k][1] = v[1]**.5

        return errs

    def make_shift_kernel(self, **kwargs):
        f = Field('f', offset=0)
        tmp = Field('tmp', offset=0)
        from pymbolic import var
        shift = var('shift')
        scale = var('scale')
        self.shift_dict = {tmp: scale * f + shift}

        args = [...]
        from pystella import ElementWiseMap
        self.shifter = ElementWiseMap(self.shift_dict, args=args, **kwargs)

    def eval_constraint(self, queue, shifts, scales, **kwargs):
        for arg, shift, scale in zip(self.unknown_args, shifts, scales):
            f = arg.name
            self.shifter(queue, f=kwargs[f], tmp=kwargs['tmp_'+f],
                         shift=np.array(shift), scale=np.array(scale))

        padded_shape = kwargs.get(self.unknown_args[0].name).shape
        rank_shape = tuple(i - 2 * self.halo_shape for i in padded_shape)
        grid_size = np.product(self.decomp.proc_shape) * np.product(rank_shape)

        args_to_avg_resid = kwargs.copy()
        for arg in self.unknown_args:
            f = arg.name
            args_to_avg_resid[f] = kwargs['tmp_'+f]

        result = self.avg_resid(queue, **args_to_avg_resid, filter_args=True,
                                rank_shape=rank_shape, grid_size=grid_size)
        return result['avg']

    def solve_constraint(self, queue, **kwargs):
        raise NotImplementedError('constraint solving untested')

        def integral_condition(shifts):
            scales = np.ones_like(shifts)
            avg = self.eval_constraint(queue, **kwargs, shifts=shifts, scales=scales)
            return np.sum(avg)

        from scipy.optimize import root_scalar
        x0 = np.zeros(len(self.unknown_args))
        x1 = x0 + 1.e-3
        x0 += - 1.e-3
        sol = root_scalar(integral_condition, x0=x0, x1=x1, method='secant')
        if not sol.converged:
            print(sol)
        else:
            shifts = sol.root
            scales = np.ones_like(shifts)
            for arg, shift, scale in zip(self.unknown_args, shifts, scales):
                f = arg.name
                self.shifter(queue, f=kwargs[f], tmp=kwargs[f],
                             shift=np.array(shift), scale=np.array(scale))


class JacobiIterator(RelaxationBase):
    """
    A subclass of :class:`RelaxationBase` which implements (damped) Jacobi iteration
    for linear systems of the form :math:`L f = \\rho`, where :math:`L` is a linear
    operator.
    A step of Jacobi iteration takes the form

    .. math::

        f \\leftarrow (1 - \\omega) f
        + \\omega D^{-1} \\left( \\rho - (L - D) f \\right)

    where :math:`D` is the diagonal part of :math:`L`.
    In practice :math:`D` is computed by differentiating :math:`L f` with respect to
    :math:`f`, which is inappropriate for nonlinear system (which Jacobi
    iteration is not intended for).
    """

    def step_operator(self, f, lhs, rho):
        from pystella import diff
        D = diff(lhs, f)
        R_y = lhs - D * f  # FIXME: only valid for linear equations

        from pymbolic import var
        omega = var('omega')

        return (1 - omega) * f + omega * (rho - R_y) / D


class NewtonIterator(RelaxationBase):
    """
    A subclass of :class:`RelaxationBase` which implements Newton iteration
    for arbitrary systems of the form :math:`L(f) = \\rho`, where :math:`L`
    is a generic function of :math:`f`.
    A step of Newton iteration takes the form

    .. math::

        f \\leftarrow f
        - \\omega \\frac{L(f) - \\rho}{\\partial L(f) / \\partial f}

    """

    def step_operator(self, f, lhs, rho):
        from pystella import diff
        D = diff(lhs, f)

        from pymbolic import var
        omega = var('omega')

        return f - omega * (lhs - rho) / D
