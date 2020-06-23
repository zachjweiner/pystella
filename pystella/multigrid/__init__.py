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
from pystella.multigrid.transfer import (Injection, FullWeighting,
                                         LinearInterpolation, CubicInterpolation)
from pystella.multigrid.relax import JacobiIterator, NewtonIterator

__doc__ = """
.. currentmodule:: pystella.multigrid
.. autoclass:: FullApproximationScheme
.. autoclass:: MultiGridSolver

.. _multigrid-cycles:

Multigrid cycles
^^^^^^^^^^^^^^^^

Multigrid cycles are represnted as a sequence of levels to visit and how many
smoothing iterations to perform on each.
Level ``i`` denotes the level with a factor ``2**i`` fewer gridpoints
in each dimension (relative to the finest grid).
The following utilities can be used to generate particular types ofcycles
by specifying, e.g., the coarsest level to visit and how many iterations
to perform on these levels.

.. autofunction:: mu_cycle
.. autofunction:: v_cycle
.. autofunction:: w_cycle
.. autofunction:: f_cycle
"""


def mu_cycle(mu, i, nu1, nu2, max_depth):
    """
    A utility for generating a generic :math:`\\mu`-cycle.

    :arg mu: The order of the cycle. See...

    :arg i: The initial and final (i.e., finest) level to traverse from/to.

    :arg nu1: The number of iterations to perform on each level after a
        transition to the next coarser level.

    :arg nu2:  The number of iterations to perform on each level after a
        transition to the next finer level.

    :arg max_depth: The lowest level to traverse to.

    :returns: A generic multigrid cycle in the form of a :class:`list` of
        :class:`tuple`\\ s ``(level, iterations)``, representing the order of levels
        to visit and how many smoothing iterations to perform on each.
    """

    if i == max_depth:
        return [(i, nu2)]
    else:
        x = mu_cycle(mu, i+1, nu1, nu2, max_depth)
        return [(i, nu1)] + x + x[1:]*(mu-1) + [(i, nu2)]


def v_cycle(nu1, nu2, max_depth):
    """
    A utility for generating a V-cycle.

    Example::

        >>> v_cycle(10, 20, 3)
        [(0, 10), (1, 10), (2, 10), (3, 20), (2, 20), (1, 20), (0, 20)]

    :arg nu1: The number of iterations to perform on each level after a
        transition to the next coarser level.

    :arg nu2:  The number of iterations to perform on each level after a
        transition to the next finer level.

    :arg max_depth: The lowest level to traverse to.

    :returns: A V-cycle in the form of a :class:`list` of
        :class:`tuple`\\ s ``(level, iterations)``, representing the order of levels
        to visit and how many smoothing iterations to perform on each.
    """

    return mu_cycle(1, 0, nu1, nu2, max_depth)


def w_cycle(nu1, nu2, max_depth):
    """
    A utility for generating a W-cycle.

    Example::

        >>> w_cycle(10, 20, 3)
        [(0, 10), (1, 10), (2, 10), (3, 20), (2, 20), (3, 20), (2, 20), (1, 20),
        (2, 10), (3, 20), (2, 20), (3, 20), (2, 20), (1, 20), (0, 20)]

    :arg nu1: The number of iterations to perform on each level after a
        transition to the next coarser level.

    :arg nu2:  The number of iterations to perform on each level after a
        transition to the next finer level.

    :arg max_depth: The lowest level to traverse to.

    :returns: A W-cycle in the form of a :class:`list` of
        :class:`tuple`\\ s ``(level, iterations)``, representing the order of levels
        to visit and how many smoothing iterations to perform on each.
    """

    return mu_cycle(2, 0, nu1, nu2, max_depth)


def _cycle(i, j, k, nu1, nu2):
    down = [(a, nu1) for a in range(i, j)]
    up = [(a, nu2) for a in range(j, k-1, -1)]
    return down + up


def f_cycle(nu1, nu2, max_depth):
    """
    A utility for generating a F-cycle.

    Example::

        >>> f_cycle(10, 20, 3)
        [(0, 10), (1, 10), (2, 10), (3, 20), (2, 20), (3, 20), (2, 20), (1, 20),
        (2, 10), (3, 20), (2, 20), (1, 20), (0, 20)]

    :arg nu1: The number of iterations to perform on each level after a
        transition to the next coarser level.

    :arg nu2:  The number of iterations to perform on each level after a
        transition to the next finer level.

    :arg max_depth: The lowest level to traverse to.

    :returns: An F-cycle in the form of a :class:`list` of
        :class:`tuple`\\ s ``(level, iterations)``, representing the order of levels
        to visit and how many smoothing iterations to perform on each.
    """

    cycle = _cycle(0, max_depth, max_depth-1, nu1, nu2)
    for top in range(max_depth-1, 0, -1):
        cycle += _cycle(top+1, max_depth, top-1, nu1, nu2)
    return cycle


class FullApproximationScheme:
    """
    A class for solving generic systems of boundary-value problems using the
    Full Approximation Scheme.

    .. automethod:: __init__
    .. automethod:: __call__

    The below methods are documented for development's sake, but are not
    intended to be called by the user.

    .. automethod:: coarse_array_like
    .. automethod:: transfer_down
    .. automethod:: transfer_up
    .. automethod:: smooth
    .. automethod:: coarse_level_like
    .. automethod:: setup
    """

    def __init__(self, solver, halo_shape, **kwargs):
        """
        :arg solver: A instance of a subclass of :class:`relax.RelaxationBase`
            (e.g., :class:`JacobiIterator` or :class:`NewtonIterator`).

    :arg halo_shape: The number of halo layers on (both sides of) each axis of
        the computational grid.
        Currently must be an :class:`int`.

        The following keyword-only arguments are recognized:

        :arg Restrictor: A mapper which restricts arrays from a fine
            to a coarser level.
            Defaults to :class:`FullWeighting`.

        :arg Interpolator: A mapper which interpolates arrays from a coarse
            to a finer level.
            Defaults to :class:`LinearInterpolation`.
        """

        self.solver = solver
        self.halo_shape = halo_shape

        Restrictor = kwargs.pop('Restrictor', FullWeighting)
        self.restrict = Restrictor(halo_shape=halo_shape)
        self.restrict_and_correct = Restrictor(halo_shape=halo_shape, correct=True)

        Interpolator = kwargs.pop('Interpolator', LinearInterpolation)
        self.interpolate = Interpolator(halo_shape=halo_shape)
        self.interpolate_and_correct = Interpolator(halo_shape=halo_shape,
                                                    correct=True)

        self.unknowns = {}
        self.rhos = {}
        self.auxiliaries = {}
        self.tmp = {}
        self.resid = {}
        self.dx = {}
        self.decomp = {}
        self.smooth_args = {}
        self.resid_args = {}

    def coarse_array_like(self, f1h):
        """
        :arg f1h: A :class:`pyopencl.array.Array`.
            Its unpadded shape will be inferred by subtracting
            ``2 * self.halo_shape`` from each axis of its shape.

        :returns: A :class:`pyopencl.array.Array` with padded shape for a
            grid with half as many points in each dimension of ``f1h``.
        """

        def halve_and_pad(i):
            return (i - 2 * self.halo_shape)//2 + 2 * self.halo_shape

        coarse_shape = tuple(map(halve_and_pad, f1h.shape))
        f2h = cla.zeros(f1h.queue, shape=coarse_shape, dtype=f1h.dtype)
        return f2h

    def transfer_down(self, queue, i):
        """
        Transfers all arrays from a fine to the next-coarser level.

        :arg queue: A :class:`pyopencl.CommandQueue`.

        :arg i: The level from to transfer to.
        """

        for k, f1 in self.unknowns[i-1].items():
            f2 = self.unknowns[i][k]
            self.restrict(queue, f1=f1, f2=f2)
            self.decomp[i].share_halos(queue, f2)

        self.solver.residual(queue, **self.resid_args[i-1])

        for k, r1 in self.resid[i-1].items():
            r2 = self.resid[i][k]
            self.decomp[i-1].share_halos(queue, r1)
            self.restrict(queue, f1=r1, f2=r2)

        self.solver.lhs_correction(queue, **self.resid_args[i])
        for k, rho in self.rhos[i].items():
            self.decomp[i].share_halos(queue, rho)

    def transfer_up(self, queue, i):
        """
        Transfers all arrays from a coarse to the next-finer level.

        :arg queue: A :class:`pyopencl.CommandQueue`.

        :arg i: The level from to transfer to.
        """

        for k, f1 in self.unknowns[i].items():
            f2 = self.unknowns[i+1][k]
            self.restrict_and_correct(queue, f1=f1, f2=f2)
            self.decomp[i+1].share_halos(queue, f2)
            self.interpolate_and_correct(queue, f1=f1, f2=f2)
            self.decomp[i].share_halos(queue, f1)

    def smooth(self, queue, i, nu):
        """
        Invokes the relaxation solver, computing the error before and after.

        :arg queue: A :class:`pyopencl.CommandQueue`.

        :arg i: On which level to perform the smoothing.

        :arg nu: The number of smoothing iterations to perform.

        :returns: A list containing the errors before and after of the form
            ``[(i, error_before), (i, error_after)]``.
        """

        errs1 = self.solver.get_error(queue, **self.resid_args[i])
        self.solver(self.decomp[i], queue, iterations=nu, **self.smooth_args[i])
        errs2 = self.solver.get_error(queue, **self.resid_args[i])
        return [(i, errs1), (i, errs2)]

    def coarse_level_like(self, dict_1):
        """
        A wrapper to :meth:`coarse_array_like` with returns a :class:`dict`
        like ``dict_1`` whose values are new :class:`pyopencl.array.Array`\\ s
        with shape appropriate for the next-coarser level.
        """

        dict_2 = {}
        for k, f1 in dict_1.items():
            dict_2[k] = self.coarse_array_like(f1)
        return dict_2

    def setup(self, decomp0, queue, dx0, depth, **kwargs):
        """
        Performs the inital setup and array allocation for each required level.
        Creates instances of :class:`~pystella.DomainDecomposition` for each level
        and all arrays needed on each level.
        Called automatically by :meth:`__call__`.

        :arg decomp0: An instance of :class:`~pystella.DomainDecomposition`
            constructed for the finest level.

        :arg queue: A :class:`pyopencl.CommandQueue`.

        :arg dx0: The grid-spacing on the finest level.

        :arg depth: The coarsest level to traverse to. That is, the deepest level
            which will be used has a factor ``2**depth`` fewer gridpoints than the
            finest level.

        All unknowns and ``rho`` arrays must be passed by keyword.
        Any additional keyword arguments are interpreted as auxillary arrays which
        must be available on all levels.
        """

        self.decomp[0] = decomp0
        self.dx[0] = np.array(dx0)

        self.unknowns[0] = {}
        self.rhos[0] = {}
        for k, v in self.solver.f_to_rho_dict.items():
            self.unknowns[0][k] = kwargs.pop(k)
            self.rhos[0][v] = kwargs.pop(v)

        self.auxiliaries[0] = kwargs

        if 0 not in self.tmp:
            self.tmp[0] = {}
            self.resid[0] = {}
            for k, f in self.unknowns[0].items():
                self.tmp[0]['tmp_'+k] = cla.zeros_like(f)
                self.resid[0]['r_'+k] = self.tmp[0]['tmp_'+k]

        for i in range(depth+1):
            if i not in self.dx:
                self.dx[i] = np.array(self.dx[i-1] * 2)

            if i not in self.decomp:
                ng_2 = tuple(ni // 2 for ni in self.decomp[i-1].rank_shape)
                from pystella import DomainDecomposition
                self.decomp[i] = \
                    DomainDecomposition(self.decomp[i-1].proc_shape,
                                        self.halo_shape, ng_2)

            if i not in self.unknowns:
                self.unknowns[i] = self.coarse_level_like(self.unknowns[i-1])

            if i not in self.tmp:
                self.tmp[i] = self.coarse_level_like(self.tmp[i-1])
                self.resid[i] = {}
                for k, f in self.unknowns[i].items():
                    self.resid[i]['r_'+k] = self.tmp[i]['tmp_'+k]

            if i not in self.rhos:
                self.rhos[i] = self.coarse_level_like(self.rhos[i-1])

            if i not in self.auxiliaries:
                self.auxiliaries[i] = self.coarse_level_like(self.auxiliaries[i-1])
                for k, f1 in self.auxiliaries[i-1].items():
                    f2 = self.auxiliaries[i][k]
                    self.restrict(queue, f1=f1, f2=f2)
                    self.decomp[i].share_halos(queue, f2)

            if i not in self.smooth_args:
                self.smooth_args[i] = {**self.unknowns[i], **self.rhos[i],
                                       **self.auxiliaries[i], **self.tmp[i]}
                self.smooth_args[i]['dx'] = np.array(self.dx[i])

            if i not in self.resid_args:
                self.resid_args[i] = {**self.unknowns[i], **self.rhos[i],
                                      **self.auxiliaries[i], **self.resid[i]}
                self.resid_args[i]['dx'] = np.array(self.dx[i])

    def __call__(self, decomp0, queue, dx0, cycle=None, **kwargs):
        """
        Executes a specified multigrid cycle.

        :arg decomp0: An instance of :class:`~pystella.DomainDecomposition`
            constructed for the finest level.

        :arg queue: A :class:`pyopencl.CommandQueue`.

        :arg dx0: The grid-spacing on the finest level.

        :arg cycle: The multigrid cycle to execute.
            See :ref:`multigrid-cycles` for details on how these are specified
            and for utilities to generate them.

        All required arrays must be passed by keyword.
        """

        if cycle is None:
            grid_shape = tuple(ni * pi
                               for ni, pi in zip(decomp0.rank_shape,
                                                 decomp0.proc_shape))
            depth = int(np.log2(min(grid_shape) / 8))
            cycle = v_cycle(25, 50, depth)

        depth = max([i for i, nu in cycle])
        self.setup(decomp0, queue, dx0, depth, **kwargs)

        nu0 = cycle[0][1]
        level_errors = self.smooth(queue, 0, nu0)

        previous = 0
        for i, nu in cycle[1:]:
            if i == previous + 1:
                self.transfer_down(queue, i)
            elif i == previous - 1:
                self.transfer_up(queue, i)
            else:
                raise ValueError('consecutive levels must be spaced by one')
            level_errors += self.smooth(queue, i, nu)
            previous = i

        return level_errors


class MultiGridSolver(FullApproximationScheme):
    """
    A class for solving systems of linear boundary-value problems using linear
    Multigrid.
    Usage is identical to :class:`FullApproximationScheme`.

    .. warning::

        Convergence is currently slower than expected, suggesting a possible
        problem with the lower levels.
        :class:`FullApproximationScheme` is perfectly suited to solve linear problems
        as well.

    The scheme is implemented by subclassing :class:`FullApproximationScheme`, with
    the only differences in the level transfer functionality (which are not intended
    to be called by the user).

    .. automethod transfer_down
    .. automethod transfer_up
    """

    # FIXME: convergence slow, possible issue with coarse levels?
    def transfer_down(self, queue, i):
        self.solver.residual(queue, **self.resid_args[i-1])

        for f, rho in self.solver.f_to_rho_dict.items():
            r1 = self.resid[i-1]['r_'+f]
            self.decomp[i-1].share_halos(queue, r1)
            r2 = self.rhos[i][rho]
            self.restrict(queue, f1=r1, f2=r2)
            self.decomp[i].share_halos(queue, r2)

    def transfer_up(self, queue, i):
        for k, f1 in self.unknowns[i].items():
            f2 = self.unknowns[i+1][k]
            self.interpolate_and_correct(queue, f1=f1, f2=f2)
            self.decomp[i].share_halos(queue, f1)


__all__ = [
    'Injection',
    'FullWeighting',
    'LinearInterpolation',
    'CubicInterpolation',
    'JacobiIterator',
    'NewtonIterator',
    'FullApproximationScheme',
    'MultiGridSolver',
    'v_cycle',
    'w_cycle',
    'f_cycle',
]
