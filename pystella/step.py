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
from pystella.field import Field, index_fields
from pystella.elementwise import ElementWiseMap
from pymbolic import var

__doc__ = """
.. currentmodule:: pystella.step
.. autoclass:: Stepper
.. currentmodule:: pystella

Low-storage Runge-Kutta methods
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. currentmodule:: pystella.step
.. autoclass:: LowStorageRKStepper
.. currentmodule:: pystella
.. autoclass:: LowStorageRK54
.. autoclass:: LowStorageRK3Williamson
.. autoclass:: LowStorageRK3Inhomogeneous
.. autoclass:: LowStorageRK3SSP

Classical Runge-Kutta methods
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

"Classical" Runge-Kutta methods are also implemented, though are not recommended
over the low-storage methods above.

.. currentmodule:: pystella.step
.. autoclass:: RungeKuttaStepper
.. currentmodule:: pystella
.. autoclass:: RungeKutta4
.. autoclass:: RungeKutta3SSP
.. autoclass:: RungeKutta3Heun
.. autoclass:: RungeKutta3Nystrom
.. autoclass:: RungeKutta3Ralston
.. autoclass:: RungeKutta2Midpoint
.. autoclass:: RungeKutta2Ralston

"""


class Stepper:
    """
    The base class for time steppers, with no implementation of a particular time
    stepper.

    .. automethod:: __init__
    .. automethod:: __call__

    .. attribute:: num_stages

        The number of substeps/stages per timestep.

    .. attribute:: expected_order

        The expected convergence order of *global* error, i.e.
        :math:`n` such that the global error is :math:`\\mathcal{O}(\\Delta t^n)`.

    .. attribute:: num_unknowns

            The number of unknown degrees of freedom which are evolved.
    """

    num_stages = None
    expected_order = None
    num_copies = None

    def make_steps(self, MapKernel=ElementWiseMap, **kwargs):
        raise NotImplementedError

    def __init__(self, input, MapKernel=ElementWiseMap, **kwargs):
        """
        :arg input: May be one of the following:

            * a :class:`dict` whose values represent the right-hand side
              of the ODEs to solve, i.e., `(key, value)` pairs corresponding to
              :math:`(y, f)` such that

                .. math::

                    \\frac{\\mathrm{d} y}{\\mathrm{d} t} = f,

              where :math:`f` is an arbitrary function of kernel data.
              Both keys and values must be :mod:`pymbolic` expressions.

            * a :class:`~pystella.Sector`. In this case, the right-hand side
              dictionary will be obtained from :attr:`~pystella.Sector.rhs_dict`.

            * a :class:`list` of :class:`~pystella.Sector`'s. In this case, the input
              obtained from each :class:`~pystella.Sector` (as described above) will
              be combined.

        The following keyword arguments are recognized:

        :arg MapKernel: The kernel class which each substep/stage will be an
            instance of---i.e., one of :class:`~pystella.ElementWiseMap` or its
            subclasses. Defaults to :class:`~pystella.ElementWiseMap`.

        :arg dt: A :class:`float` fixing the value of the timestep interval.
            Defaults to *None*, in which case it is not fixed at kernel creation.

        The remaining arguments are passed to :meth:`MapKernel.__init__` for
        each substep of the timestepper (i.e., see the documentation of
        :class:`~pystella.ElementWiseMap`).
        """

        single_stage = kwargs.pop('single_stage', True)
        from pystella import Sector
        if isinstance(input, Sector):
            self.rhs_dict = input.rhs_dict
        elif isinstance(input, list):
            self.rhs_dict = dict(i for s in input for i in s.rhs_dict.items())
        elif isinstance(input, dict):
            self.rhs_dict = input

        if not single_stage:
            prepend_with = (self.num_copies,)
        else:
            prepend_with = None

        args = kwargs.pop('args', [...])
        args = args + [lp.ValueArg('dt')]
        from pystella import get_field_args
        inferred_args = get_field_args(self.rhs_dict, prepend_with=prepend_with)
        from pystella.elementwise import append_new_args
        self.args = append_new_args(args, inferred_args)

        dt = kwargs.pop('dt', None)
        fixed_parameters = kwargs.pop('fixed_parameters', dict())
        if dt is not None:
            fixed_parameters.update(dict(dt=dt))

        self.num_unknowns = len(self.rhs_dict.keys())
        self.steps = self.make_steps(**kwargs, fixed_parameters=fixed_parameters)

    def __call__(self, stage, queue=None, **kwargs):
        """
        Calls substep/stage ``stage`` (:attr:`steps[stage]`) of the timestepper,
        i.e., :func:`pystella.ElementWiseMap.__call__` for the kernel for
        substep/stage ``stage``.

        :arg stage: The substep/stage of time timestepper to call.

        :returns: The :class:`pyopencl.Event` associated with the kernel invocation.
        """

        evt, _ = self.steps[stage](queue, **kwargs)
        return evt


class RungeKuttaStepper(Stepper):
    """
    The base implementation of classical, explicit Runge-Kutta time steppers,
    which operate by storing and operating on multiple copies of each unknown
    array. Subclasses must provide an implementation of :meth:`step_statements`
    which returns a key-value pair implementing a specific substep of the
    particular timestepper.

    .. warning::

        To minimize the required storage per unknown (i.e., number of
        temporaries), the implementation of most subclasses overwrite arrays that
        are being read as input to compute right-hand sides. This means that any
        non-local (stencil-type) operations must be precomputed and cached
        *globally* (unless otherwise noted).

    :raises ValueError: if the keys of :attr:`rhs_dict` are not
        :class:`~pystella.Field`'s (or :class:`pymbolic.primitives.Subscript`'s
        thereof). This is required for :meth:`make_steps` to be able to prepend
        unknown arrays' subscripts with the index corresponding to the temporary
        storage axis.
    """

    def __init__(self, input, **kwargs):
        super().__init__(input, single_stage=False, **kwargs)

    def step_statements(self, stage, f, dt, rhs):
        raise NotImplementedError

    def make_steps(self, MapKernel=ElementWiseMap, **kwargs):
        rhs = var('rhs')
        dt = var('dt')
        q = var('q')
        fixed_parameters = kwargs.pop('fixed_parameters', dict())

        rhs_statements = {rhs[i]: index_fields(value, prepend_with=(q,))
                          for i, value in enumerate(self.rhs_dict.values())}

        steps = []
        for stage in range(self.num_stages):
            RK_dict = {}
            for i, f in enumerate(self.rhs_dict.keys()):
                # ensure that key is either a Field or a Subscript of a Field
                # so that index_fields can prepend the q index
                from pymbolic.primitives import Subscript
                key_has_field = False
                if isinstance(f, Field):
                    key_has_field = True
                elif isinstance(f, Subscript):
                    if isinstance(f.aggregate, Field):
                        key_has_field = True

                if not key_has_field:
                    raise ValueError("rhs_dict keys must be Field instances")

                statements = self.step_statements(stage, f, dt, rhs[i])
                for k, v in statements.items():
                    RK_dict[k] = v

            fixed_parameters.update(q=0 if stage == 0 else 1)

            options = lp.Options(enforce_variable_access_ordered="no_check")
            step = MapKernel(RK_dict, tmp_instructions=rhs_statements,
                             args=self.args, **kwargs, options=options,
                             fixed_parameters=fixed_parameters)
            steps.append(step)

        return steps


class RungeKutta4(RungeKuttaStepper):
    """
    The classical, four-stage, fourth-order Runge-Kutta method.
    Requires unknown arrays to have temporary storage axes of length three.
    """

    num_stages = 4
    expected_order = 4
    num_copies = 3

    def step_statements(self, stage, f, dt, rhs):
        fq = [index_fields(f, prepend_with=(q,)) for q in range(3)]

        if stage == 0:
            return {fq[1]: fq[0] + dt/2 * rhs,
                    fq[2]: fq[0] + dt/6 * rhs}
        elif stage == 1:
            return {fq[1]: fq[0] + dt/2 * rhs,
                    fq[2]: fq[2] + dt/3 * rhs}
        elif stage == 2:
            return {fq[1]: fq[0] + dt * rhs,
                    fq[2]: fq[2] + dt/3 * rhs}
        elif stage == 3:
            return {fq[0]: fq[2] + dt/6 * rhs}


class RungeKutta3Heun(RungeKuttaStepper):
    """
    Heun's three-stage, third-order Runge-Kutta method.
    Requires unknown arrays to have temporary storage axes of length three.
    """

    num_stages = 3
    expected_order = 3
    num_copies = 3

    def step_statements(self, stage, f, dt, rhs):
        fq = [index_fields(f, prepend_with=(q,)) for q in range(3)]

        if stage == 0:
            return {fq[1]: fq[0] + dt/3 * rhs,
                    fq[2]: fq[0] + dt/4 * rhs}
        elif stage == 1:
            return {fq[1]: fq[0] + dt*2/3 * rhs}
        elif stage == 2:
            return {fq[0]: fq[2] + dt*3/4 * rhs}


class RungeKutta3Nystrom(RungeKuttaStepper):
    """
    Nystrom's three-stage, third-order Runge-Kutta method.
    Requires unknown arrays to have temporary storage axes of length three.
    """

    num_stages = 3
    expected_order = 3
    num_copies = 3

    def step_statements(self, stage, f, dt, rhs):
        fq = [index_fields(f, prepend_with=(q,)) for q in range(3)]

        if stage == 0:
            return {fq[1]: fq[0] + dt*2/3 * rhs,
                    fq[2]: fq[0] + dt*2/8 * rhs}
        elif stage == 1:
            return {fq[1]: fq[0] + dt*2/3 * rhs,
                    fq[2]: fq[2] + dt*3/8 * rhs}
        elif stage == 2:
            return {fq[0]: fq[2] + dt*3/8 * rhs}


class RungeKutta3Ralston(RungeKuttaStepper):
    """
    Ralston's three-stage, third-order Runge-Kutta method.
    Requires unknown arrays to have temporary storage axes of length three.
    """

    num_stages = 3
    expected_order = 3
    num_copies = 3

    def step_statements(self, stage, f, dt, rhs):
        fq = [index_fields(f, prepend_with=(q,)) for q in range(3)]

        if stage == 0:
            return {fq[1]: fq[0] + dt/2 * rhs,
                    fq[2]: fq[0] + dt*2/9 * rhs}
        elif stage == 1:
            return {fq[1]: fq[0] + dt*3/4 * rhs,
                    fq[2]: fq[2] + dt*1/3 * rhs}
        elif stage == 2:
            return {fq[0]: fq[2] + dt*4/9 * rhs}


class RungeKutta3SSP(RungeKuttaStepper):
    """
    A three-stage, third-order strong-stability preserving Runge-Kutta method.
    Requires unknown arrays to have temporary storage axes of length two.
    """

    num_stages = 3
    expected_order = 3
    num_copies = 2

    def step_statements(self, stage, f, dt, rhs):
        fq = [index_fields(f, prepend_with=(q,)) for q in range(3)]

        if stage == 0:
            return {fq[1]: fq[0] + dt * rhs}
        elif stage == 1:
            return {fq[1]: 3/4 * fq[0] + 1/4 * fq[1] + dt/4 * rhs}
        elif stage == 2:
            return {fq[0]: 1/3 * fq[0] + 2/3 * fq[1] + dt*2/3 * rhs}


class RungeKutta2Midpoint(RungeKuttaStepper):
    """
    The "midpoint" method, a two-stage, second-order Runge-Kutta method.
    Requires unknown arrays to have temporary storage axes of length two.
    Note that right-hand side operations *can* safely involve non-local computations
    of unknown arrays for this method.
    """

    num_stages = 2
    expected_order = 2
    num_copies = 2

    def step_statements(self, stage, f, dt, rhs):
        fq = [index_fields(f, prepend_with=(q,)) for q in range(2)]

        if stage == 0:
            return {fq[1]: fq[0] + dt/2 * rhs}
        elif stage == 1:
            return {fq[0]: fq[0] + dt * rhs}


# possible order reduction
class RungeKutta2Heun(RungeKuttaStepper):
    num_stages = 2
    expected_order = 2
    num_copies = 2

    def step_statements(self, stage, f, dt, rhs):
        fq = [index_fields(f, prepend_with=(q,)) for q in range(2)]

        if stage == 0:
            return {fq[1]: fq[0] + dt * rhs,
                    fq[0]: fq[0] + dt/2 * rhs}
        elif stage == 1:
            return {fq[0]: fq[0] + dt/2 * rhs}


class RungeKutta2Ralston(RungeKuttaStepper):
    """
    Ralstons's two-stage, second-order Runge-Kutta method.
    Requires unknown arrays to have temporary storage axes of length two.
    """

    num_stages = 2
    expected_order = 2
    num_copies = 2

    def step_statements(self, stage, f, dt, rhs):
        fq = [index_fields(f, prepend_with=(q,)) for q in range(2)]

        if stage == 0:
            return {fq[1]: fq[0] + dt*2/3 * rhs,
                    fq[0]: fq[0] + dt/4 * rhs}
        elif stage == 1:
            return {fq[0]: fq[0] + dt*3/4 * rhs}


class LowStorageRKStepper(Stepper):
    """
    The base implementation of low-storage, explicit Runge-Kutta time steppers,
    which operate by storing and operating on a single copy of each unknown array,
    plus an auxillary temporary array.

    The substeps are expressed in a standard form, drawing coefficients from
    a subclass's provided values of :attr:`_A`, :attr:`_B`, and :attr:`_C`.

    .. automethod:: __call__
    """

    _A = []
    _B = []
    _C = []

    def make_steps(self, MapKernel=ElementWiseMap, **kwargs):
        rhs = var('rhs')
        dt = var('dt')

        # filter out indices for zero axes
        test_array = list(self.rhs_dict.keys())[0]
        from pymbolic.primitives import Subscript
        if isinstance(test_array, Subscript):
            if isinstance(test_array.aggregate, Field):
                test_array = test_array.aggregate
        k = Field('k_tmp', indices=test_array.indices, shape=(self.num_unknowns,))

        rhs_statements = {rhs[i]: value
                          for i, value in enumerate(self.rhs_dict.values())}

        steps = []
        for stage in range(self.num_stages):
            RK_dict = {}
            for i, key in enumerate(self.rhs_dict.keys()):
                f = key
                k_i = k[i]
                RK_dict[k_i] = self._A[stage] * k_i + dt * rhs[i]
                RK_dict[f] = f + self._B[stage] * k_i

            step = MapKernel(RK_dict, tmp_instructions=rhs_statements,
                             args=self.args, **kwargs)
            steps.append(step)

        return steps

    def __call__(self, stage, *, k_tmp, queue=None, **kwargs):
        """
        Same as :meth:`Stepper.__call__`, but requires the
        following arguments:

        :arg k_tmp: The array used for temporary
            calculations. Its outer-/left-most axis (i.e., the axis of largest
            stride) must have length equal to the total number of unknowns,
            which may be obtained from :attr:`num_unknowns`.
            Passed by keyword only.

        For example::

            >>> stepper = LowStorageRK54(rhs_dict)
            >>> import pyopencl.array as cla
            >>> temp_shape = (stepper.num_unknowns,) + rank_shape
            >>> k_tmp = cla.zeros(queue, temp_shape, 'float64')
            >>> for stage in range(stepper.num_stages):
            ...    stepper(stage, queue, k_tmp=k_tmp, ...)
        """

        return super().__call__(stage, queue=queue, k_tmp=k_tmp, **kwargs)


class LowStorageRK54(LowStorageRKStepper):
    """
    A five-stage, fourth-order, low-storage Runge-Kutta method.

    See
    Carpenter, M.H., and Kennedy, C.A., Fourth-order-2N-storage
    Runge-Kutta schemes, NASA Langley Tech Report TM 109112, 1994
    """

    num_stages = 5
    expected_order = 4

    _A = [
        0,
        -567301805773 / 1357537059087,
        -2404267990393 / 2016746695238,
        -3550918686646 / 2091501179385,
        -1275806237668 / 842570457699,
        ]

    _B = [
        1432997174477 / 9575080441755,
        5161836677717 / 13612068292357,
        1720146321549 / 2090206949498,
        3134564353537 / 4481467310338,
        2277821191437 / 14882151754819,
        ]

    _C = [
        0,
        1432997174477 / 9575080441755,
        2526269341429 / 6820363962896,
        2006345519317 / 3224310063776,
        2802321613138 / 2924317926251,
        ]


class LowStorageRK3Williamson(LowStorageRKStepper):
    """
    A three-stage, third-order, low-storage Runge-Kutta method.

    See
    Williamson, J. H., Low-storage Runge-Kutta schemes,
    J. Comput. Phys., 35, 48-56, 1980
    """

    num_stages = 3
    expected_order = 3

    _A = [0, -5/9, -153/128]

    _B = [1/3, 15/16, 8/15]

    _C = [0, 4/9, 15/32]


class LowStorageRK3Inhomogeneous(LowStorageRKStepper):
    """
    A three-stage, third-order, low-storage Runge-Kutta method.
    """

    num_stages = 3
    expected_order = 3

    _A = [0, -17/32, -32/27]

    _B = [1/4, 8/9, 3/4]

    _C = [0, 15/32, 4/9]


# possible order reduction
class LowStorageRK3Symmetric(LowStorageRKStepper):
    num_stages = 3
    expected_order = 3

    _A = [0, -2/3, -1]

    _B = [1/3, 1, 1/2]

    _C = [0, 1/3, 2/3]


# possible order reduction
class LowStorageRK3PredictorCorrector(LowStorageRKStepper):
    num_stages = 3
    expected_order = 3

    _A = [0, -1/4, -4/3]

    _B = [1/2, 2/3, 1/2]

    _C = [0, 1/2, 1]


c2 = .924574
z1 = np.sqrt(36 * c2**4 + 36 * c2**3 - 135 * c2**2 + 84 * c2 - 12)
z2 = 2 * c2**2 + c2 - 2
z3 = 12 * c2**4 - 18 * c2**3 + 18 * c2**2 - 11 * c2 + 2
z4 = 36 * c2**4 - 36 * c2**3 + 13 * c2**2 - 8 * c2 + 4
z5 = 69 * c2**3 - 62 * c2**2 + 28 * c2 - 8
z6 = 34 * c2**4 - 46 * c2**3 + 34 * c2**2 - 13 * c2 + 2
B1 = c2
B2 = (12 * c2 * (c2 - 1) * (3 * z2 - z1) - (3 * z2 - z1)**2) \
     / (144 * c2 * (3 * c2 - 2) * (c2 - 1)**2)
B3 = - 24 * (3 * c2 - 2) * (c2 - 1)**2 \
     / ((3 * z2 - z1)**2 - 12 * c2 * (c2 - 1) * (3 * z2 - z1))
A2 = (- z1 * (6 * c2**2 - 4 * c2 + 1) + 3 * z3) \
     / ((2 * c2 + 1) * z1 - 3 * (c2 + 2) * (2 * c2 - 1)**2)
A3 = (- z4 * z1 + 108 * (2 * c2 - 1) * c2**5 - 3 * (2 * c2 - 1) * z5) \
     / (24 * z1 * c2 * (c2 - 1)**4 + 72 * c2 * z6 + 72 * c2**6 * (2 * c2 - 13))


class LowStorageRK3SSP(LowStorageRKStepper):
    """
    A three-stage, third-order, strong-stability preserving, low-storage
    Runge-Kutta method.
    """

    num_stages = 3
    expected_order = 3

    _A = [0, A2, A3]

    _B = [B1, B2, B3]

    _C = [0, B1, B1 + B2 * (A2 + 1)]


all_steppers = [RungeKutta4, RungeKutta3SSP, RungeKutta3Heun, RungeKutta3Nystrom,
                RungeKutta3Ralston, RungeKutta2Midpoint,
                RungeKutta2Ralston, LowStorageRK54,
                LowStorageRK3Williamson, LowStorageRK3Inhomogeneous,
                LowStorageRK3SSP]
