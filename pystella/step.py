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
from pymbolic.primitives import Subscript, Variable

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

        * a :class:`list` of :class:`~pystella.Sector`\\ s. In this case,
          the input obtained from each :class:`~pystella.Sector`
          (as described above) will be combined.

    The following keyword arguments are recognized:

    :arg MapKernel: The kernel class which each substep/stage will be an
        instance of---i.e., one of :class:`~pystella.ElementWiseMap` or its
        subclasses. Defaults to :class:`~pystella.ElementWiseMap`.

    :arg dt: A :class:`float` fixing the value of the timestep interval.
        Defaults to *None*, in which case it is not fixed at kernel creation.

    The remaining arguments are passed to :meth:`MapKernel` for
    each substep of the timestepper (i.e., see the documentation of
    :class:`~pystella.ElementWiseMap`).

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
        single_stage = kwargs.pop("single_stage", True)
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

        args = kwargs.pop("args", [...])
        args = args + [lp.ValueArg("dt")]
        from pystella import get_field_args
        inferred_args = get_field_args(self.rhs_dict, prepend_with=prepend_with)
        from pystella.elementwise import append_new_args
        self.args = append_new_args(args, inferred_args)

        dt = kwargs.pop("dt", None)
        fixed_parameters = kwargs.pop("fixed_parameters", dict())
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
        :class:`~pystella.Field`\\ s (or :class:`pymbolic.primitives.Subscript`\\ s
        thereof). This is required for :meth:`make_steps` to be able to prepend
        unknown arrays' subscripts with the index corresponding to the temporary
        storage axis.
    """

    def __init__(self, input, **kwargs):
        super().__init__(input, single_stage=False, **kwargs)

    def step_statements(self, stage, f, dt, rhs):
        raise NotImplementedError

    def make_steps(self, MapKernel=ElementWiseMap, **kwargs):
        rhs = var("rhs")
        dt = var("dt")
        q = var("q")
        fixed_parameters = kwargs.pop("fixed_parameters", dict())

        rhs_statements = {rhs[i]: index_fields(value, prepend_with=(q,))
                          for i, value in enumerate(self.rhs_dict.values())}

        steps = []
        for stage in range(self.num_stages):
            RK_dict = {}
            for i, f in enumerate(self.rhs_dict.keys()):
                # ensure that key is either a Field or a Subscript of a Field
                # so that index_fields can prepend the q index
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


def get_name(expr):
    if isinstance(expr, Field):
        return get_name(expr.child)
    elif isinstance(expr, Subscript):
        return get_name(expr.aggregate)
    elif isinstance(expr, Variable):
        return expr.name
    elif isinstance(expr, str):
        return expr


def gen_tmp_name(expr, prefix="_", suffix="_tmp"):
    name = get_name(expr)
    return prefix + name + suffix


def copy_and_rename(expr):
    if isinstance(expr, Field):
        return expr.copy(child=copy_and_rename(expr.child))
    elif isinstance(expr, Subscript):
        return Subscript(copy_and_rename(expr.aggregate), expr.index)
    elif isinstance(expr, Variable):
        return Variable(gen_tmp_name(expr))
    elif isinstance(expr, str):
        return gen_tmp_name(expr)


class LowStorageRKStepper(Stepper):
    """
    The base implementation of low-storage, explicit Runge-Kutta time steppers,
    which operate by storing and operating on a single copy of each unknown array,
    plus an auxillary temporary array.

    The substeps are expressed in a standard form, drawing coefficients from
    a subclass's provided values of :attr:`_A`, :attr:`_B`, and :attr:`_C`.

    Allocation of the auxillary arrays is handled internally by:

    .. automethod:: get_tmp_arrays_like

    :meth:`get_tmp_arrays_like` is called the first time
    :meth:`__call__` is  called, with the result stored in the attribute
    :attr:`tmp_arrays`.
    These arrays must not be modified between substages of a single timestep,
    but may be safely modified in between timesteps.

    .. versionchanged:: 2020.2

        Auxillary arrays handled internally by :meth:`get_tmp_arrays_like`.
        Previously, manual allocation (and passing) of a single temporary
        array ``k_tmp`` was required.
    """

    _A = []
    _B = []
    _C = []

    tmp_arrays = {}

    def make_steps(self, MapKernel=ElementWiseMap, **kwargs):
        tmp_arrays = [copy_and_rename(key) for key in self.rhs_dict.keys()]
        self.dof_names = {get_name(key) for key in self.rhs_dict.keys()}
        rhs_statements = {var(gen_tmp_name(key, suffix=f"_rhs_{i}")): val
                          for i, (key, val) in enumerate(self.rhs_dict.items())}

        steps = []
        for stage in range(self.num_stages):
            RK_dict = {}
            for i, (f, k) in enumerate(zip(self.rhs_dict.keys(), tmp_arrays)):
                rhs = var(gen_tmp_name(f, suffix=f"_rhs_{i}"))
                RK_dict[k] = self._A[stage] * k + var("dt") * rhs
                RK_dict[f] = f + self._B[stage] * k

            step = MapKernel(RK_dict, tmp_instructions=rhs_statements,
                             args=self.args, **kwargs)
            steps.append(step)

        return steps

    def get_tmp_arrays_like(self, **kwargs):
        """
        Allocates required temporary arrays matching those passed via keyword.

        :returns: A :class:`dict` of named arrays, suitable for passing via
            dictionary expansion.

        .. versionadded:: 2020.2
        """

        tmp_arrays = {}
        for name in self.dof_names:
            f = kwargs[name]
            tmp_name = gen_tmp_name(name)
            import pyopencl.array as cla
            if isinstance(f, cla.Array):
                tmp_arrays[tmp_name] = cla.empty_like(f)
            elif isinstance(f, np.ndarray):
                tmp_arrays[tmp_name] = np.empty_like(f)
            else:
                raise ValueError(f"Could not generate tmp array for {f}"
                                 f"of type {type(f)}")
            tmp_arrays[tmp_name][...] = 0.

        return tmp_arrays

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for step in self.steps:
            step.knl = lp.add_inames_for_unused_hw_axes(step.knl)

    def __call__(self, stage, *, queue=None, **kwargs):
        if len(self.tmp_arrays) == 0:
            self.tmp_arrays = self.get_tmp_arrays_like(**kwargs)

        return super().__call__(stage, queue=queue, **kwargs, **self.tmp_arrays)


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


class LowStorageRK144(LowStorageRKStepper):
    """
    A 14-stage, fourth-order low-storage Runge-Kutta method optimized for elliptic
    stability regions.

    See
    Niegemann, Jens & Diehl, Richard & Busch, Kurt. (2012). Efficient low-storage
    Runge-Kutta schemes with optimized stability regions. J. Comput. Physics. 231.
    364-372. 10.1016/j.jcp.2011.09.003.
    """

    num_stages = 14
    expected_order = 4

    _A = [
        0,
        -0.7188012108672410,
        -0.7785331173421570,
        -0.0053282796654044,
        -0.8552979934029281,
        -3.9564138245774565,
        -1.5780575380587385,
        -2.0837094552574054,
        -0.7483334182761610,
        -0.7032861106563359,
        0.0013917096117681,
        -0.0932075369637460,
        -0.9514200470875948,
        -7.1151571693922548
    ]

    _B = [
        0.0367762454319673,
        0.3136296607553959,
        0.1531848691869027,
        0.0030097086818182,
        0.3326293790646110,
        0.2440251405350864,
        0.3718879239592277,
        0.6204126221582444,
        0.1524043173028741,
        0.0760894927419266,
        0.0077604214040978,
        0.0024647284755382,
        0.0780348340049386,
        5.5059777270269628
    ]

    _C = [
        0,
        0.0367762454319673,
        0.1249685262725025,
        0.2446177702277698,
        0.2476149531070420,
        0.2969311120382472,
        0.3978149645802642,
        0.5270854589440328,
        0.6981269994175695,
        0.8190890835352128,
        0.8527059887098624,
        0.8604711817462826,
        0.8627060376969976,
        0.8734213127600976
    ]


class LowStorageRK134(LowStorageRKStepper):
    """
    A 13-stage, fourth-order low-storage Runge-Kutta method optimized for circular
    stability regions.

    See
    Niegemann, Jens & Diehl, Richard & Busch, Kurt. (2012). Efficient low-storage
    Runge-Kutta schemes with optimized stability regions. J. Comput. Physics. 231.
    364-372. 10.1016/j.jcp.2011.09.003.
    """

    num_stages = 13
    expected_order = 4

    _A = [
        0,
        0.6160178650170565,
        0.4449487060774118,
        1.0952033345276178,
        1.2256030785959187,
        0.2740182222332805,
        0.0411952089052647,
        0.179708489915356,
        1.1771530652064288,
        0.4078831463120878,
        0.8295636426191777,
        4.789597058425229,
        0.6606671432964504
    ]

    _B = [
        0.0271990297818803,
        0.1772488819905108,
        0.0378528418949694,
        0.6086431830142991,
        0.21543139743161,
        0.2066152563885843,
        0.0415864076069797,
        0.0219891884310925,
        0.9893081222650993,
        0.0063199019859826,
        0.3749640721105318,
        1.6080235151003195,
        0.0961209123818189
    ]

    _C = [
        0,
        0.0271990297818803,
        0.0952594339119365,
        0.1266450286591127,
        0.1825883045699772,
        0.3737511439063931,
        0.5301279418422206,
        0.5704177433952291,
        0.5885784947099155,
        0.6160769826246714,
        0.6223252334314046,
        0.6897593128753419,
        0.9126827615920843
    ]


class LowStorageRK124(LowStorageRKStepper):
    """
    A 12-stage, fourth-order low-storage Runge-Kutta method optimized for inviscid
    problems.

    See
    Niegemann, Jens & Diehl, Richard & Busch, Kurt. (2012). Efficient low-storage
    Runge-Kutta schemes with optimized stability regions. J. Comput. Physics. 231.
    364-372. 10.1016/j.jcp.2011.09.003.
    """

    num_stages = 12
    expected_order = 4

    _A = [
        0,
        0.0923311242368072,
        0.9441056581158819,
        4.327127324757639,
        2.155777132902607,
        0.9770727190189062,
        0.7581835342571139,
        1.79775254708255,
        2.691566797270077,
        4.646679896026814,
        0.1539613783825189,
        0.5943293901830616
    ]

    _B = [
        0.0650008435125904,
        0.0161459902249842,
        0.5758627178358159,
        0.1649758848361671,
        0.3934619494248182,
        0.0443509641602719,
        0.2074504268408778,
        0.6914247433015102,
        0.3766646883450449,
        0.0757190350155483,
        0.2027862031054088,
        0.2167029365631842
    ]

    _C = [
        0,
        0.0650008435125904,
        0.0796560563081853,
        0.1620416710085376,
        0.2248877362907778,
        0.2952293985641261,
        0.3318332506149405,
        0.4094724050198658,
        0.6356954475753369,
        0.6806551557645497,
        0.714377371241835,
        0.9032588871651854,
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
B2 = ((12 * c2 * (c2 - 1) * (3 * z2 - z1) - (3 * z2 - z1)**2)
      / (144 * c2 * (3 * c2 - 2) * (c2 - 1)**2))
B3 = (- 24 * (3 * c2 - 2) * (c2 - 1)**2
      / ((3 * z2 - z1)**2 - 12 * c2 * (c2 - 1) * (3 * z2 - z1)))
A2 = ((- z1 * (6 * c2**2 - 4 * c2 + 1) + 3 * z3)
      / ((2 * c2 + 1) * z1 - 3 * (c2 + 2) * (2 * c2 - 1)**2))
A3 = ((- z4 * z1 + 108 * (2 * c2 - 1) * c2**5 - 3 * (2 * c2 - 1) * z5)
      / (24 * z1 * c2 * (c2 - 1)**4 + 72 * c2 * z6 + 72 * c2**6 * (2 * c2 - 13)))


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
                RungeKutta2Ralston, LowStorageRK54, LowStorageRK144,
                LowStorageRK3Williamson, LowStorageRK3Inhomogeneous,
                LowStorageRK3SSP]
