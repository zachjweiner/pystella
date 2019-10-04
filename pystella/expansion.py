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


class Expansion:
    """
    Implements the time stepping of the scale factor evolution for conformal
    FLRW spacetimes with line element

    .. math::

        \\mathrm{d} s^2
        = a(\\tau)^2 \\left( - \\mathrm{d} \\tau^2
                             + \\delta_{ij} \\mathrm{d} x^i \\mathrm{d} x^j
                    \\right).

    Below, the averaged energy density and pressure are

    .. math::

        \\bar{\\rho}
        &\\equiv - \\left\\langle T_{\\hphantom{0}0}^0 \\right\\rangle

        \\bar{P}
        &\\equiv \\frac{1}{3} \\left\\langle T_{\\hphantom{i}i}^i \\right\\rangle.

    .. automethod:: __init__
    .. automethod:: adot_friedmann_1
    .. automethod:: addot_friedmann_2
    .. automethod:: step
    .. automethod:: constraint
    """

    def __init__(self, energy, Stepper, mpl=1., dtype=np.float64):
        """
        :arg energy: The initial energy density, used to initialize
            :math:`\\partial a / \\partial \\tau`.

        :arg Stepper: A :class:`~pystella.step.Stepper` to use for time stepping.

        :arg mpl: The unreduced Planck mass,
            :math:`m_\\mathrm{pl}^2 \\equiv 1 / G_N`.
            Setting this value chooses the units of the system.
            For example, to work in units of the *reduced* Planck mass,
            :math:`M_\\mathrm{pl}^2 \\equiv (8 \\pi G_N)^{-1}`, pass
            ``mpl=np.sqrt(8*np.pi)``.
            Defaults to ``1``.

        :arg dtype: The datatype of the input and output arrays.
            Defaults to `float64`.
        """

        self.mpl = mpl
        from pystella.step import LowStorageRKStepper

        self.is_low_storage = LowStorageRKStepper in Stepper.__bases__
        shape = (1,) if self.is_low_storage else (3,)
        self.a = np.ones(shape, dtype=dtype)
        self.adot = self.adot_friedmann_1(self.a, energy)
        self.hubble = self.adot / self.a

        from pystella import Field
        _a = Field('a', indices=[])[(0,) if self.is_low_storage else ()]
        _adot = Field('adot', indices=[])[(0,) if self.is_low_storage else ()]
        from pymbolic import var
        _e = var('energy')
        _p = var('pressure')
        rhs_dict = {_a: _adot,
                    _adot: self.addot_friedmann_2(_a, _e, _p)}

        args = [lp.GlobalArg('a', shape=shape, dtype=dtype),
                lp.GlobalArg('adot', shape=shape, dtype=dtype),
                lp.ValueArg('energy', dtype=dtype),
                lp.ValueArg('pressure', dtype=dtype),
                ]

        from pystella import DisableLogging
        with DisableLogging():  # silence GCCToolchain warning
            self.stepper = Stepper(rhs_dict, args=args, rank_shape=(0, 0, 0),
                                   halo_shape=0, dtype=dtype,
                                   target=lp.ExecutableCTarget())

        if self.is_low_storage:
            self.k_tmp = np.zeros(shape=(2,), dtype=dtype)

    def adot_friedmann_1(self, a, energy):
        """
        :arg a: The current scale factor, :math:`a`.

        :arg energy: The current energy density, :math:`\\bar{\\rho}`.

        :returns: The value of :math:`\\partial_\\tau a`
            as given by Friedmann's first equation,

        .. math::

            \\mathcal{H}^2
            \\equiv \\left( \\frac{\\partial_\\tau a}{a} \\right)^2
            = \\frac{8 \\pi a^2}{3 m_\\mathrm{pl}^2} \\bar{\\rho}
        """

        return np.sqrt(8 * np.pi * a**2 / 3 / self.mpl**2 * energy) * a

    def addot_friedmann_2(self, a, energy, pressure):
        """
        :arg a: The current scale factor, :math:`a`.

        :arg energy: The current energy density, :math:`\\bar{\\rho}`.

        :arg pressure: The current pressure, :math:`\\bar{P}`.

        :returns: The value of :math:`\\partial_\\tau^2 a`
            as given by Friedmann's second equation,

        .. math::

            \\partial_\\tau \\mathcal{H} + \\mathcal{H}^2
            = \\frac{\\partial_\\tau^2 a}{a}
            = \\frac{4 \\pi a^2}{3 m_\\mathrm{pl}^2}
            \\left( \\bar{\\rho} - 3 \\bar{P} \\right)
        """

        return 4 * np.pi * a**2 / 3 / self.mpl**2 * (energy - 3 * pressure) * a

    def step(self, stage, energy, pressure, dt):
        """
        Executes one stage of the time stepper.

        :arg stage: Which stage of the integrator to call.

        :arg energy: The current energy density, :math:`\\bar{\\rho}`.

        :arg pressure: The current pressure, :math:`\\bar{P}`.

        :arg dt: The timestep to take.
        """

        arg_dict = dict(a=self.a, adot=self.adot, dt=dt,
                        energy=energy, pressure=pressure)
        if self.is_low_storage:
            arg_dict['k_tmp'] = self.k_tmp

        self.stepper(stage, **arg_dict)
        self.hubble[()] = self.adot / self.a

    def constraint(self, energy):
        """
        A dimensionless measure of the satisfaction of the first Friedmann equation
        (as a constraint on the evolution), equal to

        .. math::

            \\left\\vert \\frac{1}{\\mathcal{H}}
            \\sqrt{\\frac{8 \\pi a^2}{3 m_\\mathrm{pl}^2} \\rho} - 1
            \\right\\vert

        where :math:`\\mathcal{H}` the current conformal Hubble parameter,
        :math:`\\partial_\\tau a / a`.

        :arg energy: The current energy density, :math:`\\bar{\\rho}`.
        """

        return np.abs(self.adot_friedmann_1(self.a[0], energy) / self.adot[0] - 1)
