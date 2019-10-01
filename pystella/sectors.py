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
from pystella import DynamicField, Field
from pystella.field import diff
from pymbolic import var

__doc__ = """
.. currentmodule:: pystella
.. autoclass:: Sector
.. autoclass:: ScalarSector
.. autoclass:: TensorPerturbationSector

.. currentmodule:: pystella.sectors
.. autofunction:: get_rho_and_p
"""

eta = [-1, 1, 1, 1]


class Sector:
    """
    A unimplemented base class defining the methods and properties needed for
    code generation for, e.g., preheating simulations.

    .. automethod:: __init__
    .. automethod:: get_args
    .. autoattribute:: rhs_dict
    .. autoattribute:: reducers
    .. automethod:: stress_tensor
    """

    def __init__(self):
        """
        Processes input needed to specify a model for the particular
        :class:`Sector`.
        """

        raise NotImplementedError

    def get_args(self, single_stage=True):
        """
        :returns: A :class:`list` of all :class:`loopy.KernelArgument`'s
            relevant for a particular sector.

        :arg single_stage: Whether array shapes should include an outermost axis
            denoting temporary copies (i.e., for the implementation of
            classical Runge-Kutta methods).
            Defaults to *True*.
        """
        raise NotImplementedError

    @property
    def rhs_dict(self):
        """
        An ``@property`` method returning a :class:`dict` specifying the system
        of equations to be time-integrated.
        See the documentation of :class:`~pystella.step.Stepper`.
        """
        raise NotImplementedError

    @property
    def reducers(self):
        """
        An ``@property`` method returning :class:`dict` specifying the quantities
        to be computed, e.g., energy components for :class:`Expansion` and output.
        See the documentation of :class:`Reduction`.
        """
        raise NotImplementedError

    def stress_tensor(self, mu, nu, drop_trace=True):
        """
        :arg drop_trace: Whether to drop the term
            :math:`g_{\\mu\\nu} \\mathcal{L}`.
            Defauls to *False*.

        :returns: The component :math:`T_{\\mu\\nu}` of the stress-energy
            tensor of the particular :class:`Sector`.
            Used by :class:`TensorPerturbationSector`, with ``drop_trace=True``.
        """
        raise NotImplementedError


class ScalarSector(Sector):
    """
    A :class:`Sector` of scalar fields.

    .. automethod:: __init__
    """

    def __init__(self, nscalars, **kwargs):
        """
        :arg nscalars: The total number of scalar fields.

        The following keyword-only arguments are recognized:

        :arg f: The :class:`DynamicField` of scalar fields.
            Defaults to ``DynamicField('f', offset='h')``.

        :arg potential: A :class:`callable` which takes as input a
            :mod:`pymbolic` expression or a :class:`list` thereof, returning
            the potential of the scalar fields.
            Defaults to ``lambda x: 0``.

        :raises ValueError: if a particular field is coupled to its own kinetic
            term.
        """

        self.nscalars = nscalars
        self.f = kwargs.pop('f', DynamicField('f', offset='h'))
        self.potential = kwargs.pop('potential', lambda x: 0)

    def get_args(self, single_stage=True):
        if single_stage:
            shape = "(%d, Nx+2*h, Ny+2*h, Nz+2*h)" % self.nscalars
        else:
            shape = "(3, %d, Nx+2*h, Ny+2*h, Nz+2*h)" % self.nscalars
        lap_shape = "(%d, Nx, Ny, Nz)" % self.nscalars
        pd_shape = "(%d, 3, Nx, Ny, Nz)" % self.nscalars
        a = lp.ValueArg('a') if single_stage else lp.GlobalArg('a', shape=(3,))
        H = lp.ValueArg('hubble') if single_stage \
            else lp.GlobalArg('hubble', shape=(3,))

        all_args = \
            [
                lp.GlobalArg(self.f.name, shape=shape, offset=lp.auto),
                lp.GlobalArg(self.f.dot.name, shape=shape, offset=lp.auto),
                lp.GlobalArg(self.f.lap.name, shape=lap_shape, offset=lp.auto),
                lp.GlobalArg(self.f.pd.name, shape=pd_shape, offset=lp.auto),
                a,
                H,
            ]
        return all_args

    @property
    def rhs_dict(self):
        f = self.f
        H = Field('hubble', indices=[])
        a = Field('a', indices=[])

        rhs_dict = {}
        V = self.potential(f)

        for fld in range(self.nscalars):
            rhs_dict[f[fld]] = f.dot[fld]
            rhs_dict[f.dot[fld]] = (f.lap[fld]
                                    - 2 * H * f.dot[fld]
                                    - a**2 * diff(V, f[fld]))
        return rhs_dict

    @property
    def reducers(self):
        f = self.f
        a = var('a')

        reducers = {}
        reducers['kinetic'] = [f.dot[fld]**2 / 2 / a**2
                               for fld in range(self.nscalars)]
        reducers['potential'] = [self.potential(f)]
        reducers['gradient'] = [- f[fld] * f.lap[fld] / 2 / a**2
                                for fld in range(self.nscalars)]
        return reducers

    def stress_tensor(self, mu, nu, drop_trace=False):
        f = self.f
        a = Field('a', indices=[])

        Tmunu = sum(f.d(fld, mu) * f.d(fld, nu) for fld in range(self.nscalars))

        if drop_trace:
            return Tmunu
        else:
            metric = np.diag((-1/a**2, 1/a**2, 1/a**2, 1/a**2))  # contravariant
            lag = (- sum(sum(metric[mu, nu] * f.d(fld, mu) * f.d(fld, nu)
                             for mu in range(4) for nu in range(4))
                         for fld in range(self.nscalars)) / 2
                   - self.potential(self.f))
            metric = np.diag((-a**2, a**2, a**2, a**2))  # covariant
            return Tmunu + metric[mu, nu] * lag


def tensor_index(i, j):
    a = i if i <= j else j
    b = j if i <= j else i
    return (7 - a) * a // 2 - 4 + b


class TensorPerturbationSector:
    """
    A :class:`Sector` of tensor perturbations.

    .. automethod:: __init__
    """

    def __init__(self, sectors, **kwargs):
        """
        :arg sectors: The :class:`Sector`'s whose :meth:`~Sector.stress_tensor`'s
            source the tensor perturbations.

        The following keyword-only arguments are recognized:

        :arg hij: The :class:`DynamicField` of tensor fields.
            Defaults to ``DynamicField('hij', offset='h')``.
        """

        self.hij = kwargs.pop('hij', DynamicField('hij', offset='h'))
        self.sectors = sectors

    def get_args(self, single_stage=True):
        if single_stage:
            shape = "(6, Nx+2*h, Ny+2*h, Nz+2*h)"
        else:
            shape = "(3, 6, Nx+2*h, Ny+2*h, Nz+2*h)"
        shape_unpadded = "(6, Nx, Ny, Nz)"

        # assuming that a and H arguments are in other sectors
        all_args = \
            [
                lp.GlobalArg('hij', shape=shape, offset=lp.auto),
                lp.GlobalArg('dhijdt', shape=shape, offset=lp.auto),
                lp.GlobalArg('lap_hij', shape=shape_unpadded, offset=lp.auto),
            ]
        return all_args

    @property
    def rhs_dict(self):
        hij = self.hij
        H = Field('hubble', indices=[])

        rhs_dict = {}

        for i in range(1, 4):
            for j in range(i, 4):
                fld = tensor_index(i, j)
                Sij = sum(sector.stress_tensor(i, j, drop_trace=True)
                          for sector in self.sectors)
                rhs_dict[hij[fld]] = hij.dot[fld]
                rhs_dict[hij.dot[fld]] = (hij.lap[fld]
                                          - 2 * H * hij.dot[fld]
                                          + 16 * np.pi * Sij)

        return rhs_dict

    @property
    def reducers(self):
        return {}


def get_rho_and_p(energy):
    """
    Convenience callback for energy reductions which computes :math:`\\rho` and
    :math:`P` (really, :math:`a^2` times each).

    :arg energy: A dictionary of energy components as returned by
        :class:`~pystella.Reduction`.
    """

    energy['total'] = sum(sum(e) for e in energy.values())
    energy['pressure'] = 0
    if 'kinetic' in energy:
        energy['pressure'] += sum(energy['kinetic'])
    if 'gradient' in energy:
        energy['pressure'] += - sum(energy['gradient']) / 3
    if 'potential' in energy:
        energy['pressure'] += - sum(energy['potential'])

    return energy
