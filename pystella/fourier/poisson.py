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
import pyopencl.array as cla

__doc__ = """
.. currentmodule:: pystella
.. autoclass:: SpectralPoissonSolver
"""


class SpectralPoissonSolver:
    """
    A Fourier-space solver for the Poisson equation,

    .. math::

        \\nabla^2 f - m^2 f = \\rho,

    allowing a term linear in :math:`f` with coefficient :math:`m^2`.

    .. automethod:: __init__
    .. automethod:: __call__
    """

    def __init__(self, fft, dk, dx, effective_k):
        """
        The following arguments are required:

        :arg fft: An FFT object as returned by :func:`~pystella.DFT`.
            ``grid_shape`` and ``dtype`` are determined by ``fft``'s attributes.

        :arg dk: A 3-:class:`tuple` of the momentum-space grid spacing of each
            axis (i.e., the infrared cutoff of the grid in each direction).

        :arg dx: A 3-:class:`tuple` specifying the grid spacing of each axis.

        :arg effective_k: A :class:`~collections.abc.Callable` with signature
            ``(k, dx)`` returning
            the eigenvalue of the second-difference stencil corresponding to
            :math:`\\nabla^2`.
            That is, the solver is implemented relative to the stencil
            whose eigenvalues are returned by this function.
        """

        self.fft = fft
        grid_size = fft.grid_shape[0] * fft.grid_shape[1] * fft.grid_shape[2]

        queue = self.fft.sub_k['momenta_x'].queue
        sub_k = list(x.get().astype('int') for x in self.fft.sub_k.values())
        k_names = ('k_x', 'k_y', 'k_z')
        self.momenta = {}
        self.momenta = {}
        for mu, (name, kk) in enumerate(zip(k_names, sub_k)):
            kk_mu = effective_k(dk[mu] * kk.astype(fft.rdtype), dx[mu])
            self.momenta[name] = cla.to_device(queue, kk_mu)

        args = [
            lp.GlobalArg('fk', fft.cdtype, shape="(Nx, Ny, Nz)"),
            lp.GlobalArg("k_x", fft.rdtype, shape=('Nx',)),
            lp.GlobalArg("k_y", fft.rdtype, shape=('Ny',)),
            lp.GlobalArg("k_z", fft.rdtype, shape=('Nz',)),
            lp.ValueArg('m_squared', fft.rdtype),
        ]

        from pystella.field import Field
        from pymbolic.primitives import Variable, If, Comparison

        fk = Field('fk')
        indices = fk.indices
        rho_tmp = Variable('rho_tmp')
        tmp_insns = [(rho_tmp, Field('rhok') * (1/grid_size))]

        mom_vars = tuple(Variable(name) for name in k_names)
        minus_k_squared = sum(kk_i[x_i] for kk_i, x_i in zip(mom_vars, indices))
        sol = rho_tmp / (minus_k_squared - Variable('m_squared'))

        solution = {Field('fk'): If(Comparison(minus_k_squared, '<', 0), sol, 0)}

        from pystella.elementwise import ElementWiseMap
        options = lp.Options(return_dict=True)
        self.knl = ElementWiseMap(solution, args=args, halo_shape=0, options=options,
                                  tmp_instructions=tmp_insns, lsize=(16, 2, 1))

    def __call__(self, queue, fx, rho, m_squared=0, allocator=None):
        """
        Executes the Poisson solver.

        :arg queue: A :class:`pyopencl.CommandQueue`.

        :arg fx: The array in which the solution :math:`f` is stored.

        :arg rho: The array containing the right-hand--side, :math:`\\rho`.

        :arg m_squared: The value of the coefficient :math:`m^2` of the
            linear term in the Poisson equation to be solved.
            Defaults to ``0``.

        .. versionchanged:: 2020.1

            Added `m_squared` to support solving with a linear term :math:`m^2 f`.
        """

        rhok = self.fft.dft(rho)
        evt, out = self.knl(queue, rhok=rhok, fk=rhok, m_squared=m_squared,
                            **self.momenta)
        self.fft.idft(out['fk'], fx)
