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
from pystella import Field
from pystella import Stencil, ElementWiseMap
from pystella.derivs import expand_stencil

__doc__ = """
.. currentmodule:: pystella.multigrid
.. autofunction:: pystella.multigrid.transfer.RestrictionBase
.. autofunction:: FullWeighting
.. autofunction:: Injection
.. autofunction:: pystella.multigrid.transfer.InterpolationBase
.. autofunction:: LinearInterpolation
.. autofunction:: CubicInterpolation
"""


def RestrictionBase(coefs, StencilKernel, halo_shape, **kwargs):
    """
    A base function for generating a restriction kernel.

    :arg coefs: The coefficients representing the restriction formula.
        Follows the convention of :func:`pystella.derivs.centered_diff`
        (since the restriction is applied recursively in each dimension).

    :arg StencilKernel: The stencil mapper to create an instance of.
        Defaults to :class:`~pystella.Stencil`.

    :arg halo_shape: The number of halo layers on (both sides of) each axis of
        the computational grid.
        Currently must be an :class:`int`.

    :arg lsize: The shape of prefetched arrays in shared memory.
        See :class:`~pystella.ElementWiseMap`.
        Defaults to ``(4, 4, 4)``.

    :arg correct: A :class:`bool` determining whether to produce a kernel which
        corrects an output array by the restricted array, or to only perform
        strict restriction.
        Defaults to *False*.

    :returns: An instance of ``StencilKernel`` which executes the requested
        restriction.
    """

    lsize = kwargs.pop('lsize', (4, 4, 4))

    # ensure grid dimensions are *not* passed, as they will be misinterpreted
    for N in ['Nx', 'Ny', 'Nz']:
        _ = kwargs.pop(N, None)

    restrict_coefs = {}
    for a, c_a in coefs.items():
        for b, c_b in coefs.items():
            for c, c_c in coefs.items():
                restrict_coefs[(a, b, c)] = c_a * c_b * c_c

    from pymbolic import parse, var
    i, j, k = parse('i, j, k')
    f1 = Field('f1', offset='h', indices=(2*i, 2*j, 2*k))
    f2 = Field('f2', offset='h')
    tmp = var('tmp')

    tmp_dict = {tmp: expand_stencil(f1, restrict_coefs)}

    if kwargs.pop('correct', False):
        restrict_dict = {f2: f2 - tmp}
    else:
        restrict_dict = {f2: tmp}

    args = [lp.GlobalArg('f1', shape='(2*Nx+2*h, 2*Ny+2*h, 2*Nz+2*h)'),
            lp.GlobalArg('f2', shape='(Nx+2*h, Ny+2*h, Nz+2*h)')]

    if isinstance(StencilKernel, Stencil):
        return StencilKernel(restrict_dict, tmp_instructions=tmp_dict, args=args,
                             prefetch_args=['f1'], halo_shape=halo_shape,
                             lsize=lsize, **kwargs)
    else:
        return StencilKernel(restrict_dict, tmp_instructions=tmp_dict, args=args,
                             halo_shape=halo_shape, lsize=lsize, **kwargs)


def FullWeighting(StencilKernel=Stencil, **kwargs):
    """
    Creates a full-weighting restriction kernel, which restricts in input array
    :math:`f^{(h)}` on the fine grid into an array :math:`f^{(2 h)}` on the
    coarse grid by applying

    .. math::

        f^{(2 h)}_i
        = \\frac{1}{4} f^{(h)}_{2 i - 1}
            + \\frac{1}{2} f^{(h)}_{2 i}
            + \\frac{1}{4} f^{(h)}_{2 i + 1}

    in each dimension.

    See :class:`transfer.RestrictionBase`.
    """

    from pymbolic.primitives import Quotient
    coefs = {-1: Quotient(1, 4), 0: Quotient(1, 2), 1: Quotient(1, 4)}
    return RestrictionBase(coefs, StencilKernel, **kwargs)


def Injection(StencilKernel=ElementWiseMap, **kwargs):
    """
    Creates an injection kernel, which restricts in input array
    :math:`f^{(h)}` on the fine grid into an array :math:`f^{(2 h)}` on the
    coarse grid by direct injection:

    .. math::

        f^{(2 h)}_{i, j ,k}
        = f^{(h)}_{2 i, 2 j, 2 k}

    See :class:`transfer.RestrictionBase`.
    """

    coefs = {0: 1}
    return RestrictionBase(coefs, StencilKernel, **kwargs)


def InterpolationBase(even_coefs, odd_coefs, StencilKernel, halo_shape, **kwargs):
    """
    A base function for generating a restriction kernel.

    :arg even_coefs: The coefficients representing the interpolation formula
        for gridpoints on the coarse and fine grid which coincide in space.
        Follows the convention of :func:`pystella.derivs.centered_diff`
        (since the restriction is applied recursively in each dimension).

    :arg odd_coefs: Same as ``even_coefs``, but for points on the fine grid which
        lie between points on the coarse grid.

    :arg StencilKernel: The stencil mapper to create an instance of.
        Defaults to :class:`~pystella.Stencil`.

    :arg halo_shape: The number of halo layers on (both sides of) each axis of
        the computational grid.
        Currently must be an :class:`int`.

    :arg correct: A :class:`bool` determining whether to produce a kernel which
        corrects an output array by the interpolated array, or to only perform
        strict interpolation.
        Defaults to *False*.

    :returns: An instance of ``StencilKernel`` which executes the requested
        interpolation.
    """

    from pymbolic import parse, var
    i, j, k = parse('i, j, k')
    f1 = Field('f1', offset='h')

    tmp_dict = {}
    tmp = var('tmp')

    import itertools
    for parity in tuple(itertools.product((0, 1), (0, 1), (0, 1))):
        result = 0
        for a, c_a in odd_coefs.items() if parity[0] else even_coefs.items():
            for b, c_b in odd_coefs.items() if parity[1] else even_coefs.items():
                for c, c_c in odd_coefs.items() if parity[2] else even_coefs.items():
                    f2 = Field('f2', offset='h',
                               indices=((i+a)//2, (j+b)//2, (k+c)//2))
                    result += c_a * c_b * c_c * f2

        tmp_dict[tmp[parity]] = result

    def is_odd(expr):
        from pymbolic.primitives import If, Comparison, Remainder
        return If(Comparison(Remainder(expr, 2), '==', 1), 1, 0)

    a, b, c = parse('a, b, c')
    for ind, val in zip((i, j, k), (a, b, c)):
        tmp_dict[val] = is_odd(ind)

    if kwargs.pop('correct', False):
        interp_dict = {f1: f1 + tmp[a, b, c]}
    else:
        interp_dict = {f1: tmp[a, b, c]}

    args = [lp.GlobalArg('f1', shape='(Nx+2*h, Ny+2*h, Nz+2*h)'),
            lp.GlobalArg('f2', shape='(Nx//2+2*h, Ny//2+2*h, Nz//2+2*h)')]

    return StencilKernel(interp_dict, tmp_instructions=tmp_dict, args=args,
                         prefetch_args=['f2'], halo_shape=halo_shape, **kwargs)


def LinearInterpolation(StencilKernel=Stencil, **kwargs):
    """
    Creates an linear interpolation kernel, which interpolates in input array
    :math:`f^{(h)}` on the fine grid into an array :math:`f^{(2 h)}` on the
    coarse grid via

    .. math::

        f^{(h)}_{2 i}
        &= f^{(2 h)}_{i}

        f^{(h)}_{2 i + 1}
        &= \\frac{1}{2} f^{(2 h)}_{i} + \\frac{1}{2} f^{(2 h)}_{i + 1}

    in each dimension.

    See :class:`transfer.InterpolationBase`.
    """

    from pymbolic.primitives import Quotient
    odd_coefs = {-1: Quotient(1, 2), 1: Quotient(1, 2)}
    even_coefs = {0: 1}

    return InterpolationBase(even_coefs, odd_coefs, StencilKernel, **kwargs)


def CubicInterpolation(StencilKernel=Stencil, **kwargs):
    """
    Creates an cubic interpolation kernel, which interpolates in input array
    :math:`f^{(h)}` on the fine grid into an array :math:`f^{(2 h)}` on the
    coarse grid via

    .. math::

        f^{(h)}_{2 i}
        &= f^{(2 h)}_{i}

        f^{(h)}_{2 i + 1}
        &= - \\frac{1}{16} f^{(2 h)}_{i - 1}
            + \\frac{9}{16} f^{(2 h)}_{i}
            + \\frac{9}{16} f^{(2 h)}_{i + 1}
            - \\frac{1}{16} f^{(2 h)}_{i + 2}

    in each dimension.

    See :class:`transfer.InterpolationBase`.
    """

    if kwargs.get('halo_shape', 0) < 2:
        raise ValueError('CubicInterpolation requires padding >= 2')

    from pymbolic.primitives import Quotient
    odd_coefs = {-3: Quotient(-1, 16), -1: Quotient(9, 16),
                 1: Quotient(9, 16), 3: Quotient(-1, 16)}
    even_coefs = {0: 1}

    return InterpolationBase(even_coefs, odd_coefs, StencilKernel, **kwargs)
