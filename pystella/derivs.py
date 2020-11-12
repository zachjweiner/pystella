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


from pystella.stencil import Stencil, StreamingStencil
from pystella.field import Field
from pystella.field import shift_fields

__doc__ = """
.. currentmodule:: pystella
.. autoclass:: FiniteDifferencer
.. currentmodule:: pystella.derivs
.. autofunction:: pystella.derivs.expand_stencil
.. autofunction:: pystella.derivs.centered_diff
"""


def expand_stencil(f, coefs):
    """
    Expands a stencil over a field.

    :arg f: A :class:`~pystella.Field`.

    :arg coefs: A :class:`dict` whose values are the coefficients of the stencil
        at an offset given by the key. The keys must be 3-:class:`tuple`\\ s, and the
        values may be :mod:`pymbolic` expressions or constants.

    Example::

        >>> f = Field("f", offset="h")
        >>> coefs = {(1, 0, 0): 1, (-1, 0, 0): -1}
        >>> stencil = expand_stencil(f, coefs)
        >>> print(index_fields(stencil))
        f[i + h + 1, j + h, k + h] + (-1)*f[i + h + -1, j + h, k + h]
    """

    return sum([c * shift_fields(f, shift=offset) for offset, c in coefs.items()])


def centered_diff(f, coefs, direction, order):
    """
    A convenience wrapper to :func:`expand_stencil` for computing centered
    differences. By assuming the symmetry of the stencil (which has parity given
    by the parity of ``order``), no redundant coefficients need to be supplied.
    Further, by supplying the ``direction`` parameter, the input offset (the keys
    of ``coefs``) need only be integers.

    :arg f: A :class:`~pystella.Field`.

    :arg coefs: A :class:`dict` whose values are the coefficients of the stencil
        at an offset given by the key. The keys must be integers, and the
        values may be :mod:`pymbolic` expressions or constants. Only
        non-redundant ``(offset, coefficient)`` pairs are needed.

    :arg direction: An integer in ``(0, 1, 2)`` denoting the direction over which
        to expand the stencil (i.e., to apply the offset).

    :arg order: The order of the derivative being computed, which determines
        whether coefficients at the opposite offset have the same or opposite
        sign.

    Example::

        >>> f = Field("f", offset="h")
        >>> coefs = {1: 1}
        >>> stencil = centered_diff(f, coefs, 0, 1)
        >>> print(index_fields(stencil))
        f[i + h + 1, j + h, k + h] + (-1)*f[i + h + -1, j + h, k + h]
    """

    all_coefs = {}
    for s, c in coefs.items():
        offset = [0, 0, 0]

        # skip central point (s == 0) for odd order
        if s != 0 or order % 2 == 0:
            offset[direction-1] = s
            all_coefs[tuple(offset)] = c

        # add the opposite point
        if s != 0:
            offset[direction-1] = - s
            all_coefs[tuple(offset)] = (-1)**order * c

    return expand_stencil(f, all_coefs)


class FiniteDifferenceStencil:
    coefs = NotImplemented
    truncation_order = NotImplemented
    order = NotImplemented
    is_centered = NotImplemented

    def __call__(self, f, direction):
        if self.is_centered:
            return centered_diff(f, self.coefs, direction, self.order)
        else:
            return expand_stencil(f, self.coefs)

    def get_eigenvalues(self, k, dx):
        raise NotImplementedError


_grad_coefs = {}
_grad_coefs[1] = {1: 1/2}
_grad_coefs[2] = {1: 8/12, 2: -1/12}
_grad_coefs[3] = {1: 45/60, 2: -9/60, 3: 1/60}
_grad_coefs[4] = {1: 672/840, 2: -168/840, 3: 32/840, 4: -3/840}


class FirstCenteredDifference(FiniteDifferenceStencil):
    def __init__(self, h):
        self.coefs = _grad_coefs[h]
        self.truncation_order = 2 * h
        self.order = 1
        self.is_centered = True

    def get_eigenvalues(self, k, dx):
        import numpy as np
        th = k * dx
        if self.truncation_order == 2:
            return np.sin(th) / dx
        if self.truncation_order == 4:
            return (8 * np.sin(th) - np.sin(2 * th)) / (6 * dx)
        if self.truncation_order == 6:
            return (45 * np.sin(th) - 9 * np.sin(2 * th)
                    + np.sin(3 * th)
                    ) / (30 * dx)
        if self.truncation_order == 8:
            return (672 * np.sin(th) - 168 * np.sin(2 * th)
                    + 32 * np.sin(3 * th) - 3 * np.sin(4 * th)
                    ) / (420 * dx)
        else:
            return k


_lap_coefs = {}
_lap_coefs[1] = {0: -2, 1: 1}
_lap_coefs[2] = {0: -30/12, 1: 16/12, 2: -1/12}
_lap_coefs[3] = {0: -490/180, 1: 270/180, 2: -27/180, 3: 2/180}
_lap_coefs[4] = {0: -14350/5040, 1: 8064/5040, 2: -1008/5040,
                 3: 128/5040, 4: -9/5040}


class SecondCenteredDifference(FiniteDifferenceStencil):
    def __init__(self, h):
        self.coefs = _lap_coefs[h]
        self.truncation_order = 2 * h
        self.order = 2
        self.is_centered = True

    def get_eigenvalues(self, k, dx):
        import numpy as np
        th = k * dx
        if self.truncation_order == 2:
            return (2 * np.cos(th) - 2) / dx**2
        elif self.truncation_order == 4:
            return (32 * np.cos(th) - 2 * np.cos(2 * th) - 30) / (12 * dx**2)
        elif self.truncation_order == 6:
            return (90 * np.cos(th) - 9 * np.cos(2 * th)
                    + 2/3 * np.cos(3 * th) - 245/3
                    ) / (30 * dx**2)
        elif self.truncation_order == 8:
            return (1344 * np.cos(th) - 168 * np.cos(2 * th)
                    + 64/3 * np.cos(3 * th) - 3/2 * np.cos(4 * th) - 7175/6
                    ) / (420 * dx**2)
        else:
            return - k**2


knl_h_arch_lsizes = {
    ("gradlap", 1, "volta"): (32, 16, 1),
    ("grad", 1, "volta"): (32, 16, 1),
    ("lap", 1, "volta"): (32, 16, 2),
    ("gradlap", 1, "pascal"): (32, 16, 1),
    ("grad", 1, "pascal"): (32, 16, 1),
    ("lap", 1, "pascal"): (32, 16, 4),
    ("gradlap", 2, "volta"): (32, 16, 1),
    ("grad", 2, "volta"): (32, 16, 1),
    ("lap", 2, "volta"): (32, 16, 2),
    ("gradlap", 2, "pascal"): (32, 8, 2),
    ("grad", 2, "pascal"): (32, 8, 2),
    ("lap", 2, "pascal"): (16, 8, 4),
    ("gradlap", 3, "volta"): (32, 8, 1),
    ("grad", 3, "volta"): (32, 8, 2),
    ("lap", 3, "volta"): (32, 8, 4),
    ("gradlap", 3, "pascal"): (16, 8, 4),
    ("grad", 3, "pascal"): (16, 8, 4),
    ("lap", 3, "pascal"): (16, 8, 4),
    ("gradlap", 4, "volta"): (32, 4, 4),
    ("grad", 4, "volta"): (32, 4, 4),
    ("lap", 4, "volta"): (16, 8, 4),
    ("gradlap", 4, "pascal"): (16, 8, 2),
    ("grad", 4, "pascal"): (16, 8, 2),
    ("lap", 4, "pascal"): (16, 8, 4)
}


class FiniteDifferencer:
    """
    A convenience class for generating kernels which compute spatial gradients,
    Laplacians, and combinations thereof.

    See :class:`SpectralCollocator` for a version of this
    class implementing spectral collocation.

    The following arguments are required:

    :arg decomp: An instance of :class:`DomainDecomposition`.

    :arg halo_shape: The number of halo layers on (both sides of) each axis of
        the computational grid.
        Currently must be an :class:`int`.

    :arg dx: A 3-:class:`tuple` specifying the grid spacing of each axis.

    The following keyword-only arguments are recognized:

    :arg first_stencil: A :class:`~collections.abc.Callable` with signature
        ``(f, direction)`` where f is a :class:`Field` and ``direction``
        indicates the spatial axis (1, 2, or 3) along which the stencil is taken,
        returning the (symbolic) first-order stencil.
        Defaults to the centered difference of the highest order allowed
        by the amount of array padding (set by :attr:`halo_shape`).
        See :func:`~pystella.derivs.expand_stencil`.

    :arg second_stencil: Like ``first_stencil``, but for second-order
        differences.

    :arg rank_shape: A 3-:class:`tuple` specifying the shape of looped-over
        arrays.
        Defaults to *None*, in which case these values are not fixed (and
        will be inferred when the kernel is called at a slight performance
        penalty).

    .. ifconfig:: not on_rtd

        :arg stream: Whether to use :class:`StreamingStencil`.
            Defaults to *False*.

    .. automethod:: __call__
    .. automethod:: divergence
    """

    def __init__(self, decomp, halo_shape, dx, **kwargs):
        self.decomp = decomp
        stream = kwargs.pop("stream", False)
        first_stencil = kwargs.pop("first_stencil",
                                   FirstCenteredDifference(halo_shape))
        second_stencil = kwargs.pop("second_stencil",
                                    SecondCenteredDifference(halo_shape))
        rank_shape = kwargs.pop("rank_shape", None)

        fx = Field("fx", offset="h")
        pd = tuple(Field(pdi) for pdi in ("pdx", "pdy", "pdz"))
        pdx, pdy, pdz = ({pdi: first_stencil(fx, i+1) * (1/dxi)}
                         for i, (pdi, dxi) in enumerate(zip(pd, dx)))
        lap = {Field("lap"): sum(second_stencil(fx, i+1) * dxi**-2
                                 for i, dxi in enumerate(dx))}

        common_args = dict(halo_shape=halo_shape, prefetch_args=["fx"],
                           rank_shape=rank_shape)

        self.pdx_knl = Stencil(pdx, lsize=(16, 2, 16), **common_args)
        self.pdy_knl = Stencil(pdy, lsize=(16, 16, 2), **common_args)
        self.pdz_knl = Stencil(pdz, lsize=(16, 8, 2), **common_args)

        pdx_incr, pdy_incr, pdz_incr = (
            {Field("div"): Field("div") + first_stencil(fx, i+1) * (1/dxi)}
            for i, dxi in enumerate(dx)
        )

        self.pdx_incr_knl = Stencil(pdx_incr, lsize=(16, 2, 16), **common_args)
        self.pdy_incr_knl = Stencil(pdy_incr, lsize=(16, 16, 2), **common_args)
        self.pdz_incr_knl = Stencil(pdz_incr, lsize=(16, 8, 2), **common_args)

        _h = max(max(first_stencil.coefs.keys()), max(second_stencil.coefs.keys()))

        if stream:
            arch = "volta"
            if "target" in kwargs:
                dev = kwargs["target"].device
                try:
                    arch_map = {6: "pascal", 7: "volta"}
                    arch = arch_map[dev.compute_capability_major_nv]
                except:  # noqa: E722
                    pass

            gradlap_lsize = kwargs.get(
                "gradlap_lsize", knl_h_arch_lsizes[("gradlap", _h, arch)])
            grad_lsize = kwargs.get(
                "grad_lsize", knl_h_arch_lsizes[("grad", _h, arch)])
            lap_lsize = kwargs.get(
                "lap_lsize", knl_h_arch_lsizes[("lap", _h, arch)])
        else:
            lsize = {1: (8, 8, 8), 2: (8, 4, 4), 3: (4, 4, 4), 4: (2, 2, 2)}[_h]
            gradlap_lsize = kwargs.get("gradlap_lsize", lsize)
            grad_lsize = kwargs.get("grad_lsize", lsize)
            lap_lsize = kwargs.get("lap_lsize", lsize)

        SS = StreamingStencil if stream else Stencil
        self.grad_lap_knl = SS({**pdx, **pdy, **pdz, **lap}, lsize=gradlap_lsize,
                               **common_args)
        self.grad_knl = SS({**pdx, **pdy, **pdz}, lsize=grad_lsize, **common_args)
        self.lap_knl = SS(lap, lsize=lap_lsize, **common_args)

    def __call__(self, queue, fx, *,
                 lap=None, pdx=None, pdy=None, pdz=None, grd=None, allocator=None):
        """
        Computes requested derivatives of the input ``fx``.

        :arg queue: A :class:`pyopencl.CommandQueue`.

        :arg fx: The array to compute derivatives of. Halos are shared using
            :meth:`DomainDecomposition.share_halos`, and a kernel is called
            based on what combination of the remainin input arguments are not *None*.

            Valid combinations are

            * all of ``lap``, ``pdx``, ``pdy``, and ``pdz``
              (or equivalently ``lap`` and ``grd``)

            * any single one of ``lap``, ``pdx``, ``pdy``, or ``pdz``

            * only ``pdx``, ``pdy``, and ``pdz``
              (or equivalently only ``grd``)

            If ``fx`` has shape ``(...,) + (rank_shape+2*halo_shape)``, all the
            outermost indices (i.e., in place of ``...``) are looped over.
            As an example, with ``halo_shape=1``::

                >>> fx.shape, lap.shape
                ((2, 3, 130, 130, 130), (2, 3, 128, 128, 128))
                >>> derivs(queue, fx=fx, lap=lap)

            would loop over the outermost two axes with shape ``(2, 3)``.
            Note that the shapes of ``fx`` and ``lap`` (or in general all input
            arrays) must match on these outer axes.

        :arg lap: The array which will store the Laplacian of ``fx``.
            Defaults to *None*.

        :arg pdx: The array which will store the :math:`x`-derivative of ``fx``.
            Defaults to *None*.

        :arg pdy: The array which will store the :math:`y`-derivative of ``fx``.
            Defaults to *None*.

        :arg pdz: The array which will store the :math:`z`-derivative of ``fx``.
            Defaults to *None*.

        :arg grd: The array containing the gradient of ``fx``, i.e., all three of
            ``pdx``, ``pdy``, and ``pdz``.
            If supplied, any input values to ``pdx``, ``pdy``, or ``pdz`` are
            ignored and replaced via ::

                pdx = grd[..., 0, :, :, :]
                pdy = grd[..., 1, :, :, :]
                pdz = grd[..., 2, :, :, :]

            Defaults to *None*.

        :returns: The :class:`pyopencl.Event` associated with the kernel
            invocation (i.e., of the last called kernel if multiple axes are
            being looped over).
        """

        from itertools import product
        slices = list(product(*[range(n) for n in fx.shape[:-3]]))

        if grd is not None:
            if isinstance(grd, (tuple, list)):
                pdx, pdy, pdz = grd
            else:
                pdx = grd[..., 0, :, :, :]
                pdy = grd[..., 1, :, :, :]
                pdz = grd[..., 2, :, :, :]

        for s in slices:
            self.decomp.share_halos(queue, fx[s])
            if (lap is not None and pdx is not None
                    and pdy is not None and pdz is not None):
                evt, _ = self.grad_lap_knl(queue, fx=fx[s], lap=lap[s],
                                           pdx=pdx[s], pdy=pdy[s], pdz=pdz[s])
            elif pdx is not None and pdy is not None and pdz is not None:
                evt, _ = self.grad_knl(queue, fx=fx[s],
                                       pdx=pdx[s], pdy=pdy[s], pdz=pdz[s])
            elif lap is not None:
                evt, _ = self.lap_knl(queue, fx=fx[s], lap=lap[s])
            elif pdx is not None:
                evt, _ = self.pdx_knl(queue, fx=fx[s], pdx=pdx[s])
            elif pdy is not None:
                evt, _ = self.pdy_knl(queue, fx=fx[s], pdy=pdy[s])
            elif pdz is not None:
                evt, _ = self.pdz_knl(queue, fx=fx[s], pdz=pdz[s])

        return evt

    def divergence(self, queue, vec, div, allocator=None):
        """
        Computes the divergence of the input ``vec``.

        :arg queue: A :class:`pyopencl.CommandQueue`.

        :arg vec: The array to compute the divergence of. Halos are shared using
            :meth:`DomainDecomposition.share_halos`.

            If ``vec`` has shape ``(...,) + (3, rank_shape+2*halo_shape)``, all the
            outermost indices (i.e., in place of ``...``) are looped over.
            As an example, with ``halo_shape=1``::

                >>> vec.shape, div.shape
                ((2, 3, 130, 130, 130), (2, 128, 128, 128))
                >>> derivs.divergence(queue, vec, div)

            would loop over the outermost axis with shape ``(2,)``.
            Note that the shapes of ``vec`` and ``div`` must match on these
            outer axes.

        :arg div: The array which will store the divergence of ``vec``.

        :returns: The :class:`pyopencl.Event` associated with the kernel
            invocation (i.e., of the last called kernel if multiple axes are
            being looped over).
        """

        from itertools import product
        slices = list(product(*[range(n) for n in vec.shape[:-4]]))

        for s in slices:
            self.decomp.share_halos(queue, vec[s][0])
            evt, _ = self.pdx_knl(queue, fx=vec[s][0], pdx=div[s])
            self.decomp.share_halos(queue, vec[s][1])
            evt, _ = self.pdy_incr_knl(queue, fx=vec[s][1], div=div[s])
            self.decomp.share_halos(queue, vec[s][2])
            evt, _ = self.pdz_incr_knl(queue, fx=vec[s][2], div=div[s])

        return evt
