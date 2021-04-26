Frequently Asked Questions
==========================

What doesn't :mod:`pystella` do for me?
---------------------------------------

:mod:`pystella` cannot ensure that kernels are
properly ordered (i.e., with regard to data dependence).
For example, in
`wave_equation.py <https://github.com/zachjweiner/pystella/blob/main/examples/wave_equation.py>`_,
the Laplacian of ``f`` has to be computed by ``derivs`` before
``lap_f`` is read by ``stepper``.
:mod:`pystella` only creates kernels that do exactly what you
ask for (and nothing more).

What domain dimension does :mod:`pystella` support?
---------------------------------------------------

Currently, three dimensions or fewer are supported.
While :mod:`pystella` was designed for 3-D problems, one can, for example, implement
a 2-D problem by simply setting one axis to have length 1 and halo padding 0
(to remove needless halo layers), e.g.,
``grid_shape = (1, 4096, 4096)`` and ``halo_shape = (0, 2, 2)``.
Be warned that some implemented functionality may not run optimally
in this case, and that 1- and 2-D domains have not been thoroughly tested.

Where can I find more examples?
-------------------------------

Aside from the complete examples implementing the
`wave equation <https://github.com/zachjweiner/pystella/blob/main/examples/wave_equation.py>`_
and simulating gravitational waves from
`scalar-field preheating <https://github.com/zachjweiner/pystella/blob/main/examples/scalar_preheating.py>`_,
:mod:`pystella`'s `tests <https://github.com/zachjweiner/pystella/tree/main/test>`_
provide another source of examples.

Common gotchas/pitfalls
-----------------------

* ``loopy.diagnostic.LoopyError: could not determine type of '___'``

  You likely either forgot to pass an array argument when calling a kernel
  or to fix a value at kernel creation (i.e., by passing
  ``fixed_parameters=dict(...)``).
