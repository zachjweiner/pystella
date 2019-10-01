pystella: a distributed and accelerated framework for PDE solving
=================================================================

.. image:: https://dev.azure.com/zachjweiner/pystella/_apis/build/status/zachjweiner.pystella?branchName=master
    :alt: Azure Build Status
    :target: https://dev.azure.com/zachjweiner/pystella/_build/latest?definitionId=1&branchName=master
.. image:: https://readthedocs.org/projects/pystella/badge/?version=latest
    :target: https://pystella.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status

``pystella`` enables the easy expression of both PDE systems and the algorithms
to solve them in high-performance computing environments within Python.
It provides interfaces to generate custom computational kernels
via `loopy <http://mathema.tician.de/software/loopy>`_ which can then be executed
from Python on (multiple) CPUs or GPUs using
`pyopencl <http://mathema.tician.de/software/pyopencl>`_
and `mpi4py <https://mpi4py.readthedocs.io/en/stable/>`_.
Moreover, ``pystella`` implements a number of algorithms for PDE time evolution
and spatial discretization that can be readily applied to a variety of physical
systems.

Its features include:

* code generation for performant element-wise kernels, stencil-based computations,
  and reductions
* distributed domain decomposition and grid boundary sychronization
* time-stepping algorithms, including low-storage Runge-Kutta schemes
* finite-difference and spectral-collocation methods for spatial derivatives
* a geometric multigrid solver for generic sets of nonlinear boundary-value problems
  (in beta)
* wrappers to OpenCL-based Fast Fourier Transforms (FFTs) and distributed
  CPU FFTs
* Fourier space methods for field analysis and solving Poisson's equation

All of the above functionality is configured to run at high performance by default,
as are the interfaces for generating custom kernels (though this is
entirely user-configurable!).
Additionally, the provided functionality is intended to work seamlessly whether
running in distributed- (i.e., multiple devices) or shared-memory
(i.e., a single device) contexts without sacrificing performance in either case.

``pystella`` was designed to simulate preheating and gravitational wave production
after cosmological inflation and provides a simple way to specify models of this
process.
However, ``pystella`` is also designed to be sufficiently abstract as to provide a
good framework for most systems that can be discretized onto grids
(e.g., lattice field theory, (magneto)hydrodynamics, Einstein's equations,
electromagnetism, etc.).
The preheating-specific components can be viewed as examples for the symbolic
representation of arbitrary physical systems as an interface to its code generation
routines.
``pystella`` provides entry points at varying levels of abstraction—so if you like
the idea of ``pystella`` but the algorithms you require are not implemented,
you can create new interfaces (or extend existing ones) for your purposes
with ease.
(Better yet, consider contributing a PR!)

``pystella`` is `fully documented <https://pystella.readthedocs.io/en/latest/>`_
and is licensed under the liberal `MIT license
<http://en.wikipedia.org/wiki/MIT_License>`_.
