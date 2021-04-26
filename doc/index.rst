Welcome to pystella's documentation!
====================================

:mod:`pystella` is a Python-based framework enabling the easy expression and solution
of partial differential equations.
Here's a simple example which evolves the scalar wave equation (without doing
anything interesting):

.. literalinclude:: ../examples/wave_equation.py
    :language: python
    :lines: 24-

:mod:`pystella` uses :mod:`loopy` for code generation and :mod:`pyopencl` for
execution on CPUs and GPUs, with MPI parallelization across multiple OpenCL devices
via :mod:`mpi4py`.

For a more detailed tutorial on the tools to generate OpenCL kernels provided by
:mod:`loopy` and :mod:`pystella`, see
`codegen-tutorial.ipynb <https://github.com/zachjweiner/pystella/blob/main/examples/codegen-tutorial.ipynb>`_.
For a complete example which simulates scalar-field preheating after
cosmological inflation (optionally including gravitational waves), see
`scalar_preheating.py <https://github.com/zachjweiner/pystella/blob/main/examples/scalar_preheating.py>`_
(but note that ``grid_size`` and ``end_time`` are small by default so testing is
faster).

Table of Contents
-----------------

Please check :ref:`installation` to get started.

.. toctree::
    :maxdepth: 2

    ref_codegen
    ref_stepping
    ref_finitediff
    ref_fourier
    ref_other
    ref_multigrid
    installation
    changes
    license
    faq
    citing

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
