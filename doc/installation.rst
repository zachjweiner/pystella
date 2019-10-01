.. highlight:: sh

.. _installation:

Installation
============

At the bare minimum, :mod:`pystella` requires :mod:`numpy`,
:mod:`loopy` for code generation, and :mod:`pyopencl`
(plus an OpenCL implementation) for kernel execution.
Optional dependencies (and what they are needed for) are:

* :mod:`mpi4py` (and an MPI implementation) for distributed, multi-device execution

* :mod:`gpyfft` (and :mod:`clfft` and :mod:`Cython`) for OpenCL
  Fast Fourier Transforms (:class:`pystella.fourier.gDFT`) (e.g., to run on a GPU),
  and/or :mod:`mpi4py_fft` (and :mod:`fftw`) for distributed, CPU FFTs
  (:class:`pystella.fourier.pDFT`)

* :mod:`h5py` (and :mod:`hdf5`) to use the convenience class
  :class:`pystella.output.OutputFile`

* :mod:`sympy`, to interoperate with :mod:`pymbolic`

Fortunately, :mod:`conda` greatly simplifies the installation process of any
of these dependencies.
The included :file:`environment.yml` file provides a complete
installation by default, but one can remove any optional dependencies.

Note that installation has only been tested on Linux, but similar steps should work
on macOS.

Installation steps
------------------

Install via the following steps
(first modifying :file:`environment.yml` as desired):

1. Install `miniconda <https://docs.conda.io/en/latest/miniconda.html>`_ (if you
   haven't already installed :mod:`conda`).

2. Clone the repository::

    git clone https://github.com/zachjweiner/pystella.git

3. Create a :mod:`pystella` environment as specified by :file:`environment.yml`::

    conda env create --file pystella/environment.yml

   -  This will clone and install (i.e., as if via
      :command:`python setup.py install`) :mod:`gpyfft` and :mod:`loopy` into
      :command:`src/`. To change this, first define the environment variable
      :command:`PIP_SRC` to be your desired directory,
      e.g., to your home directory with::

        export PIP_SRC=~

  Alternatively, update your active environment via::

    conda env update --file pystella/environment.yml

4. Activate the environment (if you created a new one)::

    conda activate pystella

  and set up :mod:`pystella`::

    cd pystella/ && python setup.py develop

To test that installation was successful, try running an example
(e.g., :code:`python examples/scalar-preheating.py`) or run the tests with :mod:`pytest`.

Running on other devices (GPUs, etc.)
-------------------------------------

The included :file:`environment.yml` installs `pocl <http://portablecl.org/>`__,
which provides an OpenCL implementation on most CPUs.
Enabling execution on other hardware (e.g., GPUs) requires making :mod:`pyopencl`
aware of the corresponding OpenCL driver.
See :mod:`pyopencl`'s
`instructions <https://documen.tician.de/pyopencl/misc.html#installation>`__
(specifically,
`here <https://documen.tician.de/pyopencl/misc.html#using-vendor-supplied-opencl-drivers-mainly-on-linux>`__).
For example, installing `CUDA <https://developer.nvidia.com/cuda-downloads>`__
installs the driver for NVIDIA GPUs; one must then merely copy
the :file:`nvidia.icd` file via::

    cp /etc/OpenCL/vendors/nvidia.icd $CONDA_PREFIX/etc/OpenCL/vendors

Using an existing MPI implementation
------------------------------------

To enable MPI support without :mod:`conda` installing its own MPI implementation
(e.g., to use the optimized implementation already provided on a cluster, etc.),
simply move :mod:`mpi4py` (and :mod:`mpi4py_fft`) below the :code:`pip` line
in :file:`environment.yml`::

    ...
    - pip:
      - mpi4py
      - mpi4py-fft
     ...

:mod:`pip`-installing :mod:`mpi4py` assumes that :code:`mpicc` is available
(check the output of :code:`which mpicc`).
See :mod:`mpi4py`'s
`instructions <https://mpi4py.readthedocs.io/en/stable/install.html>`__ for more
details.
