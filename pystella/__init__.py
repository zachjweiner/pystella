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


from pystella.field import (Field, DynamicField, index_fields, shift_fields,
                            diff, get_field_args)
from pystella.sectors import Sector, ScalarSector, TensorPerturbationSector
from pystella.elementwise import ElementWiseMap
from pystella.stencil import Stencil, StreamingStencil
from pystella.reduction import Reduction, FieldStatistics
from pystella.histogram import Histogrammer, FieldHistogrammer
from pystella.step import (RungeKutta4, RungeKutta3SSP, RungeKutta3Heun,
                           RungeKutta3Nystrom, RungeKutta3Ralston,
                           RungeKutta2Midpoint, RungeKutta2Ralston, LowStorageRK54,
                           LowStorageRK3Williamson, LowStorageRK3Inhomogeneous,
                           LowStorageRK3SSP)
from pystella.derivs import FiniteDifferencer
from pystella.decomp import DomainDecomposition
from pystella.expansion import Expansion
from pystella.fourier import (DFT, RayleighGenerator, Projector, PowerSpectra,
                              SpectralCollocator, SpectralPoissonSolver)

from loopy import set_caching_enabled
set_caching_enabled(True)

import logging
logger = logging.getLogger(__name__)


def choose_device_and_make_context(platform_choice=None, device_choice=None):
    """
    A wrapper that chooses a device and creates a :class:`pyopencl.Context` on
    a particular device.

    :arg platform_choice: An integer or string specifying which
        :class:`pyopencl.Platform` to choose.
        Defaults to *None*, in which case the environment variables
        ``PYOPENCL_CTX`` or ``PYOPENCL_TEST`` are queried.
        If none of the above are specified, then the first platform is chosen.

    :arg device_choice: An integer or string specifying which
        :class:`pyopencl.Device` to run on.
        Defaults to *None*, in which case a device is chosen according to the
        node-local MPI rank.
        (Note that this requires initializing MPI, i.e., importing ``mpi4py.MPI``.)

    :returns: A :class:`pyopencl.Context`.
    """

    import pyopencl as cl

    if platform_choice is None:
        import os
        if "PYOPENCL_CTX" in os.environ:
            ctx_spec = os.environ["PYOPENCL_CTX"]
            platform_choice = ctx_spec.split(":")[0]
    else:
        platform_choice = str(platform_choice)

    from pyopencl.tools import get_test_platforms_and_devices
    platform, devices = get_test_platforms_and_devices(platform_choice)[0]
    num_devices = len(devices)
    logger.info(f"platform {platform.name} with {num_devices} devices selected")

    try:
        # sort devices based on their unique pci bus id
        devices = sorted(devices, key=lambda dev: dev.pci_bus_id_nv)
    except:  # noqa
        logger.info("Non-NVIDIA platform; no pci_bus_id_nv attribute to sort on.")

    if device_choice is None:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD.Split_type(MPI.COMM_TYPE_SHARED)
        local_rank = comm.Get_rank()
        comm.Free()
        device_choice = local_rank % num_devices

    dev = devices[device_choice]

    from socket import getfqdn
    host_dev_info = f"on host {getfqdn()}: chose {dev.name} number {device_choice}"
    if "NVIDIA" in platform.name:
        host_dev_info += f" with pci_bus_id_nv={dev.pci_bus_id_nv}"
    logger.info(host_dev_info)

    return cl.Context([dev])


class DisableLogging():
    def __init__(self, logger):
        self.logger = logger

    def __enter__(self):
        self.original_level = self.logger.level
        self.logger.setLevel(logging.CRITICAL)

    def __exit__(self, exception_type, exception_value, traceback):
        self.logger.setLevel(self.original_level)


__all__ = [
    "Field",
    "DynamicField",
    "index_fields",
    "shift_fields",
    "diff",
    "get_field_args",
    "Sector",
    "ScalarSector",
    "TensorPerturbationSector",
    "ElementWiseMap",
    "RungeKutta4",
    "RungeKutta3SSP",
    "RungeKutta3Heun",
    "RungeKutta3Nystrom",
    "RungeKutta3Ralston",
    "RungeKutta2Midpoint",
    "RungeKutta2Ralston",
    "LowStorageRK54",
    "LowStorageRK3Williamson",
    "LowStorageRK3Inhomogeneous",
    "LowStorageRK3SSP",
    "Stencil",
    "StreamingStencil",
    "FiniteDifferencer",
    "Reduction",
    "FieldStatistics",
    "Histogrammer",
    "FieldHistogrammer",
    "DomainDecomposition",
    "Expansion",
    "DFT",
    "RayleighGenerator",
    "Projector",
    "PowerSpectra",
    "SpectralCollocator",
    "SpectralPoissonSolver",
    "choose_device_and_make_context",
]
