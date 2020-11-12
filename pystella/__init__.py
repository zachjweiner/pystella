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


def choose_device_and_make_context(platform_choice=None, device_choice=None):
    """
    A wrapper to choose a device and create a :class:`pyopencl.Context` on
    a particular device.

    :arg platform_number: An integer specifying which element of the
        :class:`list` returned by :func:`pyopencl.get_platforms` to choose.
        Defaults to *None*, in which case a NVIDIA platform.
        If one is not found, then the first platform is chosen.

    :arg device_number: An integer specifying which device to run on.
        Defaults to *None*, in which case a device is chosen according to any
        available environment variable defining the local MPI rank (defaulting to 0).
        Currently only looks for SLURM, OpenMPI, and MVAPICH environment variables.

    :returns: A :class:`pyopencl.Context`.
    """

    import pyopencl as cl

    # look for NVIDIA platform
    platform = None
    platforms = cl.get_platforms()
    if platform_choice is None:
        for i, plt in enumerate(platforms):
            if "NVIDIA" in plt.name:
                platform = plt
        platform = platform or platforms[0]
    else:
        platform = platforms[platform_choice]

    devices = platform.get_devices()
    try:
        # sort devices based on their unique pci bus id
        devices = sorted(devices, key=lambda dev: dev.pci_bus_id_nv)
    except:  # noqa
        pass
    num_devices = len(devices)

    if device_choice is None:
        def try_to_get_local_rank():
            import os
            options = ("SLURM_LOCALID", "OMPI_COMM_WORLD_LOCAL_RANK",
                       "MV2_COMM_WORLD_LOCAL_RANK")
            for opt in options:
                x = os.getenv(opt)
                if x is not None:
                    return int(x)

            return 0

        device_choice = try_to_get_local_rank() % num_devices

    return cl.Context([devices[device_choice]])


class DisableLogging():  # silence logging warning
    def __enter__(self):
        import logging
        self.original_level = logging.getLogger().getEffectiveLevel()
        logging.disable(logging.CRITICAL)

    def __exit__(self, exception_type, exception_value, traceback):
        import logging
        logging.disable(self.original_level)


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
