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
from pystella.fourier.dft import DFT, gDFT, pDFT
from pystella.fourier.rayleigh import RayleighGenerator
from pystella.fourier.projectors import Projector
from pystella.fourier.spectra import PowerSpectra
from pystella.fourier.derivs import SpectralCollocator
from pystella.fourier.poisson import SpectralPoissonSolver


_r_to_c_dtype_map = {np.dtype("float32"): np.dtype("complex64"),
                     np.dtype("float64"): np.dtype("complex128")}

_c_to_r_dtype_map = {np.dtype("complex64"): np.dtype("float32"),
                     np.dtype("complex128"): np.dtype("float64")}


def get_real_dtype_with_matching_prec(dtype):
    return _c_to_r_dtype_map.get(np.dtype(dtype)) or dtype


def get_complex_dtype_with_matching_prec(dtype):
    return _r_to_c_dtype_map.get(np.dtype(dtype)) or dtype


__all__ = [
    "DFT",
    "gDFT",
    "pDFT",
    "RayleighGenerator",
    "Projector",
    "PowerSpectra",
    "SpectralCollocator",
    "SpectralPoissonSolver",
    "get_real_dtype_with_matching_prec",
    "get_complex_dtype_with_matching_prec",
]
