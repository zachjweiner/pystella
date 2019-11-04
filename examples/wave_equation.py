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


import pyopencl as cl
import pyopencl.array as cla
import pyopencl.clrandom as clr
import pystella as ps

# set parameters
grid_shape = (128, 128, 128)
proc_shape = (1, 1, 1)
rank_shape = tuple(Ni // pi for Ni, pi in zip(grid_shape, proc_shape))
halo_shape = 1
dtype = 'float64'
dx = tuple(10 / Ni for Ni in grid_shape)
dt = min(dx) / 10

# create pyopencl context, queue, and halo-sharer
ctx = ps.choose_device_and_make_context()
queue = cl.CommandQueue(ctx)
decomp = ps.DomainDecomposition(proc_shape, halo_shape, rank_shape)

# initialize arrays with random data
f = clr.rand(queue, tuple(ni + 2 * halo_shape for ni in rank_shape), dtype)
dfdt = clr.rand(queue, tuple(ni + 2 * halo_shape for ni in rank_shape), dtype)
lap_f = cla.zeros(queue, rank_shape, dtype)

# define system of equations
f_ = ps.DynamicField('f', offset='h')  # don't overwrite f
rhs_dict = {
    f_: f_.dot,  # df/dt = \dot{f}
    f_.dot: f_.lap  # d\dot{f}/dt = \nabla^2 f
}

# create time-stepping and derivative-computing kernels
stepper = ps.LowStorageRK54(rhs_dict, dt=dt, halo_shape=halo_shape)
derivs = ps.FiniteDifferencer(decomp, halo_shape, dx)

# temporary array for low-storage integrator
k_tmp = cla.empty(queue, (stepper.num_unknowns,)+rank_shape, dtype)

t = 0.
# loop over time
while t < 10.:
    for s in range(stepper.num_stages):
        derivs(queue, fx=f, lap=lap_f)
        stepper(s, queue=queue, k_tmp=k_tmp, f=f, dfdt=dfdt, lap_f=lap_f)
    t += dt
