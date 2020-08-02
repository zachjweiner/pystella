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
import pyopencl as cl
import pyopencl.array as cla
import pystella as ps

# set parameters
grid_shape = (64, 64, 64)
proc_shape = (1, 1, 1)
rank_shape = tuple(Ni // pi for Ni, pi in zip(grid_shape, proc_shape))
grid_size = np.product(grid_shape)

halo_shape = 2  # will be interpreted as (2, 2, 2)
pencil_shape = tuple(ni + 2 * halo_shape for ni in rank_shape)

box_dim = (5, 5, 5)
volume = np.product(box_dim)
dx = tuple(Li / Ni for Li, Ni in zip(box_dim, grid_shape))
dk = tuple(2 * np.pi / Li for Li in box_dim)
kappa = 1/10
dt = kappa * min(dx)

dtype = np.float64
nscalars = 2
mpl = 1  # change to np.sqrt(8 * np.pi) for reduced Planck mass units
mphi = 1.20e-6 * mpl
mchi = 0.
gsq = 2.5e-7
sigma = 0.
lambda4 = 0.
f0 = [.193 * mpl, 0]  # units of mpl
df0 = [-.142231 * mpl, 0]  # units of mpl
end_time = 1
end_scale_factor = 20
Stepper = ps.LowStorageRK54
gravitational_waves = True  # whether to simulate gravitational waves

ctx = ps.choose_device_and_make_context()
queue = cl.CommandQueue(ctx)

decomp = ps.DomainDecomposition(proc_shape, halo_shape, rank_shape)
fft = ps.DFT(decomp, ctx, queue, grid_shape, dtype)
if halo_shape == 0:
    derivs = ps.SpectralCollocator(fft, dk)
else:
    derivs = ps.FiniteDifferencer(decomp, halo_shape, dx, rank_shape=rank_shape)


def potential(f):
    phi, chi = f[0], f[1]
    unscaled = (mphi**2 / 2 * phi**2
                + mchi**2 / 2 * chi**2
                + gsq / 2 * phi**2 * chi**2
                + sigma / 2 * phi * chi**2
                + lambda4 / 4 * chi**4)
    return unscaled / mphi**2


scalar_sector = ps.ScalarSector(nscalars, potential=potential)
sectors = [scalar_sector]
if gravitational_waves:
    gw_sector = ps.TensorPerturbationSector([scalar_sector])
    sectors += [gw_sector]

stepper = Stepper(sectors, halo_shape=halo_shape, rank_shape=rank_shape, dt=dt)

# create energy computation function
from pystella.sectors import get_rho_and_p
reduce_energy = ps.Reduction(decomp, scalar_sector, halo_shape=halo_shape,
                             callback=get_rho_and_p,
                             rank_shape=rank_shape, grid_size=grid_size)


def compute_energy(f, dfdt, lap_f, dfdx, a):
    if gravitational_waves:
        derivs(queue, fx=f, lap=lap_f, grd=dfdx)
    else:
        derivs(queue, fx=f, lap=lap_f)

    return reduce_energy(queue, f=f, dfdt=dfdt, lap_f=lap_f, a=np.array(a))


# create output function
if decomp.rank == 0:
    from pystella.output import OutputFile
    out = OutputFile(ctx=ctx, runfile=__file__)
else:
    out = None
statistics = ps.FieldStatistics(decomp, halo_shape, rank_shape=rank_shape,
                                grid_size=grid_size)
spectra = ps.PowerSpectra(decomp, fft, dk, volume)
projector = ps.Projector(fft, halo_shape, dk, dx)
hist = ps.FieldHistogrammer(decomp, 1000, rank_shape, dtype)

a_sq_rho = (3 * mpl**2 * ps.Field('hubble', indices=[])**2 / 8 / np.pi)
rho_dict = {ps.Field('rho'): scalar_sector.stress_tensor(0, 0) / a_sq_rho}
compute_rho = ps.ElementWiseMap(rho_dict, halo_shape=halo_shape,
                                rank_shape=rank_shape)


def output(step_count, t, energy, expand,
           f, dfdt, lap_f, dfdx, hij, dhijdt, lap_hij):
    if step_count % 4 == 0:
        f_stats = statistics(f)

        if decomp.rank == 0:
            out.output('energy', t=t, a=expand.a[0],
                       adot=expand.adot[0]/expand.a[0],
                       hubble=expand.hubble[0]/expand.a[0],
                       **energy,
                       eos=energy['pressure']/energy['total'],
                       constraint=expand.constraint(energy['total'])
                       )

            out.output('statistics/f', t=t, a=expand.a[0], **f_stats)

    if expand.a[0] / output.a_last_spec >= 1.05:
        output.a_last_spec = expand.a[0]

        if not gravitational_waves:
            derivs(queue, fx=f, grd=dfdx)

        tmp = cla.empty(queue, shape=rank_shape, dtype=dtype)
        compute_rho(queue, a=expand.a, hubble=expand.hubble, rho=tmp,
                    f=f, dfdt=dfdt, dfdx=dfdx)
        rho_hist = hist(tmp)

        spec_out = dict(scalar=spectra(f), rho=spectra(tmp))

        if gravitational_waves:
            Hnow = expand.hubble
            spec_out['gw_transfer'] = 4.e-5 / 100**(1/3)
            a = expand.a[0]
            spec_out['df'] = (spectra.bin_width * mphi
                              * 6.e10 / np.sqrt(mphi * a * Hnow))
            spec_out['gw'] = spectra.gw(dhijdt, projector, Hnow)

        if decomp.rank == 0:
            out.output('rho_histogram', t=t, a=expand.a[0], **rho_hist)
            out.output('spectra', t=t, a=expand.a[0], **spec_out)


output.a_last_spec = .1  # to ensure spectra on the first slice

print('Initializing fields')

# create cl arrays
f = cla.empty(queue, (nscalars,)+pencil_shape, dtype)
dfdt = cla.empty(queue, (nscalars,)+pencil_shape, dtype)
dfdx = cla.empty(queue, (nscalars, 3,)+rank_shape, dtype)
lap_f = cla.empty(queue, (nscalars,)+rank_shape, dtype)

if gravitational_waves:
    hij = cla.empty(queue, (6,)+pencil_shape, dtype)
    dhijdt = cla.empty(queue, (6,)+pencil_shape, dtype)
    lap_hij = cla.empty(queue, (6,)+rank_shape, dtype)
else:
    hij, dhijdt, lap_hij = None, None, None

# set field means
for i in range(nscalars):
    f[i] = f0[i]
    dfdt[i] = df0[i]

# compute energy of background fields and initialize expansion
energy = compute_energy(f, dfdt, lap_f, dfdx, 1.)
expand = ps.Expansion(energy['total'], Stepper, mpl=mpl)

# compute hubble correction to scalar field effective mass
addot = expand.addot_friedmann_2(expand.a, energy['total'], energy['pressure'])
hubbleCorrection = - addot / expand.a

# effective masses of scalar fields
from pymbolic import var
from pymbolic.mapper.evaluator import evaluate_kw
fields = [var('f0')[i] for i in range(nscalars)]
d2Vd2f = [ps.diff(potential(fields), field, field) for field in fields]
eff_mass = [evaluate_kw(x, f0=f0) + hubbleCorrection for x in d2Vd2f]

modes = ps.RayleighGenerator(ctx, fft, dk, volume, seed=49279*(decomp.rank+1))

for fld in range(nscalars):
    modes.init_WKB_fields(f[fld], dfdt[fld], norm=mphi**2,
                          omega_k=lambda k: np.sqrt(k**2 + eff_mass[fld]),
                          hubble=expand.hubble[0])

for i in range(nscalars):
    f[i] += f0[i]
    dfdt[i] += df0[i]

# re-initialize energy and expansion
energy = compute_energy(f, dfdt, lap_f, dfdx, expand.a[0])
expand = ps.Expansion(energy['total'], Stepper, mpl=mpl)

# output first slice
output(0, 0., energy, expand, f=f, dfdt=dfdt, lap_f=lap_f, dfdx=dfdx,
       hij=hij, dhijdt=dhijdt, lap_hij=lap_hij)

# evolution
t = 0.
step_count = 0

if decomp.rank == 0:
    print('Time evolution beginning')
    print('time', 'scale factor', 'ms/step\t', 'steps/second', sep='\t\t')

from time import time
start = time()
last_out = time()

while t < end_time and expand.a[0] < end_scale_factor:
    for s in range(stepper.num_stages):
        stepper(s, queue=queue, a=expand.a, hubble=expand.hubble,
                f=f, dfdt=dfdt, dfdx=dfdx, lap_f=lap_f,
                hij=hij, dhijdt=dhijdt, lap_hij=lap_hij, filter_args=True)
        expand.step(s, energy['total'], energy['pressure'], dt)
        energy = compute_energy(f, dfdt, lap_f, dfdx, expand.a)
        if gravitational_waves:
            derivs(queue, fx=hij, lap=lap_hij)

    t += dt
    step_count += 1
    output(step_count, t, energy, expand, f=f, dfdt=dfdt, lap_f=lap_f, dfdx=dfdx,
           hij=hij, dhijdt=dhijdt, lap_hij=lap_hij)
    if time() - last_out > 30 and decomp.rank == 0:
        last_out = time()
        ms_per_step = (last_out - start) * 1e3 / step_count
        print(t, expand.a[0], ms_per_step, 1e3/ms_per_step, sep='\t')

if decomp.rank == 0:
    print('Simulation complete')
