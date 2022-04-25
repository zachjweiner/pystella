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
from argparse import ArgumentParser

# create command line interface
parser = ArgumentParser()
parser.add_argument("--grid-shape", "-grid", type=int, nargs=3,
                    metavar=("Nx", "Ny", "Nz"), default=(128, 128, 128),
                    help="the number of gridpoints in each dimensions")
parser.add_argument("--proc-shape", "-proc", type=int, nargs=3,
                    metavar=("Npx", "Npy", "Npz"), default=(1, 1, 1),
                    help="the processor grid dimension")
parser.add_argument("--dtype", type=np.dtype, default=np.float64,
                    help="the simulation datatype")
parser.add_argument("--halo-shape", type=int, default=2, metavar="h",
                    help="the halo shape; determines stencil order")
parser.add_argument("--box-dim", "-box", type=float, nargs=3,
                    metavar=("Lx", "Ly", "Lz"), default=(5, 5, 5),
                    help="the box length in each dimension")
parser.add_argument("--kappa", type=float, default=1/10,
                    help="the timestep to grid spacing ratio")
parser.add_argument("--mpl", type=float, default=1,
                    help="the unreduced Planck mass in desired units")
parser.add_argument("--mphi", type=float, default=1.20e-6,
                    help="the rescaling mass; in units of --mpl")
parser.add_argument("--mchi", type=float, nargs="*", default=0.,
                    help="the mass(es) of coupled scalars")
parser.add_argument("--gsq", type=float, nargs="*", default=2.5e-7,
                    help="the 2-2 coupling of phi to other scalars")
parser.add_argument("--sigma", type=float, nargs="*", default=0.,
                    help="the trilinear coupling of phi to other scalars")
parser.add_argument("--lambda4", type=float, nargs="*", default=0.,
                    help="the quartic self-coupling of other scalars")
parser.add_argument("--end-time", "-end-t", type=float, default=20,
                    help="the (conformal) time at which to end the simulation")
parser.add_argument("--end-scale-factor", "-end-a", type=float, default=20,
                    help="the scale factor (relative to the end of inflation)"
                         " at which to end the simulation")
parser.add_argument("--gravitational-waves", "-gws", action="store_true",
                    help="pass to simulate gravitational waves")


def main():
    p = parser.parse_args()
    # process args
    p.grid_shape = tuple(p.grid_shape)
    p.grid_size = np.product(p.grid_shape)
    p.proc_shape = tuple(p.proc_shape)
    p.rank_shape = tuple(Ni // pi for Ni, pi in zip(p.grid_shape, p.proc_shape))
    p.pencil_shape = tuple(ni + 2 * p.halo_shape for ni in p.rank_shape)
    p.box_dim = tuple(p.box_dim)
    p.volume = np.product(p.box_dim)
    p.dx = tuple(Li / Ni for Li, Ni in zip(p.box_dim, p.grid_shape))
    p.dk = tuple(2 * np.pi / Li for Li in p.box_dim)
    dt = p.kappa * min(p.dx)

    p.nscalars = 2
    f0 = [.193 * p.mpl, 0]  # units of mpl
    df0 = [-.142231 * p.mpl, 0]  # units of mpl
    Stepper = ps.LowStorageRK54

    ctx = ps.choose_device_and_make_context()
    queue = cl.CommandQueue(ctx)

    decomp = ps.DomainDecomposition(p.proc_shape, p.halo_shape, p.rank_shape)
    fft = ps.DFT(decomp, ctx, queue, p.grid_shape, p.dtype)
    if p.halo_shape == 0:
        derivs = ps.SpectralCollocator(fft, p.dk)
    else:
        derivs = ps.FiniteDifferencer(
            decomp, p.halo_shape, p.dx, rank_shape=p.rank_shape)

    def potential(f):
        phi, chi = f[0], f[1]
        unscaled = (p.mphi**2 / 2 * phi**2
                    + p.mchi**2 / 2 * chi**2
                    + p.gsq / 2 * phi**2 * chi**2
                    + p.sigma / 2 * phi * chi**2
                    + p.lambda4 / 4 * chi**4)
        return unscaled / p.mphi**2

    scalar_sector = ps.ScalarSector(p.nscalars, potential=potential)
    sectors = [scalar_sector]
    if p.gravitational_waves:
        gw_sector = ps.TensorPerturbationSector([scalar_sector])
        sectors += [gw_sector]

    stepper = Stepper(
        sectors, halo_shape=p.halo_shape, rank_shape=p.rank_shape, dt=dt)

    # create energy computation function
    from pystella.sectors import get_rho_and_p
    reduce_energy = ps.Reduction(
        decomp, scalar_sector, halo_shape=p.halo_shape,
        callback=get_rho_and_p, rank_shape=p.rank_shape, grid_size=p.grid_size)

    def compute_energy(f, dfdt, lap_f, dfdx, a):
        if p.gravitational_waves:
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
    statistics = ps.FieldStatistics(
        decomp, p.halo_shape, rank_shape=p.rank_shape, grid_size=p.grid_size)
    spectra = ps.PowerSpectra(decomp, fft, p.dk, p.volume)
    projector = ps.Projector(fft, p.halo_shape, p.dk, p.dx)
    hist = ps.FieldHistogrammer(decomp, 1000, p.dtype, rank_shape=p.rank_shape)

    a_sq_rho = (3 * p.mpl**2 * ps.Field("hubble", indices=[])**2 / 8 / np.pi)
    rho_dict = {ps.Field("rho"): scalar_sector.stress_tensor(0, 0) / a_sq_rho}
    compute_rho = ps.ElementWiseMap(
        rho_dict, halo_shape=p.halo_shape, rank_shape=p.rank_shape)

    def output(step_count, t, energy, expand,
               f, dfdt, lap_f, dfdx, hij, dhijdt, lap_hij):
        if step_count % 4 == 0:
            f_stats = statistics(f)

            if decomp.rank == 0:
                out.output(
                    "energy", t=t, a=expand.a[0],
                    adot=expand.adot[0]/expand.a[0],
                    hubble=expand.hubble[0]/expand.a[0],
                    **energy,
                    eos=energy["pressure"]/energy["total"],
                    constraint=expand.constraint(energy["total"])
                )

                out.output("statistics/f", t=t, a=expand.a[0], **f_stats)

        if expand.a[0] / output.a_last_spec >= 1.05:
            output.a_last_spec = expand.a[0]

            if not p.gravitational_waves:
                derivs(queue, fx=f, grd=dfdx)

            tmp = cla.empty(queue, shape=p.rank_shape, dtype=p.dtype)
            compute_rho(queue, a=expand.a, hubble=expand.hubble, rho=tmp,
                        f=f, dfdt=dfdt, dfdx=dfdx)
            rho_hist = hist(tmp)

            spec_out = {"scalar": spectra(f), "rho": spectra(tmp)}

            if p.gravitational_waves:
                Hnow = expand.hubble
                spec_out["gw_transfer"] = 4.e-5 / 100**(1/3)
                a = expand.a[0]
                spec_out["df"] = (
                    spectra.bin_width * p.mphi * 6.e10 / np.sqrt(p.mphi * a * Hnow))
                spec_out["gw"] = spectra.gw(dhijdt, projector, Hnow)

            if decomp.rank == 0:
                out.output("rho_histogram", t=t, a=expand.a[0], **rho_hist)
                out.output("spectra", t=t, a=expand.a[0], **spec_out)

    output.a_last_spec = .1  # to ensure spectra on the first slice

    print("Initializing fields")

    # create cl arrays
    f = cla.empty(queue, (p.nscalars,)+p.pencil_shape, p.dtype)
    dfdt = cla.empty(queue, (p.nscalars,)+p.pencil_shape, p.dtype)
    dfdx = cla.empty(queue, (p.nscalars, 3,)+p.rank_shape, p.dtype)
    lap_f = cla.empty(queue, (p.nscalars,)+p.rank_shape, p.dtype)

    if p.gravitational_waves:
        hij = cla.empty(queue, (6,)+p.pencil_shape, p.dtype)
        dhijdt = cla.empty(queue, (6,)+p.pencil_shape, p.dtype)
        lap_hij = cla.empty(queue, (6,)+p.rank_shape, p.dtype)
    else:
        hij, dhijdt, lap_hij = None, None, None

    # set field means
    for i in range(p.nscalars):
        f[i] = f0[i]
        dfdt[i] = df0[i]

    # compute energy of background fields and initialize expansion
    energy = compute_energy(f, dfdt, lap_f, dfdx, 1.)
    expand = ps.Expansion(energy["total"], Stepper, mpl=p.mpl)

    # compute hubble correction to scalar field effective mass
    addot = expand.addot_friedmann_2(expand.a, energy["total"], energy["pressure"])
    hubbleCorrection = - addot / expand.a

    # effective masses of scalar fields
    from pymbolic import var
    from pymbolic.mapper.evaluator import evaluate_kw
    fields = [var("f0")[i] for i in range(p.nscalars)]
    d2Vd2f = [ps.diff(potential(fields), field, field) for field in fields]
    eff_mass = [evaluate_kw(x, f0=f0) + hubbleCorrection for x in d2Vd2f]

    modes = ps.RayleighGenerator(
        ctx, fft, p.dk, p.volume, seed=49279*(decomp.rank+1))

    for fld in range(p.nscalars):
        modes.init_WKB_fields(
            f[fld], dfdt[fld], norm=p.mphi**2,
            omega_k=lambda k: np.sqrt(k**2 + eff_mass[fld]), hubble=expand.hubble[0])

    for i in range(p.nscalars):
        f[i] += f0[i]
        dfdt[i] += df0[i]

    # re-initialize energy and expansion
    energy = compute_energy(f, dfdt, lap_f, dfdx, expand.a[0])
    expand = ps.Expansion(energy["total"], Stepper, mpl=p.mpl)

    t = 0.
    step_count = 0

    # output first slice
    output(t, step_count, energy, expand, f=f, dfdt=dfdt, lap_f=lap_f, dfdx=dfdx,
           hij=hij, dhijdt=dhijdt, lap_hij=lap_hij)

    if decomp.rank == 0:
        print("Time evolution beginning")
        print("time\t", "scale factor", "ms/step\t", "steps/second", sep="\t")

    from time import time
    start = time()
    last_out = time()

    while t < p.end_time and expand.a[0] < p.end_scale_factor:
        for s in range(stepper.num_stages):
            stepper(s, queue=queue, a=expand.a, hubble=expand.hubble,
                    f=f, dfdt=dfdt, dfdx=dfdx, lap_f=lap_f,
                    hij=hij, dhijdt=dhijdt, lap_hij=lap_hij, filter_args=True)
            expand.step(s, energy["total"], energy["pressure"], dt)
            energy = compute_energy(f, dfdt, lap_f, dfdx, expand.a)
            if p.gravitational_waves:
                derivs(queue, fx=hij, lap=lap_hij)

        t += dt
        step_count += 1
        output(step_count, t, energy, expand, f=f, dfdt=dfdt, lap_f=lap_f, dfdx=dfdx,
               hij=hij, dhijdt=dhijdt, lap_hij=lap_hij)
        if time() - last_out > 30 and decomp.rank == 0:
            last_out = time()
            ms_per_step = (last_out - start) * 1e3 / step_count
            print(f"{t:<15.3f}", f"{expand.a[0]:<15.3f}",
                  f"{ms_per_step:<15.3f}", f"{1e3 / ms_per_step:<15.3f}")

    if decomp.rank == 0:
        print("Simulation complete")


if __name__ == "__main__":
    main()
