# AVALON

![AVALON Logo](avalon_logo.svg)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22978034.svg)](https://doi.org/10.5281/zenodo.22978034)

**AVALON** (Albedo-feedback Variable Axial-tilt Latitudinal Outgoing-Net EBM) is a 1D Budyko-Sellers energy balance model built for the [FILLET intercomparison project](https://doi.org/10.3847/PSJ/ae1c3c). It solves the latitude-dependent energy balance equation:

```
C(T) ∂T/∂t = Q(x,t)·(1−α(T)) − OLR(T, CO₂) + D·∂/∂x[(1−x²)·∂T/∂x]
```

where `x = sin(lat)`, `Q` is solar insolation, `α` is surface albedo, `C` is the heat capacity, and the last term is meridional heat diffusion. Output follows the FILLET v1.1 format, and `julia avalon.jl export` assembles the layout required by the [FILLET results archive](https://github.com/projectcuisines/fillet).

## Requirements

- **Julia** (no external packages — stdlib only: `LinearAlgebra`, `Statistics`, `Printf`; tested with Julia 1.12)
- **Python 3** with `numpy` and `matplotlib` (for plotting only)

## Quick start

```bash
julia avalon.jl benchmark1          # Tuned pre-industrial Earth (Tglob = 288 K)
julia avalon.jl benchmark2          # FILLET default parameters, ε = 23.5°
julia avalon.jl benchmark3          # FILLET default parameters, ε = 60°

julia avalon.jl exp1                # Warm-start instellation × obliquity sweep
julia avalon.jl exp1a               # Same, varying the semi-major axis (and orbital period)
julia avalon.jl exp3                # Instellation hysteresis (warm- and cold-start branches)
julia avalon.jl exp4                # CO₂ hysteresis

julia avalon.jl export              # Assemble the FILLET Results/ layout from experiments/
julia avalon.jl run obliquity=45 CO2=1000 seasonal=true out=mycase
julia avalon.jl help                # Full CLI documentation
```

`julia experiments/regen_all.jl` regenerates every benchmark and experiment in one session (about three minutes).

## Physics

**Solar forcing** — Daily-mean insolation from the Berger (1978) declination formula for a circular orbit. In seasonal mode the forcing is tabulated once per run as the average over each time step and over each grid cell's extent in `sin(lat)`, so the model's orbit-mean forcing equals the exact annual mean at any step count and its area mean is S₀/4 at any obliquity. The orbit starts at the northern spring equinox. The orbital period is a parameter (`period_days`, default 365); Experiments 1a/2a set it from the semi-major axis as P = 365 d · a^1.5 together with S = S⊕/a².

**Ice** — A band's ice fraction `f` follows its instantaneous temperature: 0 above `T_ice + ΔT_ice/2`, 1 below `T_ice − ΔT_ice/2`, linear in between (defaults `T_ice = 0 °C`, `ΔT_ice = 1 K`; set `dT_ice=0` for a pure step). The ice fraction sets both the albedo and the heat capacity of the band:

- α = f · α<sub>ice</sub> + (1 − f) · [f<sub>land</sub> α<sub>land</sub> + (1 − f<sub>land</sub>) α<sub>ocean</sub>]
- C = f · C<sub>ice</sub> + (1 − f) · [f<sub>land</sub> C<sub>land</sub> + (1 − f<sub>land</sub>) C<sub>ocean</sub>]

The 1 K transition is a numerical regularization of the step: with a pure step on a discrete grid, freeze/thaw dates chatter from year to year and the equilibrium depends on the initial state (Benchmark 2 came out north–south asymmetric by 0.9 K). With the ramp every case converges, Benchmark 2 reaches the same state from 10, 15 and 30 °C starts, and global means move by less than 0.05 K relative to the step.

**OLR** — Linear in temperature with a logarithmic CO₂ correction (Myhre et al. 1998):
OLR = A + B·T − F<sub>CO₂</sub>·ln(CO₂/CO₂<sub>ref</sub>), with A = 210 W m⁻², B = 2.0 W m⁻² K⁻¹, F = 5.35 W m⁻² (T in °C).

**Diffusion** — Finite-volume operator with (1 − x²) weighting on a cell-centered equal-area `sin(lat)` grid (90 cells, no node on the poles), zero flux at the poles.

**Time stepping** — IMEX: diffusion and the OLR slope are implicit, albedo and heat capacity explicit. The tridiagonal system is solved every step with the Thomas algorithm, so the heat capacity is free to change with the ice state. Default `steps_per_orbit = 366`, i.e. 0.997 d for a 365-day orbit. The count is even on purpose: swapping the hemispheres is a half-orbit shift of `steps_per_orbit/2` steps, so with an odd count the two hemispheres sample the seasonal forcing at different phases. With 365 steps (v1.1) that sampling asymmetry left north–south asymmetries of up to 0.4 K in the annual-mean temperature of high-obliquity snowball cases and 1.3° between the two edges of an ice belt; with 366 every symmetric configuration is symmetric to below 10⁻³ K, and no climate state, hysteresis threshold or benchmark value changes at the quoted precision. A Benchmark 2 run takes about 0.1 s.

**Convergence** — Every step of an orbit is compared with the same phase 1, 2, 3 and 4 orbits earlier; the run converges when the largest such change over a whole orbit is below `tol` (10⁻⁴ K) for some lag, and that lag is the period of the cycle. Seasonally forced ice–albedo systems can settle into period-2 or period-4 (biennial) cycles, which a one-orbit test never accepts; AVALON reports the mean over the full cycle and says so in the header. Runs that hit `max_orbits` (500) are flagged `NOT CONVERGED` in the output headers, the terminal, and `convergence.log`. Every output header records the orbits run, the cycle period, the final change, and the north–south asymmetry of the annual-mean temperature.

**Ice line** — The latitude where the annual-mean temperature crosses `T_ice`, linearly interpolated between cell centers. Global outputs follow the FILLET v1.1 convention (ice-free: NMax = NMin = 90; snowball: NMax = 90, NMin = 0; polar cap: NMax = 90, NMin = edge; belt: poleward and equatorward edges). A cap coexisting with a belt is reported as the cap and listed in `convergence.log`. Land and sea columns are identical because AVALON has one temperature per band.

## FILLET benchmarks and experiments

All runs use FILLET Table 4 parameters (below) unless noted, in seasonal mode with 366 steps per orbit (about one day each). Warm starts are a uniform 30 °C, cold starts a uniform −50 °C (benchmarks use the warm start).

| Command | Description | Output |
|---------|-------------|--------|
| `benchmark1` | Tuned Earth: D = 0.52, α_ocean = 0.25676, C_ocean = 2×10⁸ (50 m mixed layer), latitude-dependent land fraction | `experiments/ben1/` |
| `benchmark2` | FILLET defaults, ε = 23.5° | `experiments/ben2/` |
| `benchmark3` | FILLET defaults, ε = 60° | `experiments/ben3/` |
| `exp1` | Warm-start instellation sweep (0.80–1.25 S⊕ × ε = 0–90°) | `experiments/exp1/` |
| `exp2` | Cold-start instellation sweep (1.05–1.50 S⊕ × ε = 0–90°) | `experiments/exp2/` |
| `exp1a` | Warm-start semi-major axis sweep (0.875–1.10 au × ε = 0–90°; S and period vary) | `experiments/exp1a/` |
| `exp2a` | Cold-start semi-major axis sweep (0.80–0.975 au × ε = 0–90°; S and period vary) | `experiments/exp2a/` |
| `exp3 [base=ben1]` | Instellation hysteresis, 0.8–1.5 S⊕, warm and cold continuation branches | `experiments/exp3/` (`exp3_ben1/`) |
| `exp4 [base=ben1]` | CO₂ hysteresis, 1–100,000 ppm, warm and cold continuation branches | `experiments/exp4/` (`exp4_ben1/`) |
| `export` | FILLET `Results/` layout assembled from the files above | `experiments/fillet_submission/avalon/` |
| `tune_ben1` | Bisection on the Benchmark 1 ocean albedo for Tglob = 288 K | prints α_ocean |

Experiments 3 and 4 use the Benchmark 2 configuration by default, which is what the other FILLET codes filed; Protocol v1.0 (§3.7–3.8) reads "taking Benchmark 1", and `base=ben1` runs that variant into a separate directory. In both hysteresis experiments each case starts from the final state of the previous case (continuation); the warm branch sweeps downward from a 30 °C start and the cold branch upward from a −50 °C start.

**Current benchmark results**

| Benchmark | Tglob | Ice edges (annual-mean 0 °C) | Orbits to converge |
|-----------|-------|------------------------------|--------------------|
| 1 (tuned) | 288.0 K | 53.0°N, 57.8°S | 43 |
| 2 (ε = 23.5°) | 299.4 K | 79.6°N, 79.6°S | 62 |
| 3 (ε = 60°) | 300.0 K | ice-free | 44 |

Under a −10 °C threshold (Budyko's convention) the Benchmark 1 edge would sit near 70°; the 0 °C convention places every FILLET model's Earth edge near 50–58° (FILLET code comparison, Sept 2026).

**FILLET Table 4 defaults** (benchmarks 2/3 and all experiments):

| Parameter | Value |
|-----------|-------|
| α_land / α_ocean / α_ice | 0.30 / 0.20 / 0.60 |
| C_land / C_ocean / C_ice | 1×10⁷ / 4×10⁸ / 1×10⁷ J m⁻² K⁻¹ |
| D | 0.50 W m⁻² K⁻¹ |
| Land fraction | 0.25 uniform |

## Output format

FILLET `.dat` files are space-separated with `#`-commented headers. Every header states the configuration (period, steps, albedos, heat capacities, OLR coefficients, ice transition, land fraction), the ice-line definition, the initial state, the convergence outcome and the hemispheric asymmetry.

All output is written under `experiments/`. Single-case runs (benchmarks, `run`) produce:
```
experiments/{tag}/lat_output_AVALON_{tag}.dat      # per-latitude: Lat Tsurf Asurf ATOA OLR Fland
experiments/{tag}/global_output_AVALON_{tag}.dat   # scalar diagnostics (FILLET v1.1 columns)
experiments/{tag}/{tag}_seasonal.csv               # final-orbit snapshots, one row per step (first column: day of orbit)
```

Sweeps (exp1/2/1a/2a) and hysteresis runs (exp3/4) produce one lat file per case, a global summary and a convergence log:
```
experiments/{tag}/lat_output_AVALON_{tag}_{case}.dat   # one per case
experiments/{tag}/global_output_AVALON_{tag}.dat       # all cases in one table (instellation outermost, obliquity innermost)
experiments/{tag}/convergence.log                      # per case: orbits, converged?, final ΔT, N–S asymmetry, ice segments
```

Hysteresis runs append a `Branch` column (`warm` / `cold`) to the global summary.

## FILLET submission

`julia avalon.jl export` writes the directory tree that `projectcuisines/fillet` requires (see its `Results/README.md`):

```
experiments/fillet_submission/avalon/
├── ben1/ ben2/ ben3/        global_output.dat + case_0/lat_output.dat  (template columns; Fland dropped)
├── exp1/ exp1a/ exp2/ exp2a/  global_output.dat
├── exp3_warm/ exp3_cold/    global_output.dat  (Branch column split off, cases renumbered from 0)
└── exp4_warm/ exp4_cold/    global_output.dat
```

Copy that `avalon/` directory over `Results/avalon/` in a fork of the FILLET repository and open a pull request.

## Plotting

```bash
python3 plot.py <tag>                # Annual-mean panels (temperature, albedo, energy balance, land fraction)
python3 plot.py <tag> seasonal       # Hovmöller temperature plot + seasonal amplitude
python3 plot.py <tag> sweep          # Obliquity × instellation phase diagram (exp1/2/1a/2a)
python3 plot.py <tag> bifurcation    # Hysteresis diagram, warm and cold branches (exp3/4, exp3_ben1/exp4_ben1)
```

Plots are written to `experiments/{tag}/{tag}.png` / `.pdf` (benchmarks) or `experiments/{tag}/{tag}_{mode}.png` / `.pdf` (sweep/bifurcation/seasonal).

## Custom runs

Any model parameter can be overridden from the command line:

```bash
julia avalon.jl run obliquity=60 CO2=1000 seasonal=true out=highco2_obl60
julia avalon.jl run au=0.9 obliquity=45 seasonal=true out=innerhz     # au sets S₀ and the orbital period
julia avalon.jl run S0=1200 alpha_ocean=0.28 D=0.44 dT_ice=0 T0=-50
```

Run `julia avalon.jl help` for the full list of parameters.

## Changes since the archived FILLET submission

The files in `Results/avalon/` of the FILLET repository come from AVALON v1.0 (April 2026); this is v1.2 (September 2026). Since then:

- **June 2026** — cell-centered grid (the archived grid had nodes on the poles and an unweighted mean, which made the global mean depend on obliquity); ice threshold 0 °C instead of −10 °C; Benchmark 1 re-tuned.
- **September 2026**, in response to the FILLET code comparison (source audit of the participating models, September 2026):
  - step- and cell-averaged insolation table (the 12-steps-per-year sampling had a 5 W m⁻² forcing error at the equator at 90° obliquity);
  - orbital period as a parameter, so Experiments 1a/2a differ from 1/2;
  - ice heat capacity (Table 4) applied to ice-covered bands; daily time steps; tridiagonal solve every step;
  - 1 K ice transition instead of a step (removes interannual chatter and the initial-state dependence of Benchmark 2);
  - convergence tested at every phase of the orbit, cap warnings, orbits run and hemispheric asymmetry in every header, `convergence.log` per sweep;
  - ice edges interpolated between cell centers; cap-plus-belt states no longer read as snowballs;
  - instellation outermost in the sweep case order, fixed-precision columns;
  - `export` command for the FILLET archive layout; `tune_ben1`; `exp3/exp4 base=ben1`.
- **v1.2 (September 2026)** — 366 steps per orbit instead of 365: an odd count samples the seasonal forcing at different phases in the two hemispheres (see *Time stepping*), which left north–south asymmetries of up to 0.4 K in v1.1's high-obliquity and belt cases; the even count removes them without changing any climate state, hysteresis threshold or benchmark value at the quoted precision. The Experiment 1a/2a global headers now state the per-case period range instead of the base 365 days. Time-step sensitivity of marginal cases documented under *Known limitations*.

Every benchmark and experiment was regenerated after these changes; the archived submission should be replaced with `julia avalon.jl export`.

## Known limitations

- One temperature per latitude band (land and ocean thermally blended), so land and sea ice lines coincide. Ice belts do occur on the cold-start branch at 50–60° obliquity near 1.05–1.08 S⊕, where the low ice heat capacity lets the summer pole thaw while the equator stays frozen; they are reported with the belt convention. Both belts are marginal states (next item).
- Time-step sensitivity of marginal cases. The ice fraction and heat capacity are explicit in time, and a polar band with the 10⁷ J m⁻² K⁻¹ ice heat capacity can move several kelvin per step through the 1 K transition, so states near a threshold depend on the step. Halving and quartering the step (730 and 1460 steps per orbit) leaves every Experiment 3/4 threshold and all but three of the 720 Experiment 1/1a/2/2a states unchanged: the two cold-start belts close into snowballs and one cold-start case (S = 1.27, 20° obliquity) deglaciates. Benchmark 2 moves by 0.03 K and Benchmark 1 by 0.12 K at 1460 steps. Set `steps_per_orbit` (keep it even) to test a case.
- Experiment 4's cold-start branch stays glaciated over the whole 1–100,000 ppm range: with a linear OLR and the Myhre CO₂ coefficient, a snowball at S = 1 needs about 74 W m⁻² of CO₂ forcing (~3×10⁸ ppm) to deglaciate. This is structural to the OLR parameterization, not a range problem.
- No zenith-angle dependence of the albedo and no atmospheric scattering, so the top-of-atmosphere albedo equals the surface albedo.

## Citation

Releases are archived on Zenodo. Cite the concept DOI [10.5281/zenodo.22978034](https://doi.org/10.5281/zenodo.22978034), which always resolves to the latest version (v1.2 is [10.5281/zenodo.22981638](https://doi.org/10.5281/zenodo.22981638)). Citation metadata is in `CITATION.cff`.

## Authors

Jacob Haqq-Misra — [jacob@bmsis.org](mailto:jacob@bmsis.org)
