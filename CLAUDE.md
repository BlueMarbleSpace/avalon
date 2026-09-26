# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

AVALON (Albedo-feedback Variable Axial-tilt Latitudinal Outgoing-Net EBM) is a 1D Budyko-Sellers energy balance model. It solves:

```
C(T) ∂T/∂t = Q(x,t)·(1−α(T)) − OLR(T, CO₂) + D·∂/∂x[(1−x²)·∂T/∂x]
```

where `x = sin(lat)`. Built for the FILLET intercomparison project (v1.0: doi:10.3847/PSJ/acba05; v1.1: doi:10.3847/PSJ/ae1c3c). Output format is FILLET v1.1 compliant (8-column ice line format), and `export` produces the `projectcuisines/fillet` Results/ layout.

**No external Julia dependencies** — only `LinearAlgebra`, `Statistics`, `Printf` from stdlib.

## Running the model

```bash
# Named experiments
julia avalon.jl benchmark2          # FILLET Benchmark 2
julia avalon.jl exp3                # instellation hysteresis (Benchmark 2 configuration)
julia avalon.jl exp3 base=ben1      # same, Benchmark 1 configuration → experiments/exp3_ben1/
julia avalon.jl run S0=1200 obliquity=45 seasonal=true out=mycase
julia avalon.jl export              # experiments/ → experiments/fillet_submission/avalon/
julia avalon.jl tune_ben1           # bisection on α_ocean for Tglob = 288 K
julia experiments/regen_all.jl      # everything, one Julia session (~3 min)

# Visualize output (requires numpy, matplotlib)
python3 plot.py ben2              # annual-mean panels (2×2 with land fraction, or 1×3)
python3 plot.py ben1 seasonal     # Hovmöller (day of orbit) + seasonal amplitude panel
python3 plot.py exp1 sweep        # obliquity × instellation phase diagram
python3 plot.py exp3 bifurcation  # hysteresis diagram (warm/cold branches); also exp4, exp3_ben1, exp4_ben1
```

All named commands: `benchmark1`, `benchmark2`, `benchmark3`, `exp1`, `exp2`, `exp1a`, `exp2a`, `exp3`, `exp4`, `run`, `export`, `tune_ben1`. Run `julia avalon.jl help` for full CLI docs.

## Architecture

Everything lives in `avalon.jl` (~1300 lines). Key layers from bottom to top:

1. **`Params` struct** (line ~47) — all physical/numerical constants; `with_params(p; k=v)` copies with overrides. Constants `T_WARM = 30`, `T_COLD = −50` (°C) define warm/cold starts; `α_OCEAN_BEN1` (line ~939) is the tuned Benchmark 1 ocean albedo.

2. **Grid helpers** (line ~123): `make_grid()` (cell-centered equal-area sin-lat, no node on the poles), `lat_deg()`, `earth_land_fraction(x)` (Ben1 only), `land_fraction_vec(p)`.

3. **Physics** (line ~159): `insolation_instant()` (Berger 1978 daily mean), `insolation_table()` (cell- and step-averaged n × steps_per_orbit table; orbit starts at NH spring equinox), `insolation()` (its orbit mean), `ice_fraction()` (ramp of width `ΔT_ice` about `T_ice`; 0 → step), `albedo()`, `heat_capacity()` (ice fraction blends in `C_ice`), `A_eff()` (Myhre CO₂ forcing).

4. **Solver** (line ~241): `build_diffusion_matrix()` (finite-volume tridiagonal), `Stepper` (work arrays), `thomas!()` (O(n) tridiagonal solve), `imex_step!()` (implicit OLR slope + diffusion, explicit albedo and heat capacity; matrix diagonal rebuilt every step).

5. **Public API** (line ~338): `run_ebm()` integrates orbit by orbit; converged when max |T(t) − T(t−kP)| over every step of an orbit < `tol` for some lag k = 1…`MAX_PERIOD` (4); returns orbits, dT, converged, `period` (k) and the orbit-resolved state averaged over the k-orbit cycle. `equilibrium()` returns orbit means (T, α, olr), `T_final` for continuation, and `asym`.

6. **Diagnostics** (line ~451): `global_mean()` (plain average = exact area mean on this grid), `hemispheric_asymmetry()`, `ice_segments()` (interpolated ice intervals), `ice_edges()` (FILLET v1.1 convention; cap reported when a belt coexists, `multiple` flag), `ice_edge_NH()`, `energy_budget()`.

7. **FILLET I/O** (line ~574): `fillet_profile()`, `fillet_global()`, `global_row()` (fixed precision), header builders (`config_notes`, `ice_line_notes`, `run_notes`, `lat_header`, `global_header`), `write_lat_file()`, `write_seasonal_csv()`, `log_case!()`, `write_fillet_output()`.

8. **Experiment runners** (line ~775): `run_fillet_sweep()` (instellation outermost; optional per-case `periods`), `run_fillet_sweep_au()` (S = S⊕/a², P = 365 d·a^1.5), `hysteresis_sweep(vals, :S0|:CO2)` (warm branch = warm start sweeping down, cold branch = cold start sweeping up, continuation), thin wrappers `bifurcation_diagram()` / `co2_bifurcation()`.

9. **Ben1 + export** (line ~935): `ben1_params()`, `tune_ben1()`, `export_fillet()`.

10. **CLI** (line ~1049): `run_custom()` parses `key=value` (aliases `alpha_*`, `dT_ice`; special keys `out`, `au`, `T0`), `run_cli()` dispatches named commands (`base=` option for exp3/exp4), `HELP_TEXT`.

## Benchmark 1 vs Ben2+

**Benchmark 1** uses non-FILLET parameters tuned to reproduce pre-industrial Earth: `D=0.52`, `α_ocean=α_OCEAN_BEN1` (0.25676), `C_ocean=2e8` (50 m mixed layer), `land_fraction=earth_land_fraction(x)`. Gives Tglob=288.0 K, NH ice edge 53°N (0 °C annual-mean convention; the −10 °C Budyko convention would give ~70°). Re-tune with `tune_ben1` whenever the numerics change, then update the constant.

**Benchmarks 2/3 and all experiments** use FILLET Table 4 defaults: uniform `land_fraction=0.25`, `C_ocean=4e8`, `C_ice=1e7`, `D=0.50`. Do not change these for FILLET submissions. Exp 3/4 default to the Benchmark 2 configuration (what the other codes filed); protocol v1.0 says "taking Benchmark 1", available via `base=ben1`.

## Output format

FILLET `.dat` files are space-separated with `#`-commented headers; every header records configuration, initial state, convergence (orbits, final ΔT) and N–S asymmetry. `plot.py` reads these via `read_dat()` and optionally reads `{tag}_seasonal.csv` (first column = day of orbit, one row per step). All output goes under `experiments/`; plots go to `experiments/{tag}/{tag}.png` and `.pdf`.

Single-case runs (benchmarks, `run`) write two files per tag plus the seasonal CSV:
- `experiments/{tag}/lat_output_AVALON_{tag}.dat` — per-latitude fields (`Lat Tsurf Asurf ATOA OLR Fland`)
- `experiments/{tag}/global_output_AVALON_{tag}.dat` — scalar diagnostics

Sweeps (exp1/exp2/exp1a/exp2a) and hysteresis runs (exp3/exp4) write a per-case lat file, one global summary and `convergence.log`:
- `experiments/{tag}/lat_output_AVALON_{tag}_{case}.dat`
- `experiments/{tag}/global_output_AVALON_{tag}.dat` — exp3/exp4 append a `Branch` column (`warm`/`cold`)
- `experiments/{tag}/convergence.log` — orbits, converged?, cycle period, ΔT, asymmetry, ice segments per case

`export` writes `experiments/fillet_submission/avalon/` in the fillet repo layout (ben*/case_0/lat_output.dat without Fland, exp*/global_output.dat, exp3/exp4 split into `_warm`/`_cold` with cases renumbered).

## Key numerical choices

- **Insolation table**: forcing averaged over each step and each cell (8 sub-points); orbit mean = exact annual mean, area mean = S₀/4 at every obliquity.
- **IMEX time stepping**: implicit for linear terms (OLR damping + diffusion), explicit for albedo and heat capacity; Thomas solve every step so `C` can follow the ice state. dt = period/366 (≈ 1 day). `steps_per_orbit` must stay **even**: an odd count samples the seasonal forcing at different phases in the two hemispheres (v1.1's 365 steps gave N–S asymmetries up to 0.4 K and 1.3° in ice edges); with 366 every symmetric case is symmetric to < 1e−3 K.
- **Ice transition** `ΔT_ice = 1 K` (ramp): a pure step (`dT_ice=0`) gives interannual chatter that never meets the tolerance and initial-state-dependent, N–S asymmetric Benchmark 2 states; the ramp changes Tglob by < 0.05 K.
- **Convergence**: same-phase comparison at every step of the orbit against 1–4 orbits earlier (period-2/4 cycles are real at high obliquity and in belt states, and are averaged over the cycle), tol 1e−4 K; `max_orbits = 500`; non-convergence is flagged, never silent.
- **Time-step sensitivity**: marginal seasonal-ice states depend on dt (730/1460 steps flip the two cold-start belts to snowballs and one cold-start case at S = 1.27, ε = 20° to ice-free; Exp 3/4 thresholds do not move; Ben2 −0.03 K, Ben1 −0.12 K at 1460). Documented in README *Known limitations*.
- **Seasonal mode** runs to a limit cycle and reports means over the final orbit; **annual-mean mode** uses the table's orbit mean and runs to a fixed point.

## Docs

`docs/` is local-only and gitignored (the FILLET code-comparison report, letter drafts, notes, comparison figures). Nothing in it is part of the repository or of a release; do not `git add -f` anything from it.
