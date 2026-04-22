# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

AVALON (Albedo-feedback Variable Axial-tilt Latitudinal Outgoing-Net EBM) is a 1D Budyko-Sellers energy balance model. It solves:

```
C ∂T/∂t = Q(x,t)·(1−α(T)) − OLR(T, CO₂) + D·∂/∂x[(1−x²)·∂T/∂x]
```

where `x = sin(lat)`. Built for the FILLET intercomparison project (v1.0: arxiv.org/abs/2511.11957; v1.1: doi:10.3847/PSJ/ae1c3c). Output format is FILLET v1.1 compliant (8-column ice line format).

**No external Julia dependencies** — only `LinearAlgebra`, `Statistics`, `Printf` from stdlib.

## Running the model

```bash
# Named experiments
julia avalon.jl benchmark2          # FILLET Benchmark 2 (annual-mean + seasonal)
julia avalon.jl exp3                # Snowball Earth bifurcation diagram
julia avalon.jl run S0=1200 obliquity=45 seasonal=true out=mycase

# Visualize output (requires numpy, matplotlib)
python3 plot.py ben2              # annual-mean panels (2×2 with land fraction, or 1×3)
python3 plot.py ben1 seasonal     # Hovmöller + seasonal amplitude panel
python3 plot.py exp1 sweep        # obliquity × instellation phase diagram (climate states + Tglob)
python3 plot.py exp3 bifurcation  # hysteresis diagram (cooling/warming branches); also works for exp4
```

All named commands: `benchmark1`, `benchmark2`, `benchmark3`, `exp1`, `exp2`, `exp1a`, `exp2a`, `exp3`, `exp4`, `run`. Run `julia avalon.jl help` for full CLI docs.

## Architecture

Everything lives in `avalon.jl` (~1110 lines). Key layers from bottom to top:

1. **`Params` struct** (line ~41) — all physical/numerical constants in one place; modify here to change defaults.

2. **Grid helpers** (line ~118):
   - `make_grid()`, `lat_deg()` — sin-lat grid utilities
   - `earth_land_fraction(x)` — piecewise-linear Earth land fraction profile (~30% global mean, NH/SH asymmetric); used by Benchmark 1 only
   - `heat_capacity()` — computes per-band `C_eff = f_land × C_land + (1−f_land) × C_ocean`; default land fraction 0.25 uniform

3. **Physics functions** (line ~155):
   - `insolation()` / `insolation_instant()` — annual-mean and daily-mean solar forcing (Berger 1978 obliquity formula)
   - `albedo()` — ice (T < −10°C) → 0.60; otherwise `f_land × α_land + (1−f_land) × α_ocean` per band
   - `A_eff()` — OLR intercept with CO₂ forcing (Myhre et al. 1998 logarithmic)

4. **IMEX solver** (line ~200):
   - `build_diffusion_matrix()` — tridiagonal operator with (1−x²) spherical weighting
   - `build_solver()` — factors implicit part (OLR slope B + diffusion) once per run; albedo is explicit
   - `imex_step!()` — single timestep update

5. **Public API** (line ~260):
   - `run_ebm()` — integrates to convergence (annual-mean → fixed point; seasonal → limit cycle)
   - `equilibrium()` — wrapper returning `(T, α, olr, T_seasonal, years)`

6. **Diagnostics** (line ~360): `global_mean()`, `ice_edges()`, `energy_budget()`

7. **FILLET I/O** (line ~430): `write_fillet_output()` writes `lat_output_AVALON_{tag}.dat` and `global_output_AVALON_{tag}.dat`

8. **Experiment runners** (line ~550): `run_fillet_sweep()`, `bifurcation_diagram()`, `co2_bifurcation()`

9. **CLI** (line ~730): `run_cli()` dispatches named commands; `run_custom()` parses `key=value` args

## Benchmark 1 vs Ben2+

**Benchmark 1** uses non-FILLET parameters tuned to best reproduce pre-industrial Earth: `D=0.52`, `α_ocean=0.2689`, `C_ocean=2e8` (50 m mixed layer), `land_fraction=earth_land_fraction(x)`. Gives Tglob=288 K, NH ice edge ~73°N, seasonal amplitude ~14°C at mid-latitudes.

**Benchmarks 2/3 and all experiments** use FILLET Table 4 defaults: uniform `land_fraction=0.25`, `C_ocean=4e8`, `D=0.50`. Do not change these for FILLET submissions.

## Output format

FILLET `.dat` files are space-separated with `#`-commented headers. `plot.py` reads these via `read_dat()` and optionally reads `{tag}_seasonal.csv` for a seasonal temperature envelope. Plots go to `{tag}/{tag}.png` and `{tag}/{tag}.pdf`.

Single-case runs (benchmarks, `run`) write two files per tag:
- `{tag}/lat_output_AVALON_{tag}.dat` — per-latitude fields
- `{tag}/global_output_AVALON_{tag}.dat` — scalar diagnostics

Sweep experiments (exp1/exp2/exp1a/exp2a) write a per-case lat file for each (obliquity, S₀) pair plus one global summary:
- `{outdir}/lat_output_AVALON_{tag}_{case}.dat` — one per simulation
- `{outdir}/global_output_AVALON_{tag}.dat` — all cases in one table

## Key numerical choices

- **IMEX time stepping**: implicit for linear terms (OLR damping + diffusion), explicit for nonlinear albedo. The solver matrix is independent of Q and CO₂, so it's factored once per run.
- **dt** ≈ 1 month; convergence criterion: ΔT < 1e−4 K/yr.
- **Seasonal mode** runs to periodic equilibrium (limit cycle), then reports annual means.
