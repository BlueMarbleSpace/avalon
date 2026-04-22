# AVALON

**AVALON** (Albedo-feedback Variable Axial-tilt Latitudinal Outgoing-Net EBM) is a 1D Budyko-Sellers energy balance model built for the [FILLET intercomparison project](https://doi.org/10.3847/PSJ/ae1c3c). It solves the latitude-dependent energy balance equation:

```
C ∂T/∂t = Q(x,t)·(1−α(T)) − OLR(T, CO₂) + D·∂/∂x[(1−x²)·∂T/∂x]
```

where `x = sin(lat)`, `Q` is solar insolation, `α` is surface albedo, and the last term is meridional heat diffusion. Output is fully compliant with the FILLET v1.1 format.

## Requirements

- **Julia** (no external packages — stdlib only: `LinearAlgebra`, `Statistics`, `Printf`)
- **Python 3** with `numpy` and `matplotlib` (for plotting only)

## Quick start

```bash
julia avalon.jl benchmark1          # Tuned pre-industrial Earth (Tglob ≈ 288 K)
julia avalon.jl benchmark2          # FILLET default parameters, ε = 23.5°
julia avalon.jl benchmark3          # FILLET default parameters, ε = 60°

julia avalon.jl exp1                # Warm-start instellation × obliquity sweep
julia avalon.jl exp3                # Instellation bifurcation (hysteresis diagram)
julia avalon.jl exp4                # CO₂ bifurcation

julia avalon.jl run obliquity=45 CO2=1000 seasonal=true out=mycase
julia avalon.jl help                # Full CLI documentation
```

## Physics

**Solar forcing** — Annual-mean or time-varying (seasonal) insolation using the Berger (1978) obliquity formula.

**Albedo** — Per-band land/ocean/ice blending:
- Ice (T < −10 °C): α = 0.60
- Otherwise: α = f<sub>land</sub> × α<sub>land</sub> + (1 − f<sub>land</sub>) × α<sub>ocean</sub>

**OLR** — Linear in temperature with a logarithmic CO₂ correction (Myhre et al. 1998):
OLR = A + B·T − F<sub>CO₂</sub>·ln(CO₂/CO₂<sub>ref</sub>)

**Heat capacity** — Per-band land/ocean blending:
C<sub>eff</sub> = f<sub>land</sub> × C<sub>land</sub> + (1 − f<sub>land</sub>) × C<sub>ocean</sub>

**Time stepping** — IMEX scheme: diffusion and OLR damping treated implicitly (solver matrix factored once per run), albedo feedback treated explicitly. Timestep ≈ 1 month; convergence criterion ΔT < 10⁻⁴ K yr⁻¹.

## FILLET benchmarks and experiments

All benchmarks and experiments use FILLET v1.1 Table 4 defaults unless noted. Seasonal mode is enabled for all runs.

| Command | Description | Output |
|---------|-------------|--------|
| `benchmark1` | Tuned Earth: D=0.52, α_ocean=0.2689, C_ocean=2×10⁸, latitude-dependent land fraction | `ben1/` |
| `benchmark2` | FILLET defaults, ε=23.5° | `ben2/` |
| `benchmark3` | FILLET defaults, ε=60° | `ben3/` |
| `exp1` | Warm-start instellation sweep (0.80–1.25 S⊕ × ε=0–90°) | `exp1/` |
| `exp2` | Cold-start instellation sweep (1.05–1.50 S⊕ × ε=0–90°) | `exp2/` |
| `exp1a` | Warm-start semi-major axis sweep (0.875–1.10 au × ε=0–90°) | `exp1a/` |
| `exp2a` | Cold-start semi-major axis sweep (0.80–0.975 au × ε=0–90°) | `exp2a/` |
| `exp3` | Instellation bifurcation diagram (0.8–1.5 S⊕, hysteresis) | `exp3/` |
| `exp4` | CO₂ bifurcation diagram (1–100,000 ppm) | `exp4/` |

**FILLET Table 4 defaults** (benchmarks 2/3 and all experiments):

| Parameter | Value |
|-----------|-------|
| α_land / α_ocean / α_ice | 0.30 / 0.20 / 0.60 |
| C_land / C_ocean / C_ice | 1×10⁷ / 4×10⁸ / 1×10⁷ J m⁻² K⁻¹ |
| D | 0.50 W m⁻² K⁻¹ |
| Land fraction | 0.25 uniform |

## Output format

FILLET `.dat` files are space-separated with `#`-commented headers.

Single-case runs (benchmarks, `run`) produce:
```
{tag}/lat_output_AVALON_{tag}.dat      # per-latitude: Lat Tsurf Asurf ATOA OLR Fland
{tag}/global_output_AVALON_{tag}.dat   # scalar diagnostics (FILLET v1.1 column format)
{tag}/{tag}_seasonal.csv               # monthly temperature snapshots (seasonal mode)
```

Sweep experiments (exp1/2/1a/2a) produce one lat file per simulation plus a global summary:
```
{tag}/lat_output_AVALON_{tag}_{case}.dat   # one per (obliquity, S₀) pair
{tag}/global_output_AVALON_{tag}.dat       # all cases in one table
```

Bifurcation experiments (exp3/4) produce the same layout, with a `Branch` column (cooling/warming) appended to the global summary.

## Plotting

```bash
python3 plot.py <tag>                # Annual-mean panels (temperature, albedo, energy balance, land fraction)
python3 plot.py <tag> seasonal       # Hovmöller temperature plot + seasonal amplitude
python3 plot.py <tag> sweep          # Obliquity × instellation phase diagram (exp1/2/1a/2a)
python3 plot.py <tag> bifurcation    # Hysteresis diagram with cooling/warming branches (exp3/4)
```

Plots are written to `{tag}/{tag}.png` / `.pdf` (benchmarks) or `{tag}/{tag}_{mode}.png` / `.pdf` (sweep/bifurcation/seasonal).

## Custom runs

Any model parameter can be overridden from the command line:

```bash
julia avalon.jl run obliquity=60 CO2=1000 seasonal=true out=highco2_obl60
julia avalon.jl run au=0.9 obliquity=45 out=innerhz
julia avalon.jl run S0=1200 alpha_ocean=0.28 D=0.44
```

Run `julia avalon.jl help` for the full list of parameters.

## FILLET intercomparison

AVALON was developed for the FILLET (Framework for Intercomparison of Low-complexity Latitudinal Energy-balance models and their Temperatures) project:

- FILLET v1.0: [arXiv:2511.11957](https://arxiv.org/abs/2511.11957)
- FILLET v1.1: [doi:10.3847/PSJ/ae1c3c](https://doi.org/10.3847/PSJ/ae1c3c)

**Known limitation:** AVALON uses a single temperature per latitudinal band (land and ocean are thermally blended), so it cannot produce a stable ice-belt state. This is a structural feature, not a bug; other FILLET models with separate land/ocean columns may find ice-belt solutions at high obliquity on the cold branch.

## Authors

Jacob Haqq-Misra — [jacob@bmsis.org](mailto:jacob@bmsis.org)
