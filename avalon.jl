"""
AVALON — Albedo-feedback Variable Axial-tilt Latitudinal Outgoing-Net EBM

1D Energy Balance Climate Model (Budyko-Sellers type)

Latitude-dependent surface temperature T(x, t) governed by:

    C ∂T/∂t = Q(x,t)·(1−α(T)) − OLR(T, CO₂) + D·∂/∂x[(1−x²)·∂T/∂x]

where x = sin(φ) ∈ [−1, 1], φ = latitude.

Insolation Q(x, t):
  - Annual-mean mode (seasonal=false): Q precomputed once by integrating over
    all solar longitudes (Berger 1978). Runs to a fixed-point equilibrium.
  - Seasonal mode (seasonal=true): Q computed at each timestep from the
    instantaneous solar declination δ(t) = arcsin(sin(ε)·sin(λ(t))).
    Runs to a periodic (limit-cycle) equilibrium; outputs are annual means.

OLR(T, CO₂) = (A − F_CO₂·ln(CO₂/CO₂_ref)) + B·T

Time discretization: IMEX — implicit for linear terms (OLR slope + diffusion),
explicit for nonlinear albedo. Implicit system matrix M = (C/dt+B)I − L is
factored once and reused; it does not depend on Q, CO₂, or season.

References:
  Budyko (1969), Sellers (1969), North et al. (1981 Rev. Geophys.)
  Berger (1978, J. Atmos. Sci.) for obliquity-dependent insolation
  Myhre et al. (1998) for CO₂ forcing coefficient

FILLET intercomparison: doi:10.3847/PSJ/ae1c3c
"""

using LinearAlgebra
using Statistics
using Printf

# ============================================================
# Parameters
# ============================================================

Base.@kwdef struct Params
    # Grid
    n::Int             = 90

    # Solar forcing
    S0::Float64        = 1361.0     # solar constant [W m⁻²]

    # Obliquity
    obliquity::Float64 = 23.5       # axial tilt [degrees]

    # Outgoing longwave radiation: OLR = A_eff + B·T
    A::Float64         = 210.0      # OLR intercept at reference CO₂ [W m⁻²]
    B::Float64         = 2.0        # OLR slope [W m⁻² K⁻¹]

    # CO₂ radiative forcing: ΔF = F_CO2·ln(CO2/CO2_ref) reduces OLR
    CO2::Float64       = 280.0      # atmospheric CO₂ [ppm]
    CO2_ref::Float64   = 280.0      # reference CO₂ [ppm]
    F_CO2::Float64     = 5.35       # forcing coefficient [W m⁻²]

    # Meridional heat diffusion
    D::Float64         = 0.50       # [W m⁻² K⁻¹]

    # Heat capacity per surface type [J m⁻² K⁻¹]  (FILLET Table 4)
    C_land::Float64    = 1e7
    C_ocean::Float64   = 4e8
    C_ice::Float64     = 1e7

    # Land fraction per latitudinal band [0–1].
    # Empty vector → FILLET default (0.25 uniform).  Length must equal n when provided.
    land_fraction::Vector{Float64} = Float64[]

    # Surface albedo [FILLET Table 4]
    α_land::Float64    = 0.30
    α_ocean::Float64   = 0.20
    α_ice::Float64     = 0.60
    T_ice::Float64     = -10.0      # [°C] annual-mean freezing threshold

    # Time stepping
    dt::Float64        = 86400.0 * 365 / 12   # ~1 month [s]
    tol::Float64       = 1e-4                  # convergence tolerance [K yr⁻¹]
    max_years::Int     = 500

    # Mode
    seasonal::Bool     = false      # false → annual-mean Q; true → time-varying Q
end

const S_earth = 1361.0   # W m⁻²

"""Scale instellation by semi-major axis: S(a) = S⊕/a² [au]."""
S0_from_au(a::Float64) = S_earth / a^2

# ============================================================
# Grid
# ============================================================

make_grid(p::Params) = collect(range(-1.0, 1.0, length=p.n))
lat_deg(x::AbstractVector) = asind.(x)

"""
Approximate Earth land fraction by sin-latitude (pre-industrial).

Knots are (sin_lat, land_fraction) pairs derived from actual Earth land/ocean
area by latitude band. Global mean ≈ 0.30. Captures the NH/SH asymmetry
(more land at NH mid-latitudes, Southern Ocean gap, Antarctic continent).
"""
function earth_land_fraction(x::AbstractVector)
    # sin_lat knots (band mid-points), land fraction values
    kx = [-1.000, -0.933, -0.787, -0.604, -0.380, -0.130,
           0.130,  0.380,  0.604,  0.787,  0.916,  0.983, 1.000]
    kf = [ 0.41,   0.41,   0.01,   0.08,   0.23,   0.24,
           0.31,   0.39,   0.46,   0.55,   0.48,   0.17,  0.17]
    return [_linterp(Float64(xi), kx, kf) for xi in x]
end

function _linterp(xi::Float64, xs::Vector{Float64}, ys::Vector{Float64})
    xi <= xs[1]   && return ys[1]
    xi >= xs[end] && return ys[end]
    i = searchsortedlast(xs, xi)
    t = (xi - xs[i]) / (xs[i+1] - xs[i])
    return ys[i] + t * (ys[i+1] - ys[i])
end

"""
Effective heat capacity [J m⁻² K⁻¹] for each latitudinal band.

Uses `p.land_fraction` if provided (length n), otherwise FILLET default (0.25 uniform).
Ice heat capacity (`C_ice`) is not applied dynamically here to keep the IMEX
solver matrix constant; it would require refactoring the matrix each timestep.
"""
function heat_capacity(x::AbstractVector, p::Params)
    n  = length(x)
    fl = isempty(p.land_fraction) ? fill(0.25, n) : p.land_fraction
    length(fl) == n || error("land_fraction length ($(length(fl))) must equal n ($n)")
    return @. fl * p.C_land + (1 - fl) * p.C_ocean
end

# ============================================================
# Physics functions
# ============================================================

"""
Annual-mean insolation [W m⁻²] for a circular orbit at obliquity ε.

Averages the Berger (1978) daily-mean formula over 360 solar longitudes.
Used in annual-mean mode and for FILLET output diagnostics in seasonal mode.
"""
function insolation(x::AbstractVector, p::Params)
    ε  = deg2rad(p.obliquity)
    nλ = 360
    Q  = zeros(length(x))
    for (i, xi) in enumerate(x)
        φ = asin(clamp(xi, -1.0, 1.0))
        q = 0.0
        for k in 0:nλ-1
            λ  = (k + 0.5) * (2π / nλ)
            δ  = asin(sin(ε) * sin(λ))
            t  = -tan(φ) * tan(δ)
            H0 = t ≤ -1.0 ? π : t ≥ 1.0 ? 0.0 : acos(t)
            q += H0 * sin(φ)*sin(δ) + cos(φ)*cos(δ)*sin(H0)
        end
        Q[i] = (p.S0 / π) * q / nλ
    end
    return Q
end

"""
Instantaneous daily-mean insolation [W m⁻²] at solar longitude λ [radians].

Uses the Berger (1978) formula for a single solar declination:
    δ = arcsin(sin(ε)·sin(λ))
    W(φ, δ) = (S₀/π)·[H₀·sin(φ)sin(δ) + cos(φ)cos(δ)·sin(H₀)]
"""
function insolation_instant(x::AbstractVector, p::Params, λ::Float64)
    ε = deg2rad(p.obliquity)
    δ = asin(sin(ε) * sin(λ))
    Q = zeros(length(x))
    for (i, xi) in enumerate(x)
        φ  = asin(clamp(xi, -1.0, 1.0))
        t  = -tan(φ) * tan(δ)
        H0 = t ≤ -1.0 ? π : t ≥ 1.0 ? 0.0 : acos(t)
        Q[i] = (p.S0 / π) * (H0 * sin(φ)*sin(δ) + cos(φ)*cos(δ)*sin(H0))
    end
    return Q
end

albedo(T::Real, fl::Real, p::Params) = T < p.T_ice ? p.α_ice : fl * p.α_land + (1 - fl) * p.α_ocean

function albedo(T::AbstractVector, p::Params)
    n  = length(T)
    fl = isempty(p.land_fraction) ? fill(0.25, n) : p.land_fraction
    return albedo.(T, fl, Ref(p))
end

"""Effective OLR intercept after CO₂ forcing [W m⁻²]."""
A_eff(p::Params) = p.A - p.F_CO2 * log(p.CO2 / p.CO2_ref)

# ============================================================
# Implicit diffusion operator
# ============================================================

function build_diffusion_matrix(x::AbstractVector, D::Float64)
    n  = length(x)
    dx = x[2] - x[1]
    dl = zeros(n - 1)
    d  = zeros(n)
    du = zeros(n - 1)

    for i in 2:n-1
        x_up = (x[i] + x[i+1]) / 2
        x_dn = (x[i-1] + x[i]) / 2
        c_up = D * (1 - x_up^2) / dx^2
        c_dn = D * (1 - x_dn^2) / dx^2
        dl[i-1] = c_dn
        d[i]    = -(c_up + c_dn)
        du[i]   = c_up
    end

    x_up = (x[1] + x[2]) / 2
    c_up = D * (1 - x_up^2) / dx^2
    d[1]  = -c_up
    du[1] =  c_up

    x_dn = (x[n-1] + x[n]) / 2
    c_dn = D * (1 - x_dn^2) / dx^2
    dl[n-1] = c_dn
    d[n]    = -c_dn

    return Tridiagonal(dl, d, du)
end

"""
Build and factor the IMEX system matrix: M = Diagonal(C_eff/dt + B) − L.

`C_eff` is the per-band effective heat capacity from `heat_capacity(x, p)`.
Independent of Q, CO₂, and season — factored once at startup.
"""
function build_solver(x::AbstractVector, p::Params, C_eff::AbstractVector)
    L = build_diffusion_matrix(x, p.D)
    M = -Matrix(L)
    for i in eachindex(C_eff)
        M[i,i] += C_eff[i] / p.dt + p.B
    end
    return factorize(M)
end

# ============================================================
# Time stepping
# ============================================================

function imex_step!(T::AbstractVector, rhs_buf::AbstractVector,
                    Q::AbstractVector, F::Factorization, p::Params,
                    C_eff::AbstractVector)
    Aeff = A_eff(p)
    α = albedo(T, p)
    @. rhs_buf = C_eff / p.dt * T + Q * (1 - α) - Aeff
    T .= F \ rhs_buf
    return T
end

# ============================================================
# Public API
# ============================================================

"""
    run_ebm(p; T0, save_every, verbose) → (t, T, x, p)

Integrate forward until convergence or `p.max_years`.

Annual-mean mode: fixed-point convergence (ΔT < tol K yr⁻¹).
Seasonal mode: limit-cycle convergence (same-phase year-over-year ΔT < tol).

Default save_every: 1 (every step) in seasonal mode, 12 (annual) otherwise.
"""
function run_ebm(p::Params;
                 T0::Union{Nothing, AbstractVector} = nothing,
                 save_every::Int = p.seasonal ? 1 : 12,
                 verbose::Bool = true)

    x     = make_grid(p)
    T     = isnothing(T0) ? fill(15.0, p.n) : float.(copy(T0))
    C_eff = heat_capacity(x, p)
    F     = build_solver(x, p, C_eff)
    buf   = similar(T)

    # Precompute annual-mean Q for non-seasonal mode
    Q_annual = p.seasonal ? nothing : insolation(x, p)

    steps_per_year = round(Int, 86400.0 * 365 / p.dt)
    max_steps      = p.max_years * steps_per_year
    nsaved         = max_steps ÷ save_every + 2

    T_hist       = zeros(p.n, nsaved)
    t_hist       = zeros(nsaved)
    T_hist[:, 1] = T
    si           = 2
    T_year_start = copy(T)

    for step in 1:max_steps
        Q = p.seasonal ?
            insolation_instant(x, p, 2π * ((step - 1) % steps_per_year) / steps_per_year) :
            Q_annual
        imex_step!(T, buf, Q, F, p, C_eff)

        if step % save_every == 0 && si <= nsaved
            T_hist[:, si] = T
            t_hist[si]    = step * p.dt / (86400.0 * 365)
            si += 1
        end

        if step % steps_per_year == 0
            yr    = step ÷ steps_per_year
            dT_yr = maximum(abs.(T .- T_year_start))
            T_year_start .= T
            verbose && yr % 50 == 0 &&
                println("  Year $yr: max ΔT/yr = $(round(dT_yr, sigdigits=3)) K")
            if dT_yr < p.tol
                verbose && println("  Converged at year $yr.")
                break
            end
        end
    end

    return (t=t_hist[1:si-1], T=T_hist[:, 1:si-1], x=x, p=p)
end

"""
    equilibrium(p; T0, verbose) → (T, α, olr, [T_seasonal,] x, p, years)

Run to equilibrium and return annual-mean state.

In annual-mean mode: T, α, olr are the fixed-point values.
In seasonal mode: T, α, olr are averages over the last converged year;
  T_seasonal (n_lat × steps_per_year) is the full annual cycle.

All downstream functions (FILLET output, diagnostics) use T, α, olr uniformly.
"""
function equilibrium(p::Params;
                     T0::Union{Nothing, AbstractVector} = nothing,
                     verbose::Bool = true)
    r    = run_ebm(p; T0=T0, verbose=verbose)
    Aeff = A_eff(p)

    if p.seasonal
        spy  = round(Int, 86400.0 * 365 / p.dt)
        last = r.T[:, max(1, size(r.T, 2) - spy + 1):end]   # last year's snapshots
        T_mean   = vec(mean(last, dims=2))
        α_mean   = vec(mean(reduce(hcat, [albedo(last[:, k], p) for k in axes(last, 2)]), dims=2))
        olr_mean = vec(mean(reduce(hcat, [@. Aeff + p.B * last[:, k] for k in axes(last, 2)]), dims=2))
        return (T=T_mean, α=α_mean, olr=olr_mean, T_seasonal=last,
                x=r.x, p=r.p, years=r.t[end])
    else
        T_eq = r.T[:, end]
        return (T=T_eq, α=albedo(T_eq, p), olr=@.(Aeff + p.B * T_eq),
                x=r.x, p=r.p, years=r.t[end])
    end
end

# ============================================================
# Diagnostics
# ============================================================

global_mean(field::AbstractVector) = sum(field) / length(field)

function fmt_time(s::Float64)
    s = round(Int, s)
    m, s = divrem(s, 60)
    h, m = divrem(m, 60)
    h > 0 ? @sprintf("%dh%02dm%02ds", h, m, s) : @sprintf("%dm%02ds", m, s)
end

"""
Ice-edge latitudes per FILLET template convention.

NH values are positive degrees; SH values are negative degrees.
  NH_max — poleward NH edge (numerically largest, ≤ 90)
  NH_min — equatorward NH edge (numerically smallest, ≥ 0)
  SH_max — equatorward SH edge (numerically largest, ≤ 0)
  SH_min — poleward SH edge (numerically smallest, ≥ −90)

Ice-free: NH → (90, 90),  SH → (−90, −90)   [FILLET template convention]
Snowball: NH → (90,  0),  SH → (  0, −90)
Ice caps: NH → (90, φ),   SH → (−φ, −90)
For equatorial ice, NH_min = 0 and SH_max = 0 (no edge at equator).
"""
function ice_edges(T::AbstractVector, x::AbstractVector, p::Params)
    ice    = T .< p.T_ice
    eq_idx = argmin(abs.(x))
    at_eq  = ice[eq_idx]

    # NH: positive degrees, Max=poleward, Min=equatorward; ice-free → (90, 90)
    nh_idx = findall(i -> x[i] > 0 && ice[i], eachindex(x))
    if isempty(nh_idx)
        nh_max = 90.0; nh_min = 90.0
    else
        nh_max = asind(x[maximum(nh_idx)])
        nh_min = at_eq ? 0.0 : asind(x[minimum(nh_idx)])
    end

    # SH: negative degrees, Max=equatorward (least negative), Min=poleward (most negative)
    # ice-free → (−90, −90)
    sh_idx = findall(i -> x[i] < 0 && ice[i], eachindex(x))
    if isempty(sh_idx)
        sh_max = -90.0; sh_min = -90.0
    else
        sh_max = at_eq ? 0.0 : asind(x[maximum(sh_idx)])   # least negative = equatorward
        sh_min = asind(x[minimum(sh_idx)])                  # most negative  = poleward
    end

    return (NH_max=nh_max, NH_min=nh_min, SH_max=sh_max, SH_min=sh_min)
end

"""NH ice-edge latitude [degrees] — equatorward edge, NaN if ice-free."""
function ice_edge_NH(T::AbstractVector, x::AbstractVector, p::Params)
    e = ice_edges(T, x, p)
    return e.NH_min == 90.0 ? NaN : e.NH_min
end

"""Format NH ice-edge for display: "ice-free" if no ice, otherwise "XX.X°"."""
fmt_ice(T, x, p) = let e = ice_edge_NH(T, x, p)
    isnan(e) ? "ice-free" : @sprintf("%.1f°", e)
end

"""
Global energy budget. Returns (SW_absorbed, OLR, imbalance) in W m⁻².
Uses annual-mean Q and the provided α (may be annual-mean α in seasonal mode).
"""
function energy_budget(T::AbstractVector, x::AbstractVector, p::Params;
                       α_mean::Union{Nothing, AbstractVector} = nothing,
                       olr_mean::Union{Nothing, AbstractVector} = nothing)
    Q    = insolation(x, p)
    α    = isnothing(α_mean)   ? albedo(T, p)              : α_mean
    olr  = isnothing(olr_mean) ? @.(A_eff(p) + p.B * T)   : olr_mean
    SW   = global_mean(@. Q * (1 - α))
    OLR  = global_mean(olr)
    return (SW_absorbed=SW, OLR=OLR, imbalance=SW - OLR)
end

# ============================================================
# FILLET intercomparison outputs
# ============================================================

const K_OFFSET = 273.15

"""
Per-latitude profile in FILLET v1.1 format.

`α_mean` and `olr_mean` should be annual-mean fields in seasonal mode.
TOA albedo equals surface albedo (no atmospheric scattering in this model).
"""
function fillet_profile(T::AbstractVector, x::AbstractVector, p::Params;
                        α_mean::Union{Nothing, AbstractVector}   = nothing,
                        olr_mean::Union{Nothing, AbstractVector} = nothing)
    n   = length(x)
    α   = isnothing(α_mean)   ? albedo(T, p)             : α_mean
    olr = isnothing(olr_mean) ? @.(A_eff(p) + p.B * T)   : olr_mean
    fl  = isempty(p.land_fraction) ? fill(0.25, n) : p.land_fraction
    return (lat=lat_deg(x), Tsurf=T .+ K_OFFSET, Asurf=α, ATOA=α, OLR=olr, Fland=fl)
end

"""
Global scalar outputs in FILLET template format.

Columns follow the FILLET v1.1 template (global_output.dat):
  Case Inst Obl XCO2 Tglob
  IceLineNMaxLand IceLineNMinLand IceLineNMaxSea IceLineNMinSea
  IceLineSMaxLand IceLineSMinLand IceLineSMaxSea IceLineSMinSea
  Diff OLRglob

AVALON uses a single temperature per band (land fraction blended), so land and sea
ice lines are identical; both sets of columns carry the same values.
Pass `α_mean` and `olr_mean` from `equilibrium` for correct annual means in seasonal mode.
"""
function fillet_global(T::AbstractVector, x::AbstractVector, p::Params;
                       instellation::Float64 = p.S0 / S_earth,
                       case::Int = 0,
                       α_mean::Union{Nothing, AbstractVector}   = nothing,
                       olr_mean::Union{Nothing, AbstractVector} = nothing)
    budget = energy_budget(T, x, p; α_mean=α_mean, olr_mean=olr_mean)
    edges  = ice_edges(T, x, p)
    return (
        Case        = case,
        Inst        = instellation,
        Obl         = p.obliquity,
        XCO2        = p.CO2,
        Tglob       = global_mean(T) + K_OFFSET,
        IceLineNMaxLand = edges.NH_max,
        IceLineNMinLand = edges.NH_min,
        IceLineNMaxSea  = edges.NH_max,
        IceLineNMinSea  = edges.NH_min,
        IceLineSMaxLand = edges.SH_max,
        IceLineSMinLand = edges.SH_min,
        IceLineSMaxSea  = edges.SH_max,
        IceLineSMinSea  = edges.SH_min,
        Diff        = p.D,
        OLRglob     = budget.OLR,
    )
end

"""
Write FILLET-format .dat files for one equilibrium state into `outdir/`.

Creates:
  `{outdir}/lat_output_AVALON_{tag}.dat`    — per-latitude profile
  `{outdir}/global_output_AVALON_{tag}.dat` — global scalar row
  `{outdir}/{tag}_seasonal.csv`             — monthly snapshots (if T_seasonal given)

Pass `α_mean` and `olr_mean` from `equilibrium` for correct annual means in seasonal mode.
"""
function write_fillet_output(T::AbstractVector, x::AbstractVector, p::Params,
                              outdir::String, tag::String;
                              instellation::Float64 = p.S0 / S_earth,
                              case::Int = 0,
                              label::String = tag,
                              α_mean::Union{Nothing, AbstractVector}    = nothing,
                              olr_mean::Union{Nothing, AbstractVector}  = nothing,
                              T_seasonal::Union{Nothing, AbstractMatrix} = nothing)
    mkpath(outdir)
    prof = fillet_profile(T, x, p; α_mean=α_mean, olr_mean=olr_mean)
    open(joinpath(outdir, "lat_output_AVALON_$(tag).dat"), "w") do io
        println(io, "# Name of benchmark/experiment: $label")
        println(io, "# Case number: $case")
        println(io, "# Instellation (S_earth): $(round(instellation, digits=6))")
        println(io, "# XCO2 (ppm): $(p.CO2)")
        println(io, "# Obliquity (degrees): $(p.obliquity)")
        println(io, "# Mode: $(p.seasonal ? "seasonal" : "annual-mean")")
        println(io, "#")
        println(io, "# Columns of data (annually averaged for last orbit)")
        println(io, "# Lat Tsurf Asurf ATOA OLR Fland")
        for i in eachindex(x)
            @printf(io, "%7.2f %8.2f %6.4f %6.4f %8.2f %6.4f\n",
                    prof.lat[i], prof.Tsurf[i], prof.Asurf[i], prof.ATOA[i], prof.OLR[i], prof.Fland[i])
        end
    end

    g = fillet_global(T, x, p; instellation=instellation, case=case, α_mean=α_mean, olr_mean=olr_mean)
    open(joinpath(outdir, "global_output_AVALON_$(tag).dat"), "w") do io
        println(io, "# Name of benchmark/experiment: $label")
        println(io, "# Model: AVALON (Albedo-feedback Variable Axial-tilt Latitudinal Outgoing-Net EBM)")
        println(io, "# Mode: $(p.seasonal ? "seasonal" : "annual-mean")")
        println(io, "# Ice line definition: latitude where annual-mean surface temperature crosses $(p.T_ice) °C")
        println(io, "# Single T per band: land and sea ice lines are identical (FILLET v1.1 four-value-per-hemisphere format)")
        println(io, "# Ice-free convention: NMaxLand=NMinLand=NMaxSea=NMinSea=90; SMaxLand=SMinLand=SMaxSea=SMinSea=-90")
        println(io, "# Case Inst Obl XCO2 Tglob IceLineNMaxLand IceLineNMinLand IceLineNMaxSea IceLineNMinSea IceLineSMaxLand IceLineSMinLand IceLineSMaxSea IceLineSMinSea Diff OLRglob")
        println(io, join(string.(values(g)), " "))
    end

    if !isnothing(T_seasonal)
        open(joinpath(outdir, "$(tag)_seasonal.csv"), "w") do io
            println(io, "month," * join(string.(round.(lat_deg(x), digits=4)), ","))
            for k in axes(T_seasonal, 2)
                println(io, "$k," * join(T_seasonal[:, k] .+ K_OFFSET, ","))
            end
        end
    end
end

"""
Run the FILLET instellation × obliquity sweep (Experiments 1 / 2).

`S0_factors`  — instellation values as multiples of S⊕
`obliquities` — obliquity values [degrees]
`warm_start`  — true → Experiment 1 (T₀ = 30 °C), false → Experiment 2 (T₀ = −50 °C)
`outdir`      — output subdirectory; tag is derived from basename(outdir)

Writes per-case `lat_output_AVALON_{tag}_{case}.dat` and one
`global_output_AVALON_{tag}.dat` inside `outdir/`.
"""
function run_fillet_sweep(S0_factors::AbstractVector, obliquities::AbstractVector;
                          warm_start::Bool  = true,
                          p_base::Params    = Params(),
                          outdir::String    = warm_start ? joinpath("experiments","exp1") : joinpath("experiments","exp2"),
                          label::String     = warm_start ? "FILLET Experiment 1 (warm start)" :
                                                           "FILLET Experiment 2 (cold start)",
                          verbose::Bool     = true)
    tag     = basename(outdir)
    T0_val  = warm_start ? 30.0 : -50.0
    rows    = NamedTuple[]
    n_total = length(S0_factors) * length(obliquities)
    t_start = time()

    mkpath(outdir)
    global_path = joinpath(outdir, "global_output_AVALON_$(tag).dat")
    open(global_path, "w") do io
        println(io, "# Name of benchmark/experiment: $label")
        println(io, "# Model: AVALON (Albedo-feedback Variable Axial-tilt Latitudinal Outgoing-Net EBM)")
        println(io, "# Mode: $(p_base.seasonal ? "seasonal" : "annual-mean")")
        println(io, "# Ice line definition: latitude where annual-mean surface temperature crosses $(p_base.T_ice) °C")
        println(io, "# Ice-free convention: NMaxLand=NMinLand=NMaxSea=NMinSea=90; SMaxLand=SMinLand=SMaxSea=SMinSea=-90")
        println(io, "# Case Inst Obl XCO2 Tglob IceLineNMaxLand IceLineNMinLand IceLineNMaxSea IceLineNMinSea IceLineSMaxLand IceLineSMinLand IceLineSMaxSea IceLineSMinSea Diff OLRglob")
        for (n_done, (obl, sf)) in enumerate((obl, sf) for obl in obliquities for sf in S0_factors)
            kw = Dict{Symbol,Any}(f => getfield(p_base, f) for f in fieldnames(Params))
            kw[:S0]        = Float64(sf) * S_earth
            kw[:obliquity] = Float64(obl)
            kw[:max_years] = 500
            p = Params(; kw...)
            r = equilibrium(p; T0=fill(T0_val, p.n), verbose=false)
            g = fillet_global(r.T, r.x, r.p;
                              instellation=Float64(sf), case=n_done-1,
                              α_mean=r.α, olr_mean=r.olr)
            push!(rows, g)
            println(io, join(string.(values(g)), " "))
            flush(io)

            # Per-case lat file
            prof     = fillet_profile(r.T, r.x, r.p; α_mean=r.α, olr_mean=r.olr)
            lat_path = joinpath(outdir, "lat_output_AVALON_$(tag)_$(n_done-1).dat")
            open(lat_path, "w") do lat_io
                println(lat_io, "# Name of benchmark/experiment: $label")
                println(lat_io, "# Case number: $(n_done-1)")
                println(lat_io, "# Instellation (S_earth): $(round(Float64(sf), digits=6))")
                println(lat_io, "# XCO2 (ppm): $(p.CO2)")
                println(lat_io, "# Obliquity (degrees): $(Float64(obl))")
                println(lat_io, "# Mode: $(p.seasonal ? "seasonal" : "annual-mean")")
                println(lat_io, "#")
                println(lat_io, "# Columns of data (annually averaged for last orbit)")
                println(lat_io, "# Lat Tsurf Asurf ATOA OLR Fland")
                for i in eachindex(r.x)
                    @printf(lat_io, "%7.2f %8.2f %6.4f %6.4f %8.2f %6.4f\n",
                            prof.lat[i], prof.Tsurf[i], prof.Asurf[i], prof.ATOA[i], prof.OLR[i], prof.Fland[i])
                end
            end

            if verbose
                elapsed = time() - t_start
                eta_str = n_done < n_total ?
                    "  ETA $(fmt_time(elapsed / n_done * (n_total - n_done)))" : ""
                ice_str = g.IceLineNMinSea == 90.0 ? "  free" :
                    @sprintf("%5.1f°", g.IceLineNMinSea)
                @printf("  [%3d/%d] obl=%2.0f°  S⊕=%.3f  Tglob=%6.1f K  ice=%s  %s elapsed%s\n",
                        n_done, n_total, Float64(obl), Float64(sf),
                        g.Tglob, ice_str, fmt_time(elapsed), eta_str)
            end
        end
    end
    verbose && println("  Done — $(n_total) cases written to $(outdir)/")
    return rows
end

"""
Run Experiments 1a / 2a: obliquity × semi-major axis sweep.
Instellation is set by S(a) = S⊕/a².
"""
function run_fillet_sweep_au(a_range::AbstractVector, obliquities::AbstractVector;
                              warm_start::Bool = true,
                              p_base::Params   = Params(),
                              outdir::String   = warm_start ? joinpath("experiments","exp1a") : joinpath("experiments","exp2a"),
                              label::String    = warm_start ? "FILLET Experiment 1a (warm start, au)" :
                                                              "FILLET Experiment 2a (cold start, au)",
                              verbose::Bool    = true)
    S0_factors = S_earth ./ (collect(a_range) .^ 2) ./ S_earth
    run_fillet_sweep(S0_factors, obliquities;
                     warm_start=warm_start, p_base=p_base,
                     outdir=outdir, label=label, verbose=verbose)
end

# ============================================================
# Research tools
# ============================================================

"""
    bifurcation_diagram(S0_range; p_base, verbose) → (S0, mean_T, ice_lat, branch)

Snowball Earth hysteresis by sweeping instellation (FILLET Experiment 3).
Obliquity is fixed via p_base.obliquity.
"""
function bifurcation_diagram(S0_range::AbstractVector;
                              p_base::Params = Params(),
                              outdir::Union{Nothing,String} = nothing,
                              label::String = "FILLET Experiment 3",
                              verbose::Bool = true)
    S0_sorted = sort(S0_range)
    mean_T  = Float64[]
    ice_lat = Float64[]
    all_S0  = Float64[]
    branch  = String[]

    if !isnothing(outdir)
        mkpath(outdir)
    end
    tag      = isnothing(outdir) ? "" : basename(outdir)
    case_idx = Ref(0)

    function run_branch!(branch_S0, branch_name, T0_val, global_io)
        T_prev = fill(T0_val, p_base.n)
        for S0 in branch_S0
            kw = Dict{Symbol,Any}(f => getfield(p_base, f) for f in fieldnames(Params))
            kw[:S0] = S0; kw[:max_years] = 500
            p = Params(; kw...)
            r = equilibrium(p; T0=T_prev, verbose=false)
            push!(mean_T,  global_mean(r.T))
            push!(ice_lat, ice_edge_NH(r.T, r.x, r.p))
            push!(all_S0,  S0)
            push!(branch,  branch_name)
            T_prev = r.T
            verbose && print(branch_name == "cooling" ? "-" : "+")
            if !isnothing(global_io)
                g = fillet_global(r.T, r.x, r.p;
                                  instellation=S0/S_earth,
                                  case=case_idx[],
                                  α_mean=r.α, olr_mean=r.olr)
                println(global_io, join(string.(values(g)), " ") * " " * branch_name)
                flush(global_io)
                prof = fillet_profile(r.T, r.x, r.p; α_mean=r.α, olr_mean=r.olr)
                open(joinpath(outdir, "lat_output_AVALON_$(tag)_$(case_idx[]).dat"), "w") do lat_io
                    println(lat_io, "# Name of benchmark/experiment: $label")
                    println(lat_io, "# Case number: $(case_idx[])")
                    println(lat_io, "# Branch: $branch_name")
                    println(lat_io, "# Instellation (S_earth): $(round(S0/S_earth, digits=6))")
                    println(lat_io, "# XCO2 (ppm): $(p.CO2)")
                    println(lat_io, "# Obliquity (degrees): $(p.obliquity)")
                    println(lat_io, "# Mode: $(p.seasonal ? "seasonal" : "annual-mean")")
                    println(lat_io, "#")
                    println(lat_io, "# Columns of data (annually averaged for last orbit)")
                    println(lat_io, "# Lat Tsurf Asurf ATOA OLR Fland")
                    for i in eachindex(r.x)
                        @printf(lat_io, "%7.2f %8.2f %6.4f %6.4f %8.2f %6.4f\n",
                                prof.lat[i], prof.Tsurf[i], prof.Asurf[i], prof.ATOA[i], prof.OLR[i], prof.Fland[i])
                    end
                end
                case_idx[] += 1
            end
        end
    end

    if !isnothing(outdir)
        open(joinpath(outdir, "global_output_AVALON_$(tag).dat"), "w") do global_io
            println(global_io, "# Name of benchmark/experiment: $label")
            println(global_io, "# Model: AVALON (Albedo-feedback Variable Axial-tilt Latitudinal Outgoing-Net EBM)")
            println(global_io, "# Mode: $(p_base.seasonal ? "seasonal" : "annual-mean")")
            println(global_io, "# Ice line definition: latitude where annual-mean surface temperature crosses $(p_base.T_ice) °C")
            println(global_io, "# Ice-free convention: NMaxLand=NMinLand=NMaxSea=NMinSea=90; SMaxLand=SMinLand=SMaxSea=SMinSea=-90")
            println(global_io, "# Case Inst Obl XCO2 Tglob IceLineNMaxLand IceLineNMinLand IceLineNMaxSea IceLineNMinSea IceLineSMaxLand IceLineSMinLand IceLineSMaxSea IceLineSMinSea Diff OLRglob Branch")
            verbose && print("Cooling: ")
            run_branch!(reverse(S0_sorted), "cooling", 30.0, global_io)
            verbose && println()
            verbose && print("Warming: ")
            run_branch!(S0_sorted, "warming", -50.0, global_io)
            verbose && println()
        end
    else
        verbose && print("Cooling: ")
        run_branch!(reverse(S0_sorted), "cooling", 30.0, nothing)
        verbose && println()
        verbose && print("Warming: ")
        run_branch!(S0_sorted, "warming", -50.0, nothing)
        verbose && println()
    end

    return (S0=all_S0, mean_T=mean_T, ice_lat=ice_lat, branch=branch)
end

"""
    co2_bifurcation(CO2_range; p_base, verbose) → (CO2, mean_T, ice_lat, branch)

Bifurcation diagram sweeping CO₂ (FILLET Experiment 4).
Recommended: `exp10.(range(0, 5, length=50))` for 1–100,000 ppm.
"""
function co2_bifurcation(CO2_range::AbstractVector;
                          p_base::Params = Params(),
                          outdir::Union{Nothing,String} = nothing,
                          label::String = "FILLET Experiment 4",
                          verbose::Bool = true)
    CO2_sorted = sort(CO2_range)
    mean_T  = Float64[]
    ice_lat = Float64[]
    all_CO2 = Float64[]
    branch  = String[]

    if !isnothing(outdir)
        mkpath(outdir)
    end
    tag      = isnothing(outdir) ? "" : basename(outdir)
    case_idx = Ref(0)

    function run_branch!(branch_CO2, branch_name, T0_val, global_io)
        T_prev = fill(T0_val, p_base.n)
        for co2 in branch_CO2
            kw = Dict{Symbol,Any}(f => getfield(p_base, f) for f in fieldnames(Params))
            kw[:CO2] = co2; kw[:max_years] = 500
            p = Params(; kw...)
            r = equilibrium(p; T0=T_prev, verbose=false)
            push!(mean_T,  global_mean(r.T))
            push!(ice_lat, ice_edge_NH(r.T, r.x, r.p))
            push!(all_CO2, co2)
            push!(branch,  branch_name)
            T_prev = r.T
            verbose && print(branch_name == "cooling" ? "-" : "+")
            if !isnothing(global_io)
                g = fillet_global(r.T, r.x, r.p;
                                  case=case_idx[],
                                  α_mean=r.α, olr_mean=r.olr)
                println(global_io, join(string.(values(g)), " ") * " " * branch_name)
                flush(global_io)
                prof = fillet_profile(r.T, r.x, r.p; α_mean=r.α, olr_mean=r.olr)
                open(joinpath(outdir, "lat_output_AVALON_$(tag)_$(case_idx[]).dat"), "w") do lat_io
                    println(lat_io, "# Name of benchmark/experiment: $label")
                    println(lat_io, "# Case number: $(case_idx[])")
                    println(lat_io, "# Branch: $branch_name")
                    println(lat_io, "# Instellation (S_earth): $(round(p.S0/S_earth, digits=6))")
                    println(lat_io, "# XCO2 (ppm): $(p.CO2)")
                    println(lat_io, "# Obliquity (degrees): $(p.obliquity)")
                    println(lat_io, "# Mode: $(p.seasonal ? "seasonal" : "annual-mean")")
                    println(lat_io, "#")
                    println(lat_io, "# Columns of data (annually averaged for last orbit)")
                    println(lat_io, "# Lat Tsurf Asurf ATOA OLR Fland")
                    for i in eachindex(r.x)
                        @printf(lat_io, "%7.2f %8.2f %6.4f %6.4f %8.2f %6.4f\n",
                                prof.lat[i], prof.Tsurf[i], prof.Asurf[i], prof.ATOA[i], prof.OLR[i], prof.Fland[i])
                    end
                end
                case_idx[] += 1
            end
        end
    end

    if !isnothing(outdir)
        open(joinpath(outdir, "global_output_AVALON_$(tag).dat"), "w") do global_io
            println(global_io, "# Name of benchmark/experiment: $label")
            println(global_io, "# Model: AVALON (Albedo-feedback Variable Axial-tilt Latitudinal Outgoing-Net EBM)")
            println(global_io, "# Mode: $(p_base.seasonal ? "seasonal" : "annual-mean")")
            println(global_io, "# Ice line definition: latitude where annual-mean surface temperature crosses $(p_base.T_ice) °C")
            println(global_io, "# Ice-free convention: NMaxLand=NMinLand=NMaxSea=NMinSea=90; SMaxLand=SMinLand=SMaxSea=SMinSea=-90")
            println(global_io, "# Case Inst Obl XCO2 Tglob IceLineNMaxLand IceLineNMinLand IceLineNMaxSea IceLineNMinSea IceLineSMaxLand IceLineSMinLand IceLineSMaxSea IceLineSMinSea Diff OLRglob Branch")
            verbose && print("Cooling (high→low CO₂): ")
            run_branch!(reverse(CO2_sorted), "cooling", 30.0, global_io)
            verbose && println()
            verbose && print("Warming (low→high CO₂): ")
            run_branch!(CO2_sorted, "warming", -50.0, global_io)
            verbose && println()
        end
    else
        verbose && print("Cooling (high→low CO₂): ")
        run_branch!(reverse(CO2_sorted), "cooling", 30.0, nothing)
        verbose && println()
        verbose && print("Warming (low→high CO₂): ")
        run_branch!(CO2_sorted, "warming", -50.0, nothing)
        verbose && println()
    end

    return (CO2=all_CO2, mean_T=mean_T, ice_lat=ice_lat, branch=branch)
end

# ============================================================
# CLI
# ============================================================

function main()
    print(HELP_TEXT)
end

"""
Run a single arbitrary case from `key=value` command-line arguments.

Special keys (not Params fields):
  out=tag   — output directory/tag (default: "run")
  au=value  — set S0 from semi-major axis: S₀ = S⊕/a² [au]

All other keys must be valid `Params` field names. The ASCII aliases
`alpha_land`, `alpha_ocean`, and `alpha_ice` map to `α_land`, `α_ocean`, and `α_ice`.

Example:
    julia avalon.jl run obliquity=60 CO2=1000 seasonal=true out=highco2_obl60
"""
function run_custom(args::Vector{String})
    aliases = Dict("alpha_land" => "α_land", "alpha_ocean" => "α_ocean", "alpha_ice" => "α_ice")
    kw  = Dict{Symbol,Any}(f => getfield(Params(), f) for f in fieldnames(Params))
    tag = "run"
    au  = nothing

    for arg in args
        parts = split(arg, "="; limit=2)
        if length(parts) != 2
            println(stderr, "Error: expected key=value, got: $arg")
            exit(1)
        end
        k, v = String(parts[1]), String(parts[2])
        k = get(aliases, k, k)

        if k == "out"
            tag = v
        elseif k == "au"
            au = parse(Float64, v)
        else
            sym = Symbol(k)
            if sym ∉ fieldnames(Params)
                println(stderr, "Error: unknown parameter '$k'")
                println(stderr, "  Valid Params fields: $(join(fieldnames(Params), ", "))")
                println(stderr, "  ASCII aliases: alpha_land, alpha_ocean, alpha_ice")
                exit(1)
            end
            T = fieldtype(Params, sym)
            kw[sym] = T == Bool  ? (v == "true" || v == "1") :
                      T == Int   ? parse(Int, v) :
                                   parse(Float64, v)
        end
    end

    isnothing(au) || (kw[:S0] = S_earth / au^2)
    p = Params(; kw...)

    # Print non-default parameters
    defaults = Params()
    changed  = [string(f, "=", getfield(p, f))
                for f in fieldnames(Params) if getfield(p, f) != getfield(defaults, f)]
    isempty(changed) ? println("Running with all default parameters.") :
                       println("Parameters: $(join(changed, "  "))")

    outdir = joinpath("experiments", tag)
    r = equilibrium(p; verbose=true)
    write_fillet_output(r.T, r.x, p, outdir, tag;
                        instellation=p.S0/S_earth, case=0,
                        label="AVALON custom: $tag",
                        α_mean=r.α, olr_mean=r.olr,
                        T_seasonal=p.seasonal ? r.T_seasonal : nothing)
    println("Tglob = $(round(global_mean(r.T)+K_OFFSET, digits=2)) K  |  " *
            "ice edge: $(fmt_ice(r.T, r.x, p))")
    println("→ $(outdir)/lat_output_AVALON_$(tag).dat, $(outdir)/global_output_AVALON_$(tag).dat")
end

"""
Run a named FILLET benchmark or experiment and write the official .dat output files.

Usage:
    julia avalon.jl benchmark1   # Benchmark 1: tuned pre-industrial Earth (seasonal, D=0.52, C_ocean=2e8, Earth land fraction)
    julia avalon.jl benchmark2   # Benchmark 2: un-tuned, ε=23.5°
    julia avalon.jl benchmark3   # Benchmark 3: un-tuned, ε=60°
    julia avalon.jl exp1         # Experiment 1: warm-start instellation sweep
    julia avalon.jl exp2         # Experiment 2: cold-start instellation sweep
    julia avalon.jl exp1a        # Experiment 1a: warm-start semi-major axis sweep
    julia avalon.jl exp2a        # Experiment 2a: cold-start semi-major axis sweep
    julia avalon.jl exp3         # Experiment 3: instellation bifurcation
    julia avalon.jl exp4         # Experiment 4: CO₂ bifurcation
    julia avalon.jl              # print help text
"""
function run_cli(cmd::String, extra_args::Vector{String}=String[])
    if cmd == "benchmark1"
        println("FILLET Benchmark 1 (tuned pre-industrial Earth: D=0.52, α_ocean=0.2689, C_ocean=2e8, Earth land fraction)")
        x_ben1 = make_grid(Params())
        p = Params(D=0.52, α_ocean=0.2689, C_ocean=2e8,
                   land_fraction=earth_land_fraction(x_ben1), seasonal=true)
        r = equilibrium(p; verbose=true)
        write_fillet_output(r.T, r.x, p, joinpath("experiments","ben1"), "ben1";
                            instellation=1.0, case=0,
                            label="FILLET Benchmark 1",
                            α_mean=r.α, olr_mean=r.olr, T_seasonal=r.T_seasonal)
        println("Tglob = $(round(global_mean(r.T)+K_OFFSET, digits=2)) K  |  " *
                "ice edge: $(fmt_ice(r.T, r.x, p))")
        println("→ experiments/ben1/lat_output_AVALON_ben1.dat, experiments/ben1/global_output_AVALON_ben1.dat, experiments/ben1/ben1_seasonal.csv")

    elseif cmd == "benchmark2"
        println("FILLET Benchmark 2 (un-tuned, ε=23.5°)")
        p = Params(seasonal=true)
        r = equilibrium(p; verbose=true)
        write_fillet_output(r.T, r.x, p, joinpath("experiments","ben2"), "ben2";
                            instellation=1.0, case=0,
                            label="FILLET Benchmark 2",
                            α_mean=r.α, olr_mean=r.olr, T_seasonal=r.T_seasonal)
        println("Tglob = $(round(global_mean(r.T)+K_OFFSET, digits=2)) K  |  " *
                "ice edge: $(fmt_ice(r.T, r.x, p))")
        println("→ experiments/ben2/lat_output_AVALON_ben2.dat, experiments/ben2/global_output_AVALON_ben2.dat, experiments/ben2/ben2_seasonal.csv")

    elseif cmd == "benchmark3"
        println("FILLET Benchmark 3 (un-tuned, ε=60°)")
        p = Params(obliquity=60.0, seasonal=true)
        r = equilibrium(p; verbose=true)
        write_fillet_output(r.T, r.x, p, joinpath("experiments","ben3"), "ben3";
                            instellation=1.0, case=0,
                            label="FILLET Benchmark 3",
                            α_mean=r.α, olr_mean=r.olr, T_seasonal=r.T_seasonal)
        println("Tglob = $(round(global_mean(r.T)+K_OFFSET, digits=2)) K  |  " *
                "ice edge: $(fmt_ice(r.T, r.x, p))")
        println("→ experiments/ben3/lat_output_AVALON_ben3.dat, experiments/ben3/global_output_AVALON_ben3.dat, experiments/ben3/ben3_seasonal.csv")

    elseif cmd == "exp1"
        println("FILLET Experiment 1: warm-start instellation sweep (0.80–1.25 S⊕, ε=0–90°)")
        run_fillet_sweep(collect(range(0.80, 1.25, step=0.025)), collect(0:10:90);
                         warm_start=true, p_base=Params(seasonal=true),
                         outdir=joinpath("experiments","exp1"), verbose=true)
        println("→ experiments/exp1/  (lat_output_AVALON_exp1_{case}.dat × N + global_output_AVALON_exp1.dat)")

    elseif cmd == "exp2"
        println("FILLET Experiment 2: cold-start instellation sweep (1.05–1.50 S⊕, ε=0–90°)")
        run_fillet_sweep(collect(range(1.05, 1.50, step=0.025)), collect(0:10:90);
                         warm_start=false, p_base=Params(seasonal=true),
                         outdir=joinpath("experiments","exp2"), verbose=true)
        println("→ experiments/exp2/  (lat_output_AVALON_exp2_{case}.dat × N + global_output_AVALON_exp2.dat)")

    elseif cmd == "exp1a"
        println("FILLET Experiment 1a: warm-start semi-major axis sweep (0.875–1.10 au, ε=0–90°)")
        run_fillet_sweep_au(collect(range(0.875, 1.10, step=0.0125)), collect(0:10:90);
                            warm_start=true, p_base=Params(seasonal=true),
                            outdir=joinpath("experiments","exp1a"), verbose=true)
        println("→ experiments/exp1a/  (lat_output_AVALON_exp1a_{case}.dat × N + global_output_AVALON_exp1a.dat)")

    elseif cmd == "exp2a"
        println("FILLET Experiment 2a: cold-start semi-major axis sweep (0.80–0.975 au, ε=0–90°)")
        run_fillet_sweep_au(collect(range(0.80, 0.975, step=0.0125)), collect(0:10:90);
                            warm_start=false, p_base=Params(seasonal=true),
                            outdir=joinpath("experiments","exp2a"), verbose=true)
        println("→ experiments/exp2a/  (lat_output_AVALON_exp2a_{case}.dat × N + global_output_AVALON_exp2a.dat)")

    elseif cmd == "run"
        run_custom(extra_args)

    elseif cmd == "exp3"
        println("FILLET Experiment 3: instellation bifurcation (0.8–1.5 S⊕, ε=23.5°)")
        S0_range = collect(range(0.8, 1.5, step=0.0125)) .* S_earth
        bifurcation_diagram(S0_range;
                            p_base=Params(seasonal=true),
                            outdir=joinpath("experiments","exp3"),
                            label="FILLET Experiment 3 (instellation bifurcation)",
                            verbose=true)
        println("→ experiments/exp3/  (lat_output_AVALON_exp3_{case}.dat × N + global_output_AVALON_exp3.dat)")

    elseif cmd == "exp4"
        println("FILLET Experiment 4: CO₂ bifurcation (1–100,000 ppm)")
        co2_bifurcation(exp10.(range(0, 5, length=50));
                        p_base=Params(seasonal=true),
                        outdir=joinpath("experiments","exp4"),
                        label="FILLET Experiment 4 (CO₂ bifurcation)",
                        verbose=true)
        println("→ experiments/exp4/  (lat_output_AVALON_exp4_{case}.dat × N + global_output_AVALON_exp4.dat)")

    else
        println("""
Unknown command: "$cmd"

Usage:  julia avalon.jl <command> [key=value ...]

Commands:
  benchmark1   Benchmark 1 — tuned to 288 K (seasonal, D=0.52, α_ocean=0.2689, C_ocean=2e8, Earth land fraction) → experiments/ben1/
  benchmark2   Benchmark 2 — un-tuned, ε = 23.5°  → experiments/ben2/
  benchmark3   Benchmark 3 — un-tuned, ε = 60°    → experiments/ben3/
  run          Custom single case — accepts key=value parameter overrides  → experiments/<tag>/
  exp1         Experiment 1 — warm-start instellation sweep (0.80–1.25 S⊕, ε=0–90°)  → experiments/exp1/
  exp2         Experiment 2 — cold-start instellation sweep (1.05–1.50 S⊕, ε=0–90°)  → experiments/exp2/
  exp1a        Experiment 1a — warm-start semi-major axis sweep (0.875–1.10 au, ε=0–90°)  → experiments/exp1a/
  exp2a        Experiment 2a — cold-start semi-major axis sweep (0.80–0.975 au, ε=0–90°)  → experiments/exp2a/
  exp3         Experiment 3 — instellation bifurcation  → experiments/exp3/
  exp4         Experiment 4 — CO₂ bifurcation           → experiments/exp4/
  help         Print this help text  (also: --help, no args)

run examples:
  julia avalon.jl run obliquity=60 CO2=1000 seasonal=true out=highco2_obl60
  julia avalon.jl run au=0.9 obliquity=45 out=innerhz
  julia avalon.jl run S0=1200 alpha_ocean=0.28 D=0.44
""")
    end
end

const HELP_TEXT = """
AVALON — Albedo-feedback Variable Axial-tilt Latitudinal Outgoing-Net EBM
A 1D Budyko-Sellers energy balance model for the FILLET intercomparison project.

Usage:
  julia avalon.jl                        # print this help text
  julia avalon.jl <command> [key=value]
  julia avalon.jl help | --help | -h     # print this help text

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FILLET benchmarks
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  benchmark1   Tuned pre-industrial Earth (seasonal, D=0.52, α_ocean=0.2689, C_ocean=2e8, Earth land fraction, ε=23.5°, CO₂=280 ppm)
               → experiments/ben1/

  benchmark2   Un-tuned default parameters (seasonal, ε=23.5°, CO₂=280 ppm)
               → experiments/ben2/

  benchmark3   Un-tuned, high obliquity (seasonal, ε=60°, CO₂=280 ppm)
               → experiments/ben3/

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FILLET experiments
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  exp1         Warm-start instellation sweep  (0.80–1.25 S⊕ × ε=0–90°)  → experiments/exp1/
  exp2         Cold-start instellation sweep  (1.05–1.50 S⊕ × ε=0–90°)  → experiments/exp2/
  exp1a        Warm-start semi-major axis     (0.875–1.10 au × ε=0–90°)  → experiments/exp1a/
  exp2a        Cold-start semi-major axis     (0.80–0.975 au × ε=0–90°)  → experiments/exp2a/
  exp3         Instellation bifurcation diagram (hysteresis)              → experiments/exp3/
  exp4         CO₂ bifurcation diagram                                    → experiments/exp4/

  Experiments 1/2/1a/2a write one lat_output_AVALON_{exp}_{case}.dat per
  simulation plus a single global_output_AVALON_{exp}.dat summary.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Custom single case
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  run [key=value ...]

  Override any model parameter by name.  Output goes to out=<tag> (default: run/).

  Special keys:
    out=<tag>       output tag; files go to experiments/<tag>/  (default: run)
    au=<value>      set instellation from semi-major axis: S₀ = S⊕/a² [au]

  Model parameters (key=value):
    S0=<W/m²>       solar constant              (default: 1361.0)
    obliquity=<°>   axial tilt                  (default: 23.5)
    CO2=<ppm>       atmospheric CO₂             (default: 280.0)
    CO2_ref=<ppm>   CO₂ reference level         (default: 280.0)
    D=<W/m²/K>      diffusion coefficient        (default: 0.50)
    A=<W/m²>        OLR intercept               (default: 210.0)
    B=<W/m²/K>      OLR slope                   (default: 2.0)
    alpha_land=<0-1>  land surface albedo        (default: 0.30)
    alpha_ocean=<0-1> open ocean surface albedo  (default: 0.20)
    alpha_ice=<0-1>   ice surface albedo         (default: 0.60)
    T_ice=<°C>      ice threshold temperature    (default: -10.0)
    C_land=<J/m²K>  land heat capacity           (default: 1e7)
    C_ocean=<J/m²K> ocean heat capacity          (default: 4e8)
    C_ice=<J/m²K>   ice heat capacity            (default: 1e7)
    seasonal=true   enable seasonal cycle        (default: false)
    max_years=<N>   convergence limit            (default: 500)

  Note: land_fraction (per-band land fraction vector) can only be set
  programmatically, not via the command line. Default: 0.25 uniform (FILLET Table 4).

  Examples:
    julia avalon.jl run obliquity=60 CO2=1000 seasonal=true out=highco2_obl60
    julia avalon.jl run au=0.9 obliquity=45 out=innerhz
    julia avalon.jl run alpha_ocean=0.28 D=0.44 S0=1200

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Plotting  (requires numpy, matplotlib)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  python3 plot.py <tag>                Annual-mean diagnostic panels
                                       (2×2 with land fraction, or 1×3 without)
                                       Reads lat_output + global_output .dat files.
                                       If <tag>_seasonal.csv exists, adds seasonal envelope.

  python3 plot.py <tag> seasonal       Hovmöller temperature plot + seasonal amplitude panel
                                       Requires <tag>/<tag>_seasonal.csv

  python3 plot.py <tag> sweep          Obliquity × instellation phase diagram
                                       (climate states + global mean temperature)
                                       For exp1, exp2, exp1a, exp2a

  python3 plot.py <tag> bifurcation    Hysteresis diagram: Tglob and NH ice edge
                                       vs instellation (exp3) or CO₂ (exp4, log scale)
                                       Shows cooling and warming branches

  Examples:
    python3 plot.py ben1
    python3 plot.py ben1 seasonal
    python3 plot.py exp1 sweep
    python3 plot.py exp3 bifurcation
    python3 plot.py exp4 bifurcation
"""

if abspath(PROGRAM_FILE) == @__FILE__
    if isempty(ARGS)
        main()
    elseif ARGS[1] in ("--help", "-h", "help")
        print(HELP_TEXT)
    else
        run_cli(ARGS[1], ARGS[2:end])
    end
end
