"""
AVALON — Albedo-feedback Variable Axial-tilt Latitudinal Outgoing-Net EBM

1D Energy Balance Climate Model (Budyko-Sellers type)

Latitude-dependent surface temperature T(x, t) governed by:

    C(T) ∂T/∂t = Q(x,t)·(1−α(T)) − OLR(T, CO₂) + D·∂/∂x[(1−x²)·∂T/∂x]

where x = sin(φ) ∈ [−1, 1], φ = latitude. One temperature per latitude band;
land and ocean are blended through the band's land fraction. A band is
ice-covered when T < T_ice; ice sets both the albedo (α_ice) and the heat
capacity (C_ice) of the band.

Insolation Q(x, t):
  - Seasonal mode (seasonal=true; all FILLET runs): the daily-mean insolation of
    Berger (1978) is averaged over each time step and tabulated once per run
    (n × steps_per_orbit). The table integrates to the exact orbit-mean forcing
    at any step count. Runs to a periodic (limit-cycle) equilibrium; outputs are
    means over the final orbit.
  - Annual-mean mode (seasonal=false): the orbit mean of the same table. Runs to
    a fixed point.

OLR(T, CO₂) = (A − F_CO₂·ln(CO₂/CO₂_ref)) + B·T

Time discretization: IMEX — implicit for the linear terms (OLR slope +
diffusion), explicit for the albedo and the heat capacity. The tridiagonal
system is solved every step with the Thomas algorithm (O(n)), so the heat
capacity may change with the ice state at no extra cost.

References:
  Budyko (1969), Sellers (1969), North et al. (1981 Rev. Geophys.)
  Berger (1978, J. Atmos. Sci.) for obliquity-dependent insolation
  Myhre et al. (1998) for CO₂ forcing coefficient

FILLET intercomparison: doi:10.3847/PSJ/acba05 (v1.0), doi:10.3847/PSJ/ae1c3c (v1.1)
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

    # Solar forcing and orbit
    S0::Float64        = 1361.0     # solar constant [W m⁻²]
    obliquity::Float64 = 23.5       # axial tilt [degrees]
    period_days::Float64 = 365.0    # orbital period [days] — length of the seasonal cycle

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
    C_ice::Float64     = 1e7        # applied to a band while it is ice-covered

    # Land fraction per latitudinal band [0–1].
    # Empty vector → FILLET default (0.25 uniform).  Length must equal n when provided.
    land_fraction::Vector{Float64} = Float64[]

    # Surface albedo [FILLET Table 4]
    α_land::Float64    = 0.30
    α_ocean::Float64   = 0.20
    α_ice::Float64     = 0.60
    T_ice::Float64     = 0.0        # [°C] ice threshold: albedo + heat-capacity switch and ice-line diagnostic
    ΔT_ice::Float64    = 1.0        # [K] width of the ice transition (ice fraction ramps 0→1 over T_ice ± ΔT_ice/2); 0 → step

    # Time stepping and convergence
    steps_per_orbit::Int = 366      # dt = period / steps_per_orbit (0.997 d for a 365-day orbit). Keep it EVEN: the
                                    # hemisphere swap is a half-orbit shift of steps_per_orbit/2 steps, so an odd count
                                    # samples the seasonal forcing differently in the two hemispheres (N-S asymmetries
                                    # of up to 0.4 K with 365 steps).
    tol::Float64       = 1e-4       # converged when max |T(t) − T(t − P)| over an orbit < tol [K]
    max_orbits::Int    = 500

    # Mode
    seasonal::Bool     = false      # false → orbit-mean Q (fixed point); true → seasonal cycle (limit cycle)
end

const S_earth = 1361.0   # W m⁻²
const T_WARM  =  30.0    # °C — uniform "warm start" (benchmarks, Exp 1/1a, warm hysteresis branches)
const T_COLD  = -50.0    # °C — uniform "cold start" (Exp 2/2a, cold hysteresis branches)

"""Scale instellation by semi-major axis: S(a) = S⊕/a² [au]."""
S0_from_au(a::Real) = S_earth / a^2

"""Orbital period [days] from semi-major axis [au] (Kepler's third law, solar-mass host)."""
period_from_au(a::Real) = 365.0 * a^1.5

"""Time step [s]."""
timestep(p::Params) = p.period_days * 86400.0 / p.steps_per_orbit

"""Copy `p` with the given fields replaced."""
function with_params(p::Params; kwargs...)
    kw = Dict{Symbol,Any}(f => getfield(p, f) for f in fieldnames(Params))
    for (k, v) in kwargs
        kw[k] = v
    end
    return Params(; kw...)
end

# ============================================================
# Grid
# ============================================================

# Cell-centered sin-latitude grid: nodes at band midpoints x_i = −1 + (i−½)·dx,
# dx = 2/n. Equal-area cells with no node ON the poles — this avoids the
# tan(φ)→∞ singularity and makes the simple average sum/n the exact area mean
# (global-mean insolation = S0/4 at every obliquity; diffusion integrates to 0).
make_grid(p::Params) = collect(range(-1.0 + 1.0/p.n, 1.0 - 1.0/p.n, length=p.n))
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

"""Per-band land fraction (length n): `p.land_fraction`, or the FILLET default 0.25."""
function land_fraction_vec(p::Params)
    isempty(p.land_fraction) && return fill(0.25, p.n)
    length(p.land_fraction) == p.n ||
        error("land_fraction length ($(length(p.land_fraction))) must equal n ($(p.n))")
    return p.land_fraction
end

# ============================================================
# Physics functions
# ============================================================

"""
Instantaneous daily-mean insolation [W m⁻²] at solar longitude λ [radians].

Uses the Berger (1978) formula for a single solar declination:
    δ = arcsin(sin(ε)·sin(λ))
    W(φ, δ) = (S₀/π)·[H₀·sin(φ)sin(δ) + cos(φ)cos(δ)·sin(H₀)]
"""
function insolation_instant(x::AbstractVector, p::Params, λ::Real)
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

"""
Cell- and step-averaged insolation table Q[i, s] [W m⁻²]: the mean daily-mean
insolation of band i over time step s of the orbit, i.e. over the band's
sin-latitude extent (nx sub-points per cell) and over solar longitudes
λ ∈ [2π(s−1)/N, 2πs/N] with N = steps_per_orbit (≥ 360/N sub-samples per step).
The orbit mean of the table is therefore the exact annual mean at any step
count, and its area mean is S₀/4 at any obliquity to within the sub-cell
quadrature error (< 0.01 W m⁻²). The orbit starts at λ = 0 (northern spring
equinox).
"""
function insolation_table(x::AbstractVector, p::Params; nx::Int = 8)
    N    = p.steps_per_orbit
    nsub = max(1, cld(360, N))
    n    = length(x)
    dx   = n > 1 ? x[2] - x[1] : 2.0
    xf   = [x[i] + dx * ((j - 0.5) / nx - 0.5) for i in 1:n for j in 1:nx]   # nx sub-points per cell
    Q    = zeros(n, N)
    for s in 1:N, k in 1:nsub
        λ  = 2π * ((s - 1) + (k - 0.5) / nsub) / N
        Qf = insolation_instant(xf, p, λ)
        for i in 1:n
            acc = 0.0
            for j in 1:nx
                acc += Qf[(i - 1) * nx + j]
            end
            Q[i, s] += acc / (nx * nsub)
        end
    end
    return Q
end

"""Orbit-mean (annual-mean) insolation [W m⁻²] — the mean of the step-averaged table."""
insolation(x::AbstractVector, p::Params) = vec(mean(insolation_table(x, p), dims=2))

"""
Ice fraction of a band, 0 (ice-free) to 1 (ice-covered), from its instantaneous
temperature [°C]: a step at T_ice, or, if ΔT_ice > 0, a linear ramp of width
ΔT_ice centered on T_ice.
"""
function ice_fraction(T::Real, p::Params)
    p.ΔT_ice > 0 || return T < p.T_ice ? 1.0 : 0.0
    return clamp((p.T_ice + 0.5 * p.ΔT_ice - T) / p.ΔT_ice, 0.0, 1.0)
end

"""Ice-free albedo of a band with land fraction `fl`."""
albedo_free(fl::Real, p::Params) = fl * p.α_land + (1 - fl) * p.α_ocean

albedo(T::Real, fl::Real, p::Params) = (f = ice_fraction(T, p); f * p.α_ice + (1 - f) * albedo_free(fl, p))
albedo(T::AbstractVector, p::Params)  = albedo.(T, land_fraction_vec(p), Ref(p))

"""Heat capacity [J m⁻² K⁻¹] of a band: C_ice while ice-covered, else the land/ocean blend."""
heat_capacity(T::Real, fl::Real, p::Params) =
    (f = ice_fraction(T, p); f * p.C_ice + (1 - f) * (fl * p.C_land + (1 - fl) * p.C_ocean))
heat_capacity(T::AbstractVector, p::Params) = heat_capacity.(T, land_fraction_vec(p), Ref(p))

"""Effective OLR intercept after CO₂ forcing [W m⁻²]."""
A_eff(p::Params) = p.A - p.F_CO2 * log(p.CO2 / p.CO2_ref)

# ============================================================
# Diffusion operator and IMEX time stepping
# ============================================================

"""Finite-volume diffusion operator L (tridiagonal) with (1−x²) weighting; zero flux at the poles."""
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

"""Work arrays for the IMEX step. The off-diagonals of M = diag(C/dt + B) − L are constant."""
struct Stepper
    L   :: Tridiagonal{Float64, Vector{Float64}}
    dl  :: Vector{Float64}   # sub-diagonal of M   (= −L.dl)
    du  :: Vector{Float64}   # super-diagonal of M (= −L.du)
    d   :: Vector{Float64}   # diagonal of M (rebuilt every step from C(T))
    rhs :: Vector{Float64}
    cp  :: Vector{Float64}
    dp  :: Vector{Float64}
    fl  :: Vector{Float64}   # land fraction per band
    dt  :: Float64
end

function Stepper(x::AbstractVector, p::Params)
    L = build_diffusion_matrix(x, p.D)
    n = length(x)
    return Stepper(L, -L.dl, -L.du, zeros(n), zeros(n), zeros(n), zeros(n),
                   land_fraction_vec(p), timestep(p))
end

"""
Solve the tridiagonal system (dl, d, du)·x = rhs in place (Thomas algorithm).
`dl[i-1]` is the sub-diagonal entry of row i and `du[i]` the super-diagonal entry.
M is strictly diagonally dominant (C/dt + B > 0), so no pivoting is needed.
"""
function thomas!(xout::AbstractVector, dl, d, du, rhs, cp, dp)
    n = length(d)
    @inbounds begin
        cp[1] = du[1] / d[1]
        dp[1] = rhs[1] / d[1]
        for i in 2:n
            m     = d[i] - dl[i-1] * cp[i-1]
            cp[i] = i < n ? du[i] / m : 0.0
            dp[i] = (rhs[i] - dl[i-1] * dp[i-1]) / m
        end
        xout[n] = dp[n]
        for i in n-1:-1:1
            xout[i] = dp[i] - cp[i] * xout[i+1]
        end
    end
    return xout
end

"""
One IMEX step: albedo and heat capacity from the current (instantaneous) T,
then the implicit solve  [C/dt + B − L] T⁺ = C/dt·T + Q(1−α) − A_eff.
"""
function imex_step!(T::AbstractVector, Q::AbstractVector, st::Stepper, p::Params)
    Aeff = A_eff(p)
    dt   = st.dt
    @inbounds for i in eachindex(T)
        f  = ice_fraction(T[i], p)
        fl = st.fl[i]
        α  = f * p.α_ice + (1 - f) * (fl * p.α_land + (1 - fl) * p.α_ocean)
        C  = f * p.C_ice + (1 - f) * (fl * p.C_land + (1 - fl) * p.C_ocean)
        st.d[i]   = C / dt + p.B - st.L.d[i]
        st.rhs[i] = C / dt * T[i] + Q[i] * (1 - α) - Aeff
    end
    thomas!(T, st.dl, st.d, st.du, st.rhs, st.cp, st.dp)
    return T
end

# ============================================================
# Public API
# ============================================================

const MAX_PERIOD = 4   # longest multi-orbit cycle the convergence test recognizes

"""
    run_ebm(p; T0, verbose) → (T, T_orbit, x, p, orbits, dT, converged, period)

Integrate orbit by orbit until the limit cycle (or fixed point) is reached.

Convergence: at every step the state is compared with the state at the same
phase k orbits earlier, k = 1 … MAX_PERIOD. The run converges when, for some k,
the largest such difference over a whole orbit is below `p.tol`; `period` is the
smallest such k. Seasonally forced ice–albedo systems can settle into
period-doubled (biennial) cycles, which a one-orbit test never accepts.
Hitting `p.max_orbits` returns `converged = false` (period 1, last orbit).

`T` is the final instantaneous state (end of the orbit, λ = 0) and `T_orbit`
(n × steps_per_orbit) holds the state at the end of every step of the orbit,
averaged over the `period` orbits of the cycle. `T0` defaults to a uniform T_WARM.
"""
function run_ebm(p::Params;
                 T0::Union{Nothing, AbstractVector} = nothing,
                 verbose::Bool = true)
    x  = make_grid(p)
    T  = isnothing(T0) ? fill(T_WARM, p.n) : float.(copy(T0))
    length(T) == p.n || error("T0 length ($(length(T))) must equal n ($(p.n))")
    st = Stepper(x, p)
    N  = p.steps_per_orbit

    Qtab = insolation_table(x, p)
    p.seasonal || (Qtab .= vec(mean(Qtab, dims=2)))   # annual-mean mode: constant forcing

    hist      = zeros(p.n, N, MAX_PERIOD)   # ring buffer of the last MAX_PERIOD orbits
    dTk       = zeros(MAX_PERIOD)           # max change vs k orbits ago, over the current orbit
    orbits    = 0
    dT        = NaN
    period    = 1
    converged = false

    for orbit in 1:p.max_orbits
        fill!(dTk, 0.0)
        slot  = mod1(orbit, MAX_PERIOD)
        kmax  = min(MAX_PERIOD, orbit - 1)
        for s in 1:N
            imex_step!(T, view(Qtab, :, s), st, p)
            for k in 1:kmax
                ks = mod1(orbit - k, MAX_PERIOD)
                m  = 0.0
                @inbounds for i in 1:p.n
                    m = max(m, abs(T[i] - hist[i, s, ks]))
                end
                dTk[k] = max(dTk[k], m)
            end
            @inbounds for i in 1:p.n
                hist[i, s, slot] = T[i]
            end
        end
        orbits = orbit
        dT     = kmax >= 1 ? dTk[1] : NaN
        for k in 1:kmax
            if dTk[k] < p.tol
                converged = true
                period    = k
                dT        = dTk[k]
                break
            end
        end
        if converged
            verbose && println("  Converged after $orbit orbits" *
                               (period > 1 ? " to a period-$period cycle" : "") *
                               " (max ΔT per cycle = $(round(dT, sigdigits=3)) K).")
            break
        end
        verbose && orbit % 50 == 0 &&
            println("  Orbit $orbit: max ΔT per orbit = $(round(dT, sigdigits=3)) K")
    end
    converged || !verbose ||
        println("  WARNING: not converged after $(p.max_orbits) orbits (max ΔT per orbit = $(round(dT, sigdigits=3)) K)")

    # Orbit-resolved state averaged over the cycle (the last `period` orbits)
    T_orbit = zeros(p.n, N)
    for k in 0:period-1
        T_orbit .+= view(hist, :, :, mod1(orbits - k, MAX_PERIOD))
    end
    T_orbit ./= period

    return (T=T, T_orbit=T_orbit, x=x, p=p, orbits=orbits, dT=dT, converged=converged, period=period)
end

"""
    equilibrium(p; T0, verbose) → (T, α, olr, T_orbit, T_final, x, p, orbits, dT, converged, asym)

Run to equilibrium and return the state averaged over the final orbit:
`T` [°C], `α` and `olr` are orbit means per band (in annual-mean mode they are
the fixed point). `T_final` is the instantaneous end-of-orbit state, used to
continue hysteresis sweeps. `asym` = max |T(φ) − T(−φ)| of the orbit-mean T.
"""
function equilibrium(p::Params;
                     T0::Union{Nothing, AbstractVector} = nothing,
                     verbose::Bool = true)
    r    = run_ebm(p; T0=T0, verbose=verbose)
    Aeff = A_eff(p)
    fl   = land_fraction_vec(p)
    T    = vec(mean(r.T_orbit, dims=2))
    α    = vec(mean(albedo.(r.T_orbit, fl, Ref(p)), dims=2))
    olr  = @. Aeff + p.B * T
    return (T=T, α=α, olr=olr, T_orbit=r.T_orbit, T_final=r.T, x=r.x, p=p,
            orbits=r.orbits, dT=r.dT, converged=r.converged, period=r.period,
            asym=hemispheric_asymmetry(T))
end

# ============================================================
# Diagnostics
# ============================================================

# Global area-weighted mean. The cell-centered make_grid gives equal-area bands
# with no nodes on the poles, so the simple average IS the exact area mean.
global_mean(field::AbstractVector) = sum(field) / length(field)

"""Largest north–south difference of a zonal field, max |f(φ) − f(−φ)|."""
hemispheric_asymmetry(f::AbstractVector) = maximum(abs.(f .- reverse(f)))

"""True when the configuration is mirror-symmetric about the equator (uniform or symmetric land fraction)."""
symmetric_setup(p::Params) = isempty(p.land_fraction) || p.land_fraction ≈ reverse(p.land_fraction)

function fmt_time(s::Float64)
    s = round(Int, s)
    m, s = divrem(s, 60)
    h, m = divrem(m, 60)
    h > 0 ? @sprintf("%dh%02dm%02ds", h, m, s) : @sprintf("%dm%02ds", m, s)
end

"""
Contiguous ice-covered latitude intervals [φ_lo, φ_hi] (degrees, south to north)
of an annual-mean profile. Ice where T < T_ice. Edges are linearly interpolated
in latitude between cell centers; an interval that includes the polemost cell
extends to ±90.
"""
function ice_segments(T::AbstractVector, x::AbstractVector, p::Params)
    φ = lat_deg(x)
    f = T .- p.T_ice
    n = length(T)
    segs = Tuple{Float64,Float64}[]
    i = 1
    while i <= n
        if f[i] < 0
            lo = i == 1 ? -90.0 : φ[i-1] + (φ[i] - φ[i-1]) * f[i-1] / (f[i-1] - f[i])
            j = i
            while j < n && f[j+1] < 0
                j += 1
            end
            hi = j == n ? 90.0 : φ[j] + (φ[j+1] - φ[j]) * f[j] / (f[j] - f[j+1])
            push!(segs, (lo, hi))
            i = j + 1
        else
            i += 1
        end
    end
    return segs
end

"""
Ice-edge latitudes per the FILLET v1.1 template convention.

  NH_max — poleward extent of NH ice (90 if the pole is ice-covered)
  NH_min — equatorward extent of NH ice (0 if ice reaches the equator)
  SH_max — equatorward extent of SH ice (0 if ice reaches the equator)
  SH_min — poleward extent of SH ice (−90 if the pole is ice-covered)

Ice-free: NH → (90, 90), SH → (−90, −90).  Snowball: NH → (90, 0), SH → (0, −90).
Polar cap with its edge at φ: NH → (90, φ), SH → (−φ, −90).
Belt: (poleward edge, equatorward edge). When a cap and a belt coexist the cap
is reported and `multiple` is true; `segments` lists every interval.
"""
function ice_edges(T::AbstractVector, x::AbstractVector, p::Params)
    segs = ice_segments(T, x, p)
    nh = [(max(lo, 0.0), hi)  for (lo, hi) in segs if hi > 0]
    sh = [(lo, min(hi, 0.0))  for (lo, hi) in segs if lo < 0]

    if isempty(nh)
        nh_max = 90.0; nh_min = 90.0
    else
        caps = filter(s -> s[2] >= 90.0, nh)
        if !isempty(caps)
            nh_max = 90.0;                 nh_min = caps[1][1]
        else
            nh_max = maximum(last, nh);    nh_min = minimum(first, nh)
        end
    end

    if isempty(sh)
        sh_max = -90.0; sh_min = -90.0
    else
        caps = filter(s -> s[1] <= -90.0, sh)
        if !isempty(caps)
            sh_min = -90.0;                sh_max = caps[1][2]
        else
            sh_max = maximum(last, sh);    sh_min = minimum(first, sh)
        end
    end

    return (NH_max=nh_max, NH_min=nh_min, SH_max=sh_max, SH_min=sh_min,
            segments=segs, multiple=(length(nh) > 1 || length(sh) > 1))
end

"""NH ice-edge latitude [degrees] — equatorward extent of NH ice; NaN if ice-free, 0 if snowball."""
function ice_edge_NH(T::AbstractVector, x::AbstractVector, p::Params)
    e = ice_edges(T, x, p)
    return e.NH_min == 90.0 ? NaN : e.NH_min
end

"""Format NH ice-edge for display: "ice-free" if no ice, otherwise "XX.X°"."""
fmt_ice(T, x, p) = let e = ice_edge_NH(T, x, p)
    isnan(e) ? "ice-free" : @sprintf("%.1f°", e)
end

"""Format an ice-segment list, e.g. "[-90.0,-64.2];[63.9,90.0]"."""
fmt_segments(segs) = isempty(segs) ? "none" : join([@sprintf("[%.1f,%.1f]", lo, hi) for (lo, hi) in segs], ";")

"""
Global energy budget. Returns (SW_absorbed, OLR, imbalance) in W m⁻².
Uses the orbit-mean Q and the provided α (orbit-mean α in seasonal mode).
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
const GLOBAL_COLUMNS = "Case Inst Obl XCO2 Tglob IceLineNMaxLand IceLineNMinLand IceLineNMaxSea IceLineNMinSea IceLineSMaxLand IceLineSMinLand IceLineSMaxSea IceLineSMinSea Diff OLRglob"

"""
Per-latitude profile in FILLET v1.1 format.

`α_mean` and `olr_mean` should be orbit-mean fields in seasonal mode.
TOA albedo equals surface albedo (no atmospheric scattering in this model).
"""
function fillet_profile(T::AbstractVector, x::AbstractVector, p::Params;
                        α_mean::Union{Nothing, AbstractVector}   = nothing,
                        olr_mean::Union{Nothing, AbstractVector} = nothing)
    α   = isnothing(α_mean)   ? albedo(T, p)             : α_mean
    olr = isnothing(olr_mean) ? @.(A_eff(p) + p.B * T)   : olr_mean
    fl  = land_fraction_vec(p)
    return (lat=lat_deg(x), Tsurf=T .+ K_OFFSET, Asurf=α, ATOA=α, OLR=olr, Fland=fl)
end

"""
Global scalar outputs in FILLET v1.1 template format (see GLOBAL_COLUMNS).

AVALON has a single temperature per band (land fraction blended), so the land
and sea ice lines are identical; both sets of columns carry the same values.
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
        multiple    = edges.multiple,
        segments    = edges.segments,
    )
end

"""One data row of a global_output file (fixed precision; Inst resolves the 0.0125 grid)."""
global_row(g) = @sprintf("%d %.6f %.1f %.6g %.4f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.4f %.4f",
                         g.Case, g.Inst, g.Obl, g.XCO2, g.Tglob,
                         g.IceLineNMaxLand, g.IceLineNMinLand, g.IceLineNMaxSea, g.IceLineNMinSea,
                         g.IceLineSMaxLand, g.IceLineSMinLand, g.IceLineSMaxSea, g.IceLineSMinSea,
                         g.Diff, g.OLRglob)

"""Header lines describing the model configuration (shared by lat and global files)."""
function config_notes(p::Params)
    lf  = isempty(p.land_fraction) ? "0.25 uniform (FILLET Table 4)" :
          @sprintf("latitude-dependent Earth profile (area mean %.3f)", global_mean(land_fraction_vec(p)))
    ice = p.ΔT_ice > 0 ?
          @sprintf("linear transition of width %.2f K centered on %.2f C", p.ΔT_ice, p.T_ice) :
          @sprintf("step at %.2f C", p.T_ice)
    return [
        "# Mode: " * (p.seasonal ? "seasonal (limit cycle; values are means over the final orbit)" :
                                   "orbit-mean insolation (fixed point)"),
        @sprintf("# Orbital period (days): %.4f; steps per orbit: %d (dt = %.4f days%s); grid: %d equal-area cells in sin(lat), cell-centered, no node on the poles",
                 p.period_days, p.steps_per_orbit, p.period_days / p.steps_per_orbit,
                 iseven(p.steps_per_orbit) ? "; even count, so the sampled seasonal forcing is mirror-symmetric between hemispheres" : "", p.n),
        @sprintf("# Albedo land/ocean/ice: %.4f/%.4f/%.4f; heat capacity land/ocean/ice: %.3g/%.3g/%.3g J m^-2 K^-1; D = %.4f W m^-2 K^-1",
                 p.α_land, p.α_ocean, p.α_ice, p.C_land, p.C_ocean, p.C_ice, p.D),
        @sprintf("# OLR = A + B*T - F*ln(CO2/%g ppm), T in C: A = %.2f W m^-2, B = %.3f W m^-2 K^-1, F = %.2f W m^-2",
                 p.CO2_ref, p.A, p.B, p.F_CO2),
        "# Ice: " * ice * "; an ice-covered band takes the ice albedo and the ice heat capacity (from the instantaneous temperature)",
        "# Land fraction: " * lf * "; one temperature per band (land and ocean blended in albedo and heat capacity)",
    ]
end

function ice_line_notes(p::Params)
    return [
        @sprintf("# Describe how ice line latitude is determined: latitude where the annual-mean surface temperature crosses %.2f C, linearly interpolated in latitude between cell centers. Land and Sea columns are identical (one temperature per band).", p.T_ice),
        "# Conventions: ice-free NMax=NMin=90, SMax=SMin=-90; snowball NMax=90 NMin=0 SMax=0 SMin=-90; polar cap NMax=90 NMin=edge (SMax=-edge SMin=-90); belt NMax/NMin = poleward/equatorward edges. If a cap and a belt coexist the cap is reported.",
    ]
end

function run_notes(r, p::Params)
    cyc = r.period > 1 ?
        @sprintf(" to a period-%d cycle (values are means over the %d-orbit cycle); max |T(t) - T(t - %dP)|", r.period, r.period, r.period) :
        "; max |T(t) - T(t - P)|"
    return [
        @sprintf("# Convergence: %s after %d orbits%s over the final orbit = %.2e K (tolerance %.1e K)",
                 r.converged ? "converged" : "NOT CONVERGED", r.orbits, cyc, r.dT, p.tol),
        @sprintf("# Hemispheric symmetry: max |T(lat) - T(-lat)| of the annual mean = %.4f K%s",
                 r.asym, symmetric_setup(p) ? "" : " (land fraction is asymmetric, so asymmetry is expected)"),
    ]
end

start_note(T0_val, kind) = @sprintf("# Initial state: uniform %.1f C at all latitudes (%s start), integration starts at northern spring equinox", T0_val, kind)

function lat_header(label::String, case::Int, inst::Float64, p::Params; branch=nothing)
    h = ["# Name of benchmark/experiment: $label",
         "# Code: AVALON (Albedo-feedback Variable Axial-tilt Latitudinal Outgoing-Net EBM)",
         "# Case number: $case"]
    branch === nothing || push!(h, "# Branch: $branch")
    push!(h, @sprintf("# Instellation (S_earth): %.6f", inst))
    push!(h, @sprintf("# XCO2 (ppm): %.6g", p.CO2))
    push!(h, @sprintf("# Obliquity (degrees): %.2f", p.obliquity))
    append!(h, config_notes(p))
    return h
end

function global_header(label::String, p::Params; extra::Vector{String} = String[], branch_col::Bool = false)
    h = ["# Name of benchmark/experiment: $label",
         "# Code: AVALON (Albedo-feedback Variable Axial-tilt Latitudinal Outgoing-Net EBM)"]
    append!(h, ice_line_notes(p))
    append!(h, config_notes(p))
    append!(h, extra)
    push!(h, "#")
    push!(h, "# Columns: Case = case number; Inst = instellation (S_earth = 1361 W m^-2); Obl = obliquity (deg); XCO2 = CO2 mixing ratio (ppm); Tglob = area-weighted annual mean surface temperature (K); IceLine* = ice edges (deg, see above); Diff = diffusion coefficient used (W m^-2 K^-1); OLRglob = area-weighted annual mean OLR (W m^-2)" *
             (branch_col ? "; Branch = warm (warm start, decreasing sweep) or cold (cold start, increasing sweep)" : ""))
    push!(h, "# " * GLOBAL_COLUMNS * (branch_col ? " Branch" : ""))
    return h
end

"""Write a lat_output file: header lines, then the profile (with the Fland column unless `fland=false`)."""
function write_lat_file(path::String, prof, header::Vector{String}; fland::Bool = true)
    open(path, "w") do io
        foreach(l -> println(io, l), header)
        println(io, "#")
        println(io, "# Columns of data (annually averaged for last orbit)")
        println(io, "# Lat = latitude (deg); Tsurf = surface temperature (K); Asurf = surface albedo (unweighted time mean over the orbit); ATOA = top-of-atmosphere albedo (= Asurf, no atmospheric scattering); OLR = outgoing longwave radiation (W m^-2)" *
                    (fland ? "; Fland = land fraction of the band" : ""))
        println(io, fland ? "# Lat Tsurf Asurf ATOA OLR Fland" : "# Lat Tsurf Asurf ATOA OLR")
        for i in eachindex(prof.lat)
            if fland
                @printf(io, "%7.2f %8.2f %6.4f %6.4f %8.2f %6.4f\n",
                        prof.lat[i], prof.Tsurf[i], prof.Asurf[i], prof.ATOA[i], prof.OLR[i], prof.Fland[i])
            else
                @printf(io, "%7.2f %8.2f %6.4f %6.4f %8.2f\n",
                        prof.lat[i], prof.Tsurf[i], prof.Asurf[i], prof.ATOA[i], prof.OLR[i])
            end
        end
    end
end

"""Final-orbit temperature snapshots [K]: one row per step, first column = day of orbit at the end of the step."""
function write_seasonal_csv(path::String, r)
    N       = size(r.T_orbit, 2)
    dt_days = r.p.period_days / N
    open(path, "w") do io
        println(io, "day," * join(string.(round.(lat_deg(r.x), digits=4)), ","))
        for s in 1:N
            println(io, string(round(s * dt_days, digits=4)) * "," *
                        join(string.(round.(r.T_orbit[:, s] .+ K_OFFSET, digits=4)), ","))
        end
    end
end

"""Append one line to a convergence log and print warnings for non-converged or belt-plus-cap cases."""
function log_case!(log::IO, prefix::String, r, p::Params)
    e = ice_edges(r.T, r.x, p)
    @printf(log, "%s %d %s %d %.3e %.4f %s\n", prefix, r.orbits, r.converged ? "yes" : "NO", r.period, r.dT, r.asym, fmt_segments(e.segments))
    flush(log)
    r.converged || println("  WARNING: $prefix — not converged after $(r.orbits) orbits (max ΔT per orbit = $(round(r.dT, sigdigits=3)) K)")
    r.period > 1 && println("  NOTE: $prefix — period-$(r.period) cycle; values are means over the cycle")
    e.multiple  && println("  NOTE: $prefix — ice cap and belt coexist $(fmt_segments(e.segments)); cap reported in the ice-line columns")
end

"""
Write FILLET-format .dat files for one equilibrium state `r` (from `equilibrium`) into `outdir/`.

Creates:
  `{outdir}/lat_output_AVALON_{tag}.dat`    — per-latitude profile
  `{outdir}/global_output_AVALON_{tag}.dat` — global scalar row
  `{outdir}/{tag}_seasonal.csv`             — final-orbit snapshots (seasonal mode)
"""
function write_fillet_output(r, outdir::String, tag::String;
                             instellation::Float64 = r.p.S0 / S_earth,
                             case::Int = 0,
                             label::String = tag,
                             extra_global::Vector{String} = String[])
    p = r.p
    mkpath(outdir)
    prof = fillet_profile(r.T, r.x, p; α_mean=r.α, olr_mean=r.olr)
    write_lat_file(joinpath(outdir, "lat_output_AVALON_$(tag).dat"), prof,
                   vcat(lat_header(label, case, instellation, p), run_notes(r, p)))
    g = fillet_global(r.T, r.x, p; instellation=instellation, case=case, α_mean=r.α, olr_mean=r.olr)
    open(joinpath(outdir, "global_output_AVALON_$(tag).dat"), "w") do io
        foreach(l -> println(io, l), global_header(label, p; extra=vcat(extra_global, run_notes(r, p))))
        println(io, global_row(g))
    end
    p.seasonal && write_seasonal_csv(joinpath(outdir, "$(tag)_seasonal.csv"), r)
    return g
end

# ============================================================
# Experiment runners
# ============================================================

function progress_line(k, n_total, obl, sf, g, r, t_start)
    elapsed = time() - t_start
    eta     = k < n_total ? "  ETA $(fmt_time(elapsed / k * (n_total - k)))" : ""
    ice     = g.IceLineNMinSea == 90.0 ? " free " : @sprintf("%5.1f°", g.IceLineNMinSea)
    @printf("  [%3d/%d] S⊕=%.4f obl=%2.0f°  Tglob=%6.1f K  ice=%s  %3d orbits%s%s  %s elapsed%s\n",
            k, n_total, sf, obl, g.Tglob, ice, r.orbits, r.converged ? "" : " (NOT CONVERGED)", r.period > 1 ? " (period $(r.period))" : "",
            fmt_time(elapsed), eta)
end

"""
Run the FILLET instellation × obliquity sweep (Experiments 1 / 2).

`S0_factors`  — instellation values as multiples of S⊕ (outer loop)
`obliquities` — obliquity values [degrees] (inner loop)
`periods`     — optional orbital period [days] per instellation value (Experiments 1a / 2a)
`warm_start`  — true → warm start (uniform T_WARM), false → cold start (uniform T_COLD)
`outdir`      — output subdirectory; tag is derived from basename(outdir)

Writes per-case `lat_output_AVALON_{tag}_{case}.dat`, one `global_output_AVALON_{tag}.dat`
and a `convergence.log` inside `outdir/`.
"""
function run_fillet_sweep(S0_factors::AbstractVector, obliquities::AbstractVector;
                          periods::Union{Nothing, AbstractVector} = nothing,
                          warm_start::Bool  = true,
                          p_base::Params    = Params(),
                          outdir::String    = joinpath("experiments", warm_start ? "exp1" : "exp2"),
                          label::String     = warm_start ? "FILLET Experiment 1 (warm start)" :
                                                           "FILLET Experiment 2 (cold start)",
                          verbose::Bool     = true)
    tag     = basename(outdir)
    T0_val  = warm_start ? T_WARM : T_COLD
    cases   = [(Float64(sf), Float64(obl), isnothing(periods) ? p_base.period_days : Float64(periods[k]))
               for (k, sf) in enumerate(S0_factors) for obl in obliquities]
    n_total = length(cases)
    rows    = NamedTuple[]
    t_start = time()

    mkpath(outdir)
    extra = [start_note(T0_val, warm_start ? "warm" : "cold"),
             "# Case order: instellation outermost, obliquity innermost",
             isnothing(periods) ? @sprintf("# Orbital period: %.2f days for every case", p_base.period_days) :
                                  "# Orbital period: 365 d x a^1.5 with a = Inst^(-1/2) au (Kepler's third law); the period of each case is in its lat file header",
             "# Per-case convergence and hemispheric-symmetry diagnostics are kept with the AVALON repository (convergence.log and per-case lat files)"]

    header = global_header(label, p_base; extra=extra)
    if !isnothing(periods)   # the config line would otherwise print the base period; state the per-case range instead
        header = map(header) do l
            startswith(l, "# Orbital period (days):") ?
                @sprintf("# Orbital period (days): %.4f to %.4f, set per case as 365 d x a^1.5 (see below); steps per orbit: %d (dt = period/%d%s); grid: %d equal-area cells in sin(lat), cell-centered, no node on the poles",
                         minimum(periods), maximum(periods), p_base.steps_per_orbit, p_base.steps_per_orbit,
                         iseven(p_base.steps_per_orbit) ? "; even count, so the sampled seasonal forcing is mirror-symmetric between hemispheres" : "", p_base.n) : l
        end
    end
    open(joinpath(outdir, "global_output_AVALON_$(tag).dat"), "w") do io
        foreach(l -> println(io, l), header)
        open(joinpath(outdir, "convergence.log"), "w") do log
            println(log, "# case inst obl period_days orbits converged cycle_orbits max_dT_K asym_K ice_segments")
            for (k, (sf, obl, per)) in enumerate(cases)
                p = with_params(p_base; S0=sf * S_earth, obliquity=obl, period_days=per)
                r = equilibrium(p; T0=fill(T0_val, p.n), verbose=false)
                g = fillet_global(r.T, r.x, p; instellation=sf, case=k - 1, α_mean=r.α, olr_mean=r.olr)
                push!(rows, g)
                println(io, global_row(g))
                flush(io)
                prof = fillet_profile(r.T, r.x, p; α_mean=r.α, olr_mean=r.olr)
                write_lat_file(joinpath(outdir, "lat_output_AVALON_$(tag)_$(k-1).dat"), prof,
                               vcat(lat_header(label, k - 1, sf, p), run_notes(r, p)))
                log_case!(log, @sprintf("%d %.6f %.1f %.4f", k - 1, sf, obl, per), r, p)
                verbose && progress_line(k, n_total, obl, sf, g, r, t_start)
            end
        end
    end
    verbose && println("  Done — $(n_total) cases written to $(outdir)/")
    return rows
end

"""
Run Experiments 1a / 2a: obliquity × semi-major axis sweep.
Instellation S(a) = S⊕/a² and orbital period P(a) = 365 d · a^1.5.
"""
function run_fillet_sweep_au(a_range::AbstractVector, obliquities::AbstractVector;
                              warm_start::Bool = true,
                              p_base::Params   = Params(),
                              outdir::String   = joinpath("experiments", warm_start ? "exp1a" : "exp2a"),
                              label::String    = warm_start ? "FILLET Experiment 1a (warm start, semi-major axis)" :
                                                              "FILLET Experiment 2a (cold start, semi-major axis)",
                              verbose::Bool    = true)
    a = collect(Float64, a_range)
    return run_fillet_sweep(S0_from_au.(a) ./ S_earth, obliquities;
                            periods=period_from_au.(a), warm_start=warm_start, p_base=p_base,
                            outdir=outdir, label=label, verbose=verbose)
end

"""
    hysteresis_sweep(vals, field; p_base, outdir, label, notes, verbose) → (vals, mean_T, ice_lat, branch)

Two continuation sweeps of `field` (:S0 or :CO2) over `vals` (FILLET Experiments 3 / 4):
  "warm" branch — warm start (uniform T_WARM), values decreasing;
  "cold" branch — cold start (uniform T_COLD), values increasing.
Each case starts from the final instantaneous state of the previous case.
With `outdir`, writes `global_output_AVALON_{tag}.dat` (Branch column appended),
per-case lat files and `convergence.log`.
"""
function hysteresis_sweep(vals::AbstractVector, field::Symbol;
                          p_base::Params = Params(),
                          outdir::Union{Nothing, String} = nothing,
                          label::String = "hysteresis",
                          notes::Vector{String} = String[],
                          verbose::Bool = true)
    field in (:S0, :CO2) || error("field must be :S0 or :CO2")
    v_sorted = sort(collect(Float64, vals))
    out = (vals=Float64[], mean_T=Float64[], ice_lat=Float64[], branch=String[])
    tag = isnothing(outdir) ? "" : basename(outdir)
    isnothing(outdir) || mkpath(outdir)
    io  = isnothing(outdir) ? nothing : open(joinpath(outdir, "global_output_AVALON_$(tag).dat"), "w")
    log = isnothing(outdir) ? nothing : open(joinpath(outdir, "convergence.log"), "w")
    try
        if io !== nothing
            extra = vcat(notes, [
                @sprintf("# Branches: warm = warm start (uniform %.1f C) sweeping %s downward; cold = cold start (uniform %.1f C) sweeping upward. Each case starts from the final state of the previous case (continuation).",
                         T_WARM, String(field), T_COLD),
                "# Per-case convergence and hemispheric-symmetry diagnostics are kept with the AVALON repository (convergence.log and per-case lat files)"])
            foreach(l -> println(io, l), global_header(label, p_base; extra=extra, branch_col=true))
            println(log, "# case branch value orbits converged cycle_orbits max_dT_K asym_K ice_segments")
        end
        case = 0
        for (branch_vals, branch, T0_val) in ((reverse(v_sorted), "warm", T_WARM), (v_sorted, "cold", T_COLD))
            T_prev = fill(T0_val, p_base.n)
            verbose && print(branch == "warm" ? "Warm start, decreasing: " : "Cold start, increasing: ")
            for v in branch_vals
                p = with_params(p_base; (field => v,)...)
                r = equilibrium(p; T0=T_prev, verbose=false)
                push!(out.vals, v)
                push!(out.mean_T, global_mean(r.T))
                push!(out.ice_lat, ice_edge_NH(r.T, r.x, p))
                push!(out.branch, branch)
                T_prev = r.T_final
                verbose && print(branch == "warm" ? "-" : "+")
                if io !== nothing
                    g = fillet_global(r.T, r.x, p; instellation=p.S0 / S_earth, case=case, α_mean=r.α, olr_mean=r.olr)
                    println(io, global_row(g) * " " * branch)
                    flush(io)
                    prof = fillet_profile(r.T, r.x, p; α_mean=r.α, olr_mean=r.olr)
                    write_lat_file(joinpath(outdir, "lat_output_AVALON_$(tag)_$(case).dat"), prof,
                                   vcat(lat_header(label, case, p.S0 / S_earth, p; branch=branch), run_notes(r, p)))
                    log_case!(log, @sprintf("%d %s %.6g", case, branch, v), r, p)
                end
                case += 1
            end
            verbose && println()
        end
    finally
        io  === nothing || close(io)
        log === nothing || close(log)
    end
    return out
end


"""Snowball hysteresis in instellation (FILLET Experiment 3). See `hysteresis_sweep`."""
bifurcation_diagram(S0_range::AbstractVector; kwargs...) = hysteresis_sweep(S0_range, :S0; kwargs...)

"""Snowball hysteresis in CO₂ (FILLET Experiment 4). See `hysteresis_sweep`."""
co2_bifurcation(CO2_range::AbstractVector; kwargs...) = hysteresis_sweep(CO2_range, :CO2; kwargs...)

# ============================================================
# Benchmark 1 configuration and tuning
# ============================================================

# Tuned so that Benchmark 1 gives Tglob = 288.0 K (julia avalon.jl tune_ben1).
const α_OCEAN_BEN1 = 0.25676

"""
Benchmark 1: FILLET Table 4 values except the tuned quantities — D = 0.52,
ocean albedo α_OCEAN_BEN1, C_ocean = 2e8 (50 m mixed layer) and an Earth-like
latitude-dependent land fraction. Seasonal mode.
"""
ben1_params(; α_ocean::Float64 = α_OCEAN_BEN1) =
    Params(D=0.52, α_ocean=α_ocean, C_ocean=2e8,
           land_fraction=earth_land_fraction(make_grid(Params())), seasonal=true)

"""Bisection on the Benchmark 1 ocean albedo so that Tglob = `target` K (warm start)."""
function tune_ben1(; target::Float64 = 288.0, lo::Float64 = 0.05, hi::Float64 = 0.45,
                     atol::Float64 = 0.002, verbose::Bool = true)
    f(α) = global_mean(equilibrium(ben1_params(α_ocean=α); T0=fill(T_WARM, Params().n), verbose=false).T) + K_OFFSET - target
    flo, fhi = f(lo), f(hi)
    flo > 0 > fhi || error("target not bracketed: Tglob(α=$lo) = $(flo + target) K, Tglob(α=$hi) = $(fhi + target) K")
    α = 0.5 * (lo + hi)
    while true
        α  = 0.5 * (lo + hi)
        fα = f(α)
        verbose && @printf("  α_ocean = %.5f → Tglob = %.4f K\n", α, fα + target)
        (abs(fα) < atol || hi - lo < 1e-5) && break
        fα > 0 ? (lo = α) : (hi = α)
    end
    return α
end

# ============================================================
# FILLET archive export
# ============================================================

"""
    export_fillet(; src="experiments", dst="experiments/fillet_submission/avalon")

Assemble the layout required by projectcuisines/fillet (Results/README.md) from
the files under `src`:

  ben1, ben2, ben3          global_output.dat + case_0/lat_output.dat (template columns, no Fland)
  exp1, exp1a, exp2, exp2a  global_output.dat
  exp3_warm, exp3_cold,     global_output.dat — the Branch column split into the two
  exp4_warm, exp4_cold      directories, cases renumbered from 0 within each file

Errors if any source file is missing.
"""
function export_fillet(; src::String = "experiments",
                         dst::String = joinpath("experiments", "fillet_submission", "avalon"))
    read_lines(path) = (isfile(path) || error("missing $path — run the corresponding command first"); readlines(path))
    written = String[]
    function put(relpath, lines)
        path = joinpath(dst, relpath)
        mkpath(dirname(path))
        open(path, "w") do io
            foreach(l -> println(io, l), lines)
        end
        push!(written, relpath)
    end

    for ben in ("ben1", "ben2", "ben3")
        put(joinpath(ben, "global_output.dat"),
            read_lines(joinpath(src, ben, "global_output_AVALON_$(ben).dat")))
        out = String[]
        for l in read_lines(joinpath(src, ben, "lat_output_AVALON_$(ben).dat"))
            if startswith(l, "#")
                l == "# Lat Tsurf Asurf ATOA OLR Fland" && (l = "# Lat Tsurf Asurf ATOA OLR")
                push!(out, replace(l, "; Fland = land fraction of the band" => ""))
            elseif !isempty(strip(l))
                c = parse.(Float64, split(l))
                push!(out, @sprintf("%7.2f %8.2f %6.4f %6.4f %8.2f", c[1], c[2], c[3], c[4], c[5]))
            end
        end
        put(joinpath(ben, "case_0", "lat_output.dat"), out)
    end

    for exp in ("exp1", "exp1a", "exp2", "exp2a")
        put(joinpath(exp, "global_output.dat"),
            read_lines(joinpath(src, exp, "global_output_AVALON_$(exp).dat")))
    end

    for exp in ("exp3", "exp4")
        lines  = read_lines(joinpath(src, exp, "global_output_AVALON_$(exp).dat"))
        header = [l for l in lines if startswith(l, "#")]
        rows   = [split(l) for l in lines if !startswith(l, "#") && !isempty(strip(l))]
        for branch in ("warm", "cold")
            hdr = map(header) do l
                startswith(l, "# Case Inst") ? replace(l, r" Branch$" => "") :
                startswith(l, "# Columns:")  ? replace(l, "; Branch = warm (warm start, decreasing sweep) or cold (cold start, increasing sweep)" => "") :
                l
            end
            colidx = findlast(l -> startswith(l, "# Case Inst"), hdr)
            insert!(hdr, colidx, branch == "warm" ?
                @sprintf("# This file: warm-start branch (uniform %.1f C initial state), continuation sweep with the varied parameter decreasing", T_WARM) :
                @sprintf("# This file: cold-start branch (uniform %.1f C initial state), continuation sweep with the varied parameter increasing", T_COLD))
            out = copy(hdr)
            k = 0
            for c in rows
                c[end] == branch || continue
                push!(out, string(k) * " " * join(c[2:end-1], " "))
                k += 1
            end
            put(joinpath("$(exp)_$(branch)", "global_output.dat"), out)
        end
    end

    println("Exported $(length(written)) files to $dst/")
    foreach(f -> println("  ", f), written)
    return written
end

# ============================================================
# CLI
# ============================================================

function main()
    print(HELP_TEXT)
end

"""Print the summary line for a single-case run."""
function report(g, r, outdir::String, tag::String)
    println("Tglob = $(round(g.Tglob, digits=2)) K  |  ice edge: $(fmt_ice(r.T, r.x, r.p))  |  " *
            "$(r.converged ? "converged" : "NOT CONVERGED") after $(r.orbits) orbits" *
            "$(r.period > 1 ? " (period-$(r.period) cycle)" : "")  |  " *
            "N-S asymmetry $(round(r.asym, digits=3)) K")
    println("→ $(outdir)/lat_output_AVALON_$(tag).dat, $(outdir)/global_output_AVALON_$(tag).dat" *
            (r.p.seasonal ? ", $(outdir)/$(tag)_seasonal.csv" : ""))
end

"""
Run a single arbitrary case from `key=value` command-line arguments.

Special keys (not Params fields):
  out=tag   — output directory/tag (default: "run")
  au=value  — set S0 and the orbital period from the semi-major axis [au]
  T0=value  — uniform initial temperature [°C] (default: T_WARM)

All other keys must be valid `Params` field names. ASCII aliases:
`alpha_land`, `alpha_ocean`, `alpha_ice`, `dT_ice`.
"""
function run_custom(args::Vector{String})
    aliases = Dict("alpha_land" => "α_land", "alpha_ocean" => "α_ocean", "alpha_ice" => "α_ice",
                   "dT_ice" => "ΔT_ice")
    kw  = Dict{Symbol,Any}(f => getfield(Params(), f) for f in fieldnames(Params))
    tag = "run"
    au  = nothing
    T0  = T_WARM

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
        elseif k == "T0"
            T0 = parse(Float64, v)
        else
            sym = Symbol(k)
            if sym ∉ fieldnames(Params)
                println(stderr, "Error: unknown parameter '$k'")
                println(stderr, "  Valid Params fields: $(join(fieldnames(Params), ", "))")
                println(stderr, "  ASCII aliases: alpha_land, alpha_ocean, alpha_ice, dT_ice")
                exit(1)
            end
            T = fieldtype(Params, sym)
            kw[sym] = T == Bool  ? (v == "true" || v == "1") :
                      T == Int   ? parse(Int, v) :
                                   parse(Float64, v)
        end
    end

    if !isnothing(au)
        kw[:S0]          = S0_from_au(au)
        kw[:period_days] = period_from_au(au)
    end
    p = Params(; kw...)

    # Print non-default parameters
    defaults = Params()
    changed  = [string(f, "=", getfield(p, f))
                for f in fieldnames(Params) if getfield(p, f) != getfield(defaults, f)]
    isempty(changed) ? println("Running with all default parameters.") :
                       println("Parameters: $(join(changed, "  "))")

    outdir = joinpath("experiments", tag)
    r = equilibrium(p; T0=fill(T0, p.n), verbose=true)
    g = write_fillet_output(r, outdir, tag; instellation=p.S0 / S_earth, case=0,
                            label="AVALON custom: $tag",
                            extra_global=[start_note(T0, T0 > 0 ? "warm" : "cold")])
    report(g, r, outdir, tag)
end

"""
Run a named FILLET benchmark or experiment and write the official .dat output files.

    julia avalon.jl benchmark1            # Benchmark 1: tuned pre-industrial Earth
    julia avalon.jl benchmark2            # Benchmark 2: un-tuned, ε = 23.5°
    julia avalon.jl benchmark3            # Benchmark 3: un-tuned, ε = 60°
    julia avalon.jl exp1 | exp2           # instellation × obliquity sweeps (warm / cold start)
    julia avalon.jl exp1a | exp2a         # semi-major axis × obliquity sweeps (period varies)
    julia avalon.jl exp3 [base=ben1]      # instellation hysteresis (default base: Benchmark 2 configuration)
    julia avalon.jl exp4 [base=ben1]      # CO₂ hysteresis
    julia avalon.jl export                # assemble the FILLET Results/ layout from experiments/
    julia avalon.jl tune_ben1             # re-tune the Benchmark 1 ocean albedo
"""
function run_cli(cmd::String, extra_args::Vector{String} = String[])
    opts = Dict{String,String}()
    for a in extra_args
        kv = split(a, "="; limit=2)
        length(kv) == 2 && (opts[String(kv[1])] = String(kv[2]))
    end

    if cmd == "benchmark1"
        p = ben1_params()
        println("FILLET Benchmark 1 (tuned pre-industrial Earth: D=$(p.D), α_ocean=$(p.α_ocean), C_ocean=$(p.C_ocean), Earth land fraction, seasonal)")
        r = equilibrium(p; T0=fill(T_WARM, p.n), verbose=true)
        g = write_fillet_output(r, joinpath("experiments", "ben1"), "ben1";
                                instellation=1.0, case=0, label="FILLET Benchmark 1",
                                extra_global=[start_note(T_WARM, "warm"),
                                              @sprintf("# Tuning: alpha_ocean = %.5f, D = %.2f, C_ocean = %.3g (50 m mixed layer), Earth-like land fraction; target Tglob = 288 K (julia avalon.jl tune_ben1)", p.α_ocean, p.D, p.C_ocean)])
        report(g, r, "experiments/ben1", "ben1")

    elseif cmd == "benchmark2" || cmd == "benchmark3"
        obl = cmd == "benchmark2" ? 23.5 : 60.0
        tag = cmd == "benchmark2" ? "ben2" : "ben3"
        println("FILLET Benchmark $(tag[end]) (un-tuned FILLET Table 4 parameters, ε=$(obl)°, seasonal)")
        p = Params(obliquity=obl, seasonal=true)
        r = equilibrium(p; T0=fill(T_WARM, p.n), verbose=true)
        g = write_fillet_output(r, joinpath("experiments", tag), tag;
                                instellation=1.0, case=0, label="FILLET Benchmark $(tag[end])",
                                extra_global=[start_note(T_WARM, "warm")])
        report(g, r, "experiments/$tag", tag)

    elseif cmd == "exp1" || cmd == "exp2"
        warm = cmd == "exp1"
        println(warm ? "FILLET Experiment 1: warm-start instellation sweep (0.80–1.25 S⊕, ε=0–90°)" :
                       "FILLET Experiment 2: cold-start instellation sweep (1.05–1.50 S⊕, ε=0–90°)")
        S = warm ? collect(range(0.80, 1.25, step=0.025)) : collect(range(1.05, 1.50, step=0.025))
        run_fillet_sweep(S, collect(0:10:90); warm_start=warm, p_base=Params(seasonal=true),
                         outdir=joinpath("experiments", cmd), verbose=true)
        println("→ experiments/$cmd/  (lat_output_AVALON_$(cmd)_{case}.dat × N + global_output_AVALON_$cmd.dat + convergence.log)")

    elseif cmd == "exp1a" || cmd == "exp2a"
        warm = cmd == "exp1a"
        println(warm ? "FILLET Experiment 1a: warm-start semi-major axis sweep (0.875–1.10 au, ε=0–90°; period = 365 d·a^1.5)" :
                       "FILLET Experiment 2a: cold-start semi-major axis sweep (0.80–0.975 au, ε=0–90°; period = 365 d·a^1.5)")
        a = warm ? collect(range(0.875, 1.10, step=0.0125)) : collect(range(0.80, 0.975, step=0.0125))
        run_fillet_sweep_au(a, collect(0:10:90); warm_start=warm, p_base=Params(seasonal=true),
                            outdir=joinpath("experiments", cmd), verbose=true)
        println("→ experiments/$cmd/  (lat_output_AVALON_$(cmd)_{case}.dat × N + global_output_AVALON_$cmd.dat + convergence.log)")

    elseif cmd == "exp3" || cmd == "exp4"
        base = get(opts, "base", "ben2")
        base in ("ben1", "ben2") || (println(stderr, "Error: base must be ben1 or ben2"); exit(1))
        p_base = base == "ben1" ? ben1_params() : Params(seasonal=true)
        tag    = base == "ben2" ? cmd : "$(cmd)_$(base)"
        cfg    = base == "ben2" ?
            "# Configuration: Benchmark 2 (un-tuned FILLET Table 4 parameters), obliquity 23.5" :
            @sprintf("# Configuration: Benchmark 1 (tuned pre-industrial Earth: D = %.2f, alpha_ocean = %.5f, C_ocean = %.3g, Earth-like land fraction), obliquity 23.5", p_base.D, p_base.α_ocean, p_base.C_ocean)
        if cmd == "exp3"
            println("FILLET Experiment 3: instellation hysteresis (0.8–1.5 S⊕, ε=23.5°, base=$base)")
            hysteresis_sweep(collect(range(0.8, 1.5, step=0.0125)) .* S_earth, :S0;
                             p_base=p_base, outdir=joinpath("experiments", tag),
                             label="FILLET Experiment 3 (instellation bifurcation)", notes=[cfg], verbose=true)
        else
            println("FILLET Experiment 4: CO₂ hysteresis (1–100,000 ppm, ε=23.5°, base=$base)")
            hysteresis_sweep(exp10.(range(0, 5, length=50)), :CO2;
                             p_base=p_base, outdir=joinpath("experiments", tag),
                             label="FILLET Experiment 4 (CO2 bifurcation)",
                             notes=[cfg, "# Note: with OLR = A + B*T - F*ln(CO2/280) the CO2 forcing over 1-1e5 ppm is 62 W m^-2, less than a snowball needs to deglaciate at S = 1 (~74 W m^-2, i.e. ~3e8 ppm), so the cold-start branch stays glaciated over the whole range"],
                             verbose=true)
        end
        println("→ experiments/$tag/  (lat_output_AVALON_$(tag)_{case}.dat × N + global_output_AVALON_$tag.dat + convergence.log)")

    elseif cmd == "run"
        run_custom(extra_args)

    elseif cmd == "export"
        export_fillet()

    elseif cmd == "tune_ben1"
        println("Tuning the Benchmark 1 ocean albedo for Tglob = 288.0 K (bisection):")
        α = tune_ben1()
        println("α_ocean = $(round(α, digits=5)) → set α_OCEAN_BEN1 in avalon.jl and re-run benchmark1")

    else
        println("Unknown command: \"$cmd\"\n")
        print(HELP_TEXT)
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
  benchmark1   Tuned pre-industrial Earth (seasonal, D=0.52, α_ocean=$(α_OCEAN_BEN1), C_ocean=2e8, Earth land fraction, ε=23.5°, CO₂=280 ppm)
               → experiments/ben1/
  benchmark2   Un-tuned FILLET Table 4 parameters (seasonal, ε=23.5°, CO₂=280 ppm)   → experiments/ben2/
  benchmark3   Un-tuned, high obliquity (seasonal, ε=60°, CO₂=280 ppm)              → experiments/ben3/

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FILLET experiments
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  exp1         Warm-start instellation sweep  (0.80–1.25 S⊕ × ε=0–90°)            → experiments/exp1/
  exp2         Cold-start instellation sweep  (1.05–1.50 S⊕ × ε=0–90°)            → experiments/exp2/
  exp1a        Warm-start semi-major axis     (0.875–1.10 au × ε=0–90°)            → experiments/exp1a/
  exp2a        Cold-start semi-major axis     (0.80–0.975 au × ε=0–90°)            → experiments/exp2a/
               (1a/2a: S = S⊕/a² and orbital period = 365 d·a^1.5)
  exp3         Instellation hysteresis, warm + cold continuation branches          → experiments/exp3/
  exp4         CO₂ hysteresis (1–100,000 ppm)                                      → experiments/exp4/
               exp3/exp4 use the Benchmark 2 configuration; add base=ben1 for the tuned
               Benchmark 1 configuration (protocol v1.0 §3.7–3.8) → experiments/exp3_ben1/ etc.

  Sweeps write one lat_output_AVALON_{exp}_{case}.dat per case, a single
  global_output_AVALON_{exp}.dat summary and a convergence.log (orbits run,
  convergence, hemispheric symmetry, ice segments per case).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Submission and tuning
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  export       Assemble the projectcuisines/fillet Results/ layout from experiments/
               (ben*/case_0/lat_output.dat, exp*/global_output.dat, exp3_warm|cold, exp4_warm|cold)
               → experiments/fillet_submission/avalon/
  tune_ben1    Bisection on the Benchmark 1 ocean albedo for Tglob = 288 K

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Custom single case
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  run [key=value ...]

  Override any model parameter by name.  Output goes to experiments/<tag>/ (out=<tag>, default: run).

  Special keys:
    out=<tag>       output tag
    au=<value>      semi-major axis [au]: sets S₀ = S⊕/a² and period = 365 d·a^1.5
    T0=<°C>         uniform initial temperature (default: $(T_WARM), the warm start)

  Model parameters (key=value):
    S0=<W/m²>         solar constant                 (default: 1361.0)
    obliquity=<°>     axial tilt                     (default: 23.5)
    period_days=<d>   orbital period                 (default: 365.0)
    CO2=<ppm>         atmospheric CO₂                (default: 280.0)
    CO2_ref=<ppm>     CO₂ reference level            (default: 280.0)
    D=<W/m²/K>        diffusion coefficient          (default: 0.50)
    A=<W/m²>          OLR intercept                  (default: 210.0)
    B=<W/m²/K>        OLR slope                      (default: 2.0)
    alpha_land=<0-1>  land surface albedo            (default: 0.30)
    alpha_ocean=<0-1> open ocean surface albedo      (default: 0.20)
    alpha_ice=<0-1>   ice surface albedo             (default: 0.60)
    T_ice=<°C>        ice threshold temperature      (default: 0.0)
    dT_ice=<K>        width of the ice transition    (default: 1.0; 0 = step)
    C_land=<J/m²K>    land heat capacity             (default: 1e7)
    C_ocean=<J/m²K>   ocean heat capacity            (default: 4e8)
    C_ice=<J/m²K>     ice heat capacity              (default: 1e7; applied to ice-covered bands)
    steps_per_orbit=<N> time steps per orbit         (default: 366; use an even N, see README)
    tol=<K>           convergence tolerance          (default: 1e-4 K per orbit)
    max_orbits=<N>    convergence limit              (default: 500)
    seasonal=true     enable seasonal cycle          (default: false)

  Note: land_fraction (per-band land fraction vector) can only be set
  programmatically, not via the command line. Default: 0.25 uniform (FILLET Table 4).

  Examples:
    julia avalon.jl run obliquity=60 CO2=1000 seasonal=true out=highco2_obl60
    julia avalon.jl run au=0.9 obliquity=45 seasonal=true out=innerhz
    julia avalon.jl run alpha_ocean=0.28 D=0.44 S0=1200

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Plotting  (requires numpy, matplotlib)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  python3 plot.py <tag>                Annual-mean diagnostic panels (2×2 with land fraction, or 1×3)
  python3 plot.py <tag> seasonal       Hovmöller temperature plot + seasonal amplitude (needs <tag>_seasonal.csv)
  python3 plot.py <tag> sweep          Obliquity × instellation phase diagram (exp1, exp2, exp1a, exp2a)
  python3 plot.py <tag> bifurcation    Hysteresis diagram, warm and cold branches (exp3, exp4, exp3_ben1, exp4_ben1)
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
