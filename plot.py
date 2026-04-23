"""
Plot AVALON / FILLET benchmark output.

Usage:
    python3 plot.py <tag>                # annual-mean diagnostic panels (2×2 if Fland present)
    python3 plot.py <tag> seasonal       # Hovmöller + seasonal amplitude (requires {tag}_seasonal.csv)
    python3 plot.py <tag> sweep          # obliquity×instellation phase diagram (exp1/2/1a/2a)
    python3 plot.py <tag> bifurcation    # hysteresis diagram with cooling/warming branches (exp3/4)

Single-case plots read:
    experiments/{tag}/lat_output_AVALON_{tag}.dat
    experiments/{tag}/global_output_AVALON_{tag}.dat
and write experiments/{tag}/{tag}.png and .pdf.

Sweep/bifurcation plots read experiments/{tag}/global_output_AVALON_{tag}.dat
and write experiments/{tag}/{tag}_sweep.{png,pdf} or _{bifurcation,seasonal}.{png,pdf}.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt

sys.dont_write_bytecode = True


def _annual_mean_insolation(lat_deg, obliquity_deg, S0=1361.0, n_lambda=360):
    """Annual-mean insolation [W m⁻²] using the Berger (1978) formula."""
    eps = np.radians(obliquity_deg)
    lat = np.radians(np.asarray(lat_deg, dtype=float))
    Q = np.zeros_like(lat)
    for k in range(n_lambda):
        lam   = (k + 0.5) * (2 * np.pi / n_lambda)
        delta = np.arcsin(np.sin(eps) * np.sin(lam))
        t     = -np.tan(lat) * np.tan(delta)
        H0    = np.where(t <= -1, np.pi, np.where(t >= 1, 0.0,
                         np.arccos(np.clip(t, -1.0, 1.0))))
        Q    += H0 * np.sin(lat) * np.sin(delta) + np.cos(lat) * np.cos(delta) * np.sin(H0)
    return (S0 / np.pi) * Q / n_lambda


def read_dat(path):
    """Read a space-separated .dat file with # comment headers into a dict of arrays."""
    rows = []
    headers = None
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("#"):
                # Last comment line before data is the column header
                tokens = line.lstrip("# ").split()
                if all(t[0].isupper() or t[0].isdigit() for t in tokens):
                    headers = tokens
            else:
                rows.append(line.split())
    if headers is None or not rows:
        raise ValueError(f"Could not parse {path}")
    data = {h: [] for h in headers}
    for row in rows:
        for h, v in zip(headers, row):
            try:
                data[h].append(float(v))
            except ValueError:
                data[h].append(v)
    return {k: np.array(v) if isinstance(v[0], float) else v for k, v in data.items()}


def plot_benchmark(tag):
    outdir        = os.path.join("experiments", tag)
    lat_file      = os.path.join(outdir, f"lat_output_AVALON_{tag}.dat")
    global_file   = os.path.join(outdir, f"global_output_AVALON_{tag}.dat")
    seasonal_file = os.path.join(outdir, f"{tag}_seasonal.csv")

    if not os.path.exists(lat_file) or not os.path.exists(global_file):
        TAG_TO_CMD = {"ben1": "benchmark1", "ben2": "benchmark2", "ben3": "benchmark3"}
        cmd = TAG_TO_CMD.get(tag, tag)
        sys.exit(f"Error: output files for '{tag}' not found.\n"
                 f"  Run: julia avalon.jl {cmd}\n"
                 f"  (or check that '{tag}' is the correct tag name)")

    lat_data = read_dat(lat_file)
    glob_data = read_dat(global_file)

    lat   = lat_data["Lat"]
    T_C   = lat_data["Tsurf"] - 273.15
    alb   = lat_data["Asurf"]
    olr   = lat_data["OLR"]
    Tglob = glob_data["Tglob"][0]

    # Ice edge: NMinSea=90 means ice-free (FILLET v1.1 column names)
    ice_min = glob_data["IceLineNMinSea"][0]
    ice_min = None if ice_min == 90.0 else ice_min

    obl  = glob_data["Obl"][0]
    co2  = glob_data["XCO2"][0]
    inst = glob_data["Inst"][0]

    # Absorbed SW using the actual obliquity and S0 from the run
    Q  = _annual_mean_insolation(lat, obl, S0=inst * 1361.0)
    sw = (1 - alb) * Q

    # Read mode from header for subtitle
    mode = "unknown"
    with open(global_file) as f:
        for line in f:
            if "Mode:" in line:
                mode = line.split("Mode:")[-1].strip()
                break

    # Read seasonal envelope if available
    T_seas = None
    if os.path.exists(seasonal_file):
        import csv
        with open(seasonal_file) as f:
            reader = csv.reader(f)
            header = next(reader)
            rows = [list(map(float, r)) for r in reader]
        T_seas = np.array(rows)[:, 1:] - 273.15   # (12, n_lat)

    title = (f"AVALON — {tag.replace('_', ' ').title()}\n"
             f"S₀ = {inst} S⊕,  ε = {obl}°,  CO₂ = {co2:.0f} ppm  "
             f"|  mode: {mode}  |  $T_{{\\rm glob}}$ = {Tglob:.2f} K")

    has_fland = "Fland" in lat_data
    if has_fland:
        fig, axes = plt.subplots(2, 2, figsize=(11, 8))
        ax1, ax2, ax3, ax4 = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]
    else:
        fig, axes_row = plt.subplots(1, 3, figsize=(13, 4.5))
        ax1, ax2, ax3 = axes_row
    fig.suptitle(title, fontsize=10)

    # --- Temperature ---
    if T_seas is not None:
        ax1.fill_between(lat, T_seas.min(axis=0), T_seas.max(axis=0),
                         color="tab:red", alpha=0.2, label="Seasonal range",
                         rasterized=True)
    ax1.plot(lat, T_C, color="tab:red", lw=1.8,
             label="Annual mean" if T_seas is not None else "Temperature")
    ax1.axhline(-10, color="tab:blue", lw=0.9, ls="--", label="$T_{\\rm ice}$ = −10 °C")
    ax1.axhline(0,   color="gray",     lw=0.6, ls=":")
    if ice_min is not None:
        ax1.axvline( ice_min, color="tab:blue", lw=0.9, ls=":",
                     label=f"Ice edge ±{ice_min:.1f}°")
        ax1.axvline(-ice_min, color="tab:blue", lw=0.9, ls=":")
    ax1.set_xlabel("Latitude (°)"); ax1.set_ylabel("Temperature (°C)")
    ax1.set_title("Surface temperature")
    ax1.legend(fontsize=7.5); ax1.set_xlim(-90, 90); ax1.set_xticks(range(-90, 91, 30))

    # --- Albedo ---
    ax2.plot(lat, alb, color="tab:green", lw=1.8)
    ax2.set_xlabel("Latitude (°)"); ax2.set_ylabel("Albedo")
    ax2.set_title("Annual-mean surface / TOA albedo")
    ax2.set_ylim(0, 0.75); ax2.set_xlim(-90, 90); ax2.set_xticks(range(-90, 91, 30))

    # --- Energy balance ---
    ax3.plot(lat, sw,  color="tab:purple", lw=1.8, label="SW absorbed")
    ax3.plot(lat, olr, color="tab:orange", lw=1.8, ls="--", label="OLR")
    ax3.set_xlabel("Latitude (°)"); ax3.set_ylabel("W m⁻²")
    ax3.set_title("Annual-mean energy balance")
    ax3.legend(fontsize=8); ax3.set_xlim(-90, 90); ax3.set_xticks(range(-90, 91, 30))
    ax3.text(0.97, 0.05,
             f"Global mean:\nSW = {np.mean(sw):.1f} W m⁻²\nOLR = {np.mean(olr):.1f} W m⁻²",
             transform=ax3.transAxes, ha="right", va="bottom", fontsize=8,
             bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))

    # --- Land fraction ---
    if has_fland:
        fland = lat_data["Fland"]
        ax4.plot(lat, fland * 100, color="tab:brown", lw=1.8)
        ax4.fill_between(lat, 0, fland * 100, color="tab:brown", alpha=0.25)
        ax4.axhline(np.mean(fland) * 100, color="tab:brown", lw=0.9, ls="--",
                    label=f"Global mean {np.mean(fland)*100:.1f}%")
        ax4.set_xlabel("Latitude (°)")
        ax4.set_ylabel("Land fraction (%)")
        ax4.set_title("Land fraction per band")
        ax4.set_xlim(-90, 90)
        ax4.set_xticks(range(-90, 91, 30))
        ax4.set_ylim(0, 100)
        ax4.legend(fontsize=8)

    plt.tight_layout()
    for ext in ("png", "pdf"):
        out = os.path.join(outdir, f"{tag}.{ext}")
        plt.savefig(out, dpi=150, bbox_inches="tight", format=ext)
        print(f"Saved {out}")
    plt.close()


def plot_seasonal(tag):
    """
    Two-panel seasonal cycle plot:
      Left:  T(lat, month) Hovmöller with ice-threshold contour
      Right: seasonal amplitude (max − min) vs latitude
    """
    outdir        = os.path.join("experiments", tag)
    seasonal_file = os.path.join(outdir, f"{tag}_seasonal.csv")
    if not os.path.exists(seasonal_file):
        TAG_TO_CMD = {"ben1": "benchmark1", "ben2": "benchmark2", "ben3": "benchmark3"}
        cmd   = TAG_TO_CMD.get(tag, tag)
        extra = "" if tag in TAG_TO_CMD else " seasonal=true"
        sys.exit(
            f"Error: {seasonal_file} not found.\n"
            f"  Run: julia avalon.jl {cmd}{extra}\n"
            f"  or check that '{tag}' is the correct experiment tag."
        )

    import csv
    with open(seasonal_file) as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = [list(map(float, r)) for r in reader]

    T_K   = np.array(rows)[:, 1:]          # (n_months, n_lat) in K
    T_C   = T_K - 273.15
    lats  = np.array([float(h) for h in header[1:]])
    months = np.arange(1, T_C.shape[0] + 1)
    amplitude = T_C.max(axis=0) - T_C.min(axis=0)

    MONTH_LABELS = ["Jan","Feb","Mar","Apr","May","Jun",
                    "Jul","Aug","Sep","Oct","Nov","Dec"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.suptitle(f"AVALON — {tag.replace('_',' ').title()} — Seasonal cycle", fontsize=11)

    # --- Hovmöller ---
    vmin, vmax = np.floor(T_C.min() / 5) * 5, np.ceil(T_C.max() / 5) * 5
    cf = ax1.contourf(lats, months, T_C, levels=20, cmap="RdBu_r",
                      vmin=vmin, vmax=vmax)
    ax1.contour(lats, months, T_C, levels=[-10], colors="cyan",
                linewidths=1.2, linestyles="--")
    plt.colorbar(cf, ax=ax1, label="Temperature (°C)")
    ax1.set_xlabel("Latitude (°)")
    ax1.set_ylabel("Month")
    ax1.set_yticks(months)
    ax1.set_yticklabels(MONTH_LABELS)
    ax1.set_xlim(-90, 90)
    ax1.set_xticks(range(-90, 91, 30))
    ax1.set_title("Surface temperature (°C)\n[cyan dashed = −10 °C ice threshold]")

    # --- Seasonal amplitude ---
    ax2.plot(lats, amplitude, color="tab:orange", lw=2)
    ax2.fill_between(lats, 0, amplitude, color="tab:orange", alpha=0.2)
    ax2.axhline(0, color="gray", lw=0.6)
    ax2.set_xlabel("Latitude (°)")
    ax2.set_ylabel("Seasonal amplitude (°C)")
    ax2.set_title("Peak-to-peak seasonal range\n(max T − min T per band)")
    ax2.set_xlim(-90, 90)
    ax2.set_xticks(range(-90, 91, 30))
    ax2.set_ylim(bottom=0)

    # Annotate peak amplitude
    peak_idx = np.argmax(amplitude)
    ax2.annotate(f"{amplitude[peak_idx]:.1f} °C at {lats[peak_idx]:.0f}°",
                 xy=(lats[peak_idx], amplitude[peak_idx]),
                 xytext=(lats[peak_idx] - 25, amplitude[peak_idx] * 0.85),
                 arrowprops=dict(arrowstyle="->", color="gray"), fontsize=9)

    plt.tight_layout()
    for ext in ("png", "pdf"):
        out = os.path.join(outdir, f"{tag}_seasonal.{ext}")
        plt.savefig(out, dpi=150, bbox_inches="tight", format=ext)
        print(f"Saved {out}")
    plt.close()


def _climate_state(nmax, nmin):
    """Classify a single grid point into one of four FILLET climate states."""
    if nmin == 90.0:
        return "ice-free"
    elif nmax < 90.0 and nmin == 0.0:
        return "ice-belt"
    elif nmin == 0.0:
        return "snowball"
    else:
        return "ice-caps"


def plot_sweep(tag):
    """
    Phase diagram for instellation × obliquity sweep experiments (exp1, exp2, exp1a, exp2a).
    X-axis: obliquity (°), Y-axis: instellation (S⊕), colored by climate state.
    Analogous to FILLET protocol Figure 3 / Figure 4.
    """
    from matplotlib.colors import BoundaryNorm, ListedColormap
    from matplotlib.patches import Patch

    outdir      = os.path.join("experiments", tag)
    global_file = os.path.join(outdir, f"global_output_AVALON_{tag}.dat")
    if not os.path.exists(global_file):
        sys.exit(f"Error: {global_file} not found.\n"
                 f"  Run: julia avalon.jl {tag}")

    data  = read_dat(global_file)
    inst  = data["Inst"]
    obl   = data["Obl"]
    nmax  = data["IceLineNMaxSea"]
    nmin  = data["IceLineNMinSea"]
    tglob = data["Tglob"] - 273.15

    n_obl  = len(set(np.round(obl,  4)))
    n_inst = len(set(np.round(inst, 6)))
    if n_obl < 2 or n_inst < 2:
        sys.exit(
            f"Error: '{tag}' does not look like a sweep experiment "
            f"(found {n_obl} obliquity value(s) and {n_inst} instellation value(s)).\n"
            f"  'sweep' requires an obliquity × instellation grid — intended for exp1, exp2, exp1a, exp2a.\n"
            f"  For a single-case run, use:  python3 plot.py {tag}"
        )

    states = np.array([_climate_state(nm, ni) for nm, ni in zip(nmax, nmin)])

    # Fixed state ordering and colours (consistent across exp1/exp2)
    STATE_ORDER  = ["ice-free", "ice-caps", "ice-belt", "snowball"]
    STATE_COLORS = ["#a8d8ea", "#3a7ebf", "#e07b39", "#1a1a2e"]
    state_idx    = {s: i for i, s in enumerate(STATE_ORDER)}

    # Build regular 2-D grid (obliquity × instellation)
    obliquities = np.array(sorted(set(np.round(obl, 4))))
    inst_vals   = np.array(sorted(set(np.round(inst, 6))))
    Z     = np.full((len(inst_vals), len(obliquities)), np.nan)
    Tgrid = np.full((len(inst_vals), len(obliquities)), np.nan)

    for s, o, st, tg in zip(inst, obl, states, tglob):
        ii = np.searchsorted(inst_vals,   round(s, 6))
        oi = np.searchsorted(obliquities, round(o, 4))
        Z[ii, oi]     = state_idx[st]
        Tgrid[ii, oi] = tg

    # pcolormesh cell edges (half-step extension)
    def _edges(arr):
        d = np.diff(arr)
        return np.concatenate([[arr[0] - d[0]/2], arr[:-1] + d/2, [arr[-1] + d[-1]/2]])

    OBL_m, INST_m = np.meshgrid(obliquities, inst_vals)

    cmap  = ListedColormap(STATE_COLORS)
    norm  = BoundaryNorm(np.arange(-0.5, 4), cmap.N)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    fig.suptitle(f"AVALON — {tag.upper()} climate state diagram", fontsize=11)

    # --- Left: climate state ---
    ax = axes[0]
    ax.pcolormesh(_edges(obliquities), _edges(inst_vals), Z,
                  cmap=cmap, norm=norm, shading="flat")
    # Contour lines at state boundaries
    if not np.all(np.isnan(Z)):
        try:
            ax.contour(OBL_m, INST_m, Z, levels=[0.5, 1.5, 2.5],
                       colors="white", linewidths=0.8, linestyles="-")
        except Exception:
            pass
    ax.set_xlabel("Obliquity (°)")
    ax.set_ylabel("Instellation (S⊕)")
    ax.set_title("Climate state")
    ax.set_xlim(obliquities[0] - 5, obliquities[-1] + 5)
    ax.set_xticks(obliquities[::2])
    legend_patches = [Patch(facecolor=STATE_COLORS[i], label=STATE_ORDER[i])
                      for i in range(4) if i in [state_idx[s] for s in np.unique(states)]]
    ax.legend(handles=legend_patches, fontsize=9, loc="upper left")

    # --- Right: global mean temperature ---
    ax2 = axes[1]
    pcm = ax2.pcolormesh(_edges(obliquities), _edges(inst_vals), Tgrid,
                         cmap="RdBu_r", shading="flat")
    plt.colorbar(pcm, ax=ax2, label="Global mean temperature (°C)")
    # Overlay state boundaries
    if not np.all(np.isnan(Z)):
        try:
            ax2.contour(OBL_m, INST_m, Z, levels=[0.5, 1.5, 2.5],
                        colors="black", linewidths=0.9, linestyles="--")
        except Exception:
            pass
    ax2.set_xlabel("Obliquity (°)")
    ax2.set_ylabel("Instellation (S⊕)")
    ax2.set_title("Global mean temperature (°C)\n[dashed = state boundaries]")
    ax2.set_xlim(obliquities[0] - 5, obliquities[-1] + 5)
    ax2.set_xticks(obliquities[::2])

    plt.tight_layout()
    for ext in ("png", "pdf"):
        out = os.path.join(outdir, f"{tag}_sweep.{ext}")
        plt.savefig(out, dpi=150, bbox_inches="tight", format=ext)
        print(f"Saved {out}")
    plt.close()


def plot_bifurcation(tag):
    """
    Bifurcation (hysteresis) diagram for exp3 (instellation sweep) or exp4 (CO₂ sweep).
    Two panels: global mean temperature and NH ice edge vs the swept parameter,
    cooling and warming branches overlaid to show the bistable region.
    """
    outdir      = os.path.join("experiments", tag)
    global_file = os.path.join(outdir, f"global_output_AVALON_{tag}.dat")
    if not os.path.exists(global_file):
        sys.exit(f"Error: {global_file} not found.\n"
                 f"  Run: julia avalon.jl {tag}")

    data = read_dat(global_file)

    if "Branch" not in data:
        sys.exit(
            f"Error: '{tag}' does not look like a bifurcation experiment "
            f"(no 'Branch' column found in {global_file}).\n"
            f"  'bifurcation' requires cooling/warming branch data — intended for exp3, exp4.\n"
            f"  For a single-case run, use:  python3 plot.py {tag}"
        )

    # Detect swept parameter: CO2 varies in exp4, Inst in exp3
    co2_vals = data["XCO2"]
    is_co2   = co2_vals.min() != co2_vals.max()

    if is_co2:
        x_all   = co2_vals
        xlabel  = "CO₂ (ppm)"
        xlog    = True
    else:
        x_all   = data["Inst"]
        xlabel  = "Instellation (S⊕)"
        xlog    = False

    tglob   = data["Tglob"] - 273.15
    ice_min = data["IceLineNMinSea"]   # 90 = ice-free, 0 = snowball
    branch  = np.array(data["Branch"]) if isinstance(data["Branch"][0], str) else data["Branch"]

    BRANCH_STYLE = {
        "cooling": dict(color="tab:blue",   lw=2.0, label="Cooling branch (warm start)"),
        "warming": dict(color="tab:orange", lw=2.0, label="Warming branch (cold start)"),
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"AVALON — {tag.upper()} bifurcation diagram", fontsize=11)

    for br, style in BRANCH_STYLE.items():
        mask = branch == br
        if not np.any(mask):
            continue
        idx = np.argsort(x_all[mask])
        x   = x_all[mask][idx]
        tg  = tglob[mask][idx]
        ic  = ice_min[mask][idx]

        ax1.plot(x, tg, **style)
        # Ice edge: replace 90 (ice-free) with NaN so it doesn't plot at 90°
        ic_plot = np.where(ic == 90.0, np.nan, ic)
        ax2.plot(x, ic_plot, **style)
        # Mark snowball and ice-free endpoints
        ax2.plot(x[ic == 0.0],  np.zeros(np.sum(ic == 0.0)),
                 "v", color=style["color"], ms=5, zorder=3)
        ax2.plot(x[ic == 90.0], np.full(np.sum(ic == 90.0), 90.0),
                 "^", color=style["color"], ms=5, zorder=3,
                 label=f"{br} ice-free" if np.any(ic == 90.0) else "")

    # Shade bistable region (instellation range where both branches exist)
    cool_x = x_all[branch == "cooling"]
    warm_x = x_all[branch == "warming"]
    if len(cool_x) and len(warm_x):
        bistable_lo = max(cool_x.min(), warm_x.min())
        bistable_hi = min(cool_x.max(), warm_x.max())
        for ax in (ax1, ax2):
            ax.axvspan(bistable_lo, bistable_hi, color="gray", alpha=0.08,
                       label="Overlapping range")

    ax1.axhline(-10, color="tab:blue", lw=0.8, ls="--", alpha=0.5, label="T = −10 °C")
    ax1.axhline(0,   color="gray",     lw=0.6, ls=":")
    ax1.set_xlabel(xlabel); ax1.set_ylabel("Global mean temperature (°C)")
    ax1.set_title("Global mean temperature")
    ax1.legend(fontsize=8)
    if xlog: ax1.set_xscale("log")

    ax2.axhline(0,  color="gray", lw=0.6, ls=":")
    ax2.set_xlabel(xlabel); ax2.set_ylabel("NH ice edge (°)")
    ax2.set_title("NH ice edge\n(triangle = ice-free / snowball endpoint)")
    ax2.set_ylim(-5, 95)
    ax2.legend(fontsize=8)
    if xlog: ax2.set_xscale("log")

    plt.tight_layout()
    for ext in ("png", "pdf"):
        out = os.path.join(outdir, f"{tag}_bifurcation.{ext}")
        plt.savefig(out, dpi=150, bbox_inches="tight", format=ext)
        print(f"Saved {out}")
    plt.close()


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] in ("--help", "-h"):
        sys.exit("Usage: python3 plot.py <tag> [seasonal|sweep|bifurcation]\n"
                 "Examples: python3 plot.py ben1\n"
                 "          python3 plot.py ben1 seasonal\n"
                 "          python3 plot.py exp1 sweep\n"
                 "          python3 plot.py exp3 bifurcation")
    if len(sys.argv) == 3 and sys.argv[2] == "seasonal":
        plot_seasonal(sys.argv[1])
    elif len(sys.argv) == 3 and sys.argv[2] == "sweep":
        plot_sweep(sys.argv[1])
    elif len(sys.argv) == 3 and sys.argv[2] == "bifurcation":
        plot_bifurcation(sys.argv[1])
    elif len(sys.argv) == 3:
        sys.exit(f"Error: unknown plot mode '{sys.argv[2]}'.\n"
                 f"  Valid modes: seasonal, sweep, bifurcation\n"
                 f"  To plot annual-mean panels, omit the mode: python3 plot.py {sys.argv[1]}")
    else:
        plot_benchmark(sys.argv[1])
