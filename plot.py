"""
Plot AVALON / FILLET benchmark output.

Usage:
    python3 plot.py <tag>

Reads {tag}/lat_output_AVALON_{tag}.dat and {tag}/global_output_AVALON_{tag}.dat,
and writes {tag}/{tag}.png and {tag}/{tag}.pdf.

If a {tag}/{tag}_seasonal.csv file exists (monthly snapshots), the temperature
panel shows the seasonal envelope as a shaded band.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt


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
    lat_file      = os.path.join(tag, f"lat_output_AVALON_{tag}.dat")
    global_file   = os.path.join(tag, f"global_output_AVALON_{tag}.dat")
    seasonal_file = os.path.join(tag, f"{tag}_seasonal.csv")

    if not os.path.exists(lat_file) or not os.path.exists(global_file):
        sys.exit(f"Error: could not find {lat_file} or {global_file}\n"
                 f"  (Run 'julia avalon.jl {tag}' first, or check the tag name)")

    lat_data = read_dat(lat_file)
    glob_data = read_dat(global_file)

    lat   = lat_data["Lat"]
    T_C   = lat_data["Tsurf"] - 273.15
    alb   = lat_data["Asurf"]
    olr   = lat_data["OLR"]
    Tglob = glob_data["Tglob"][0]

    # Ice edge: NMin=90 means ice-free
    ice_min = glob_data["IceLineNMin"][0]
    ice_min = None if ice_min == 90.0 else ice_min

    # Absorbed SW from annual-mean insolation
    x  = np.sin(np.deg2rad(lat))
    P2 = (3*x**2 - 1) / 2
    Q  = (1361/4) * (1 + (-0.482) * P2)
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

    # --- Build header from global file comments ---
    meta = {}
    with open(global_file) as f:
        for line in f:
            if "Instellation" in line:
                meta["inst"] = line.split(":")[-1].strip() if ":" in line else "?"
    obl  = glob_data["Obl"][0]
    co2  = glob_data["XCO2"][0]
    inst = glob_data["Inst"][0]

    title = (f"AVALON — {tag.replace('_', ' ').title()}\n"
             f"S₀ = {inst} S⊕,  ε = {obl}°,  CO₂ = {co2:.0f} ppm  "
             f"|  mode: {mode}  |  $T_{{\\rm glob}}$ = {Tglob:.2f} K")

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    fig.suptitle(title, fontsize=10)

    # --- Temperature ---
    ax1 = axes[0]
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
    ax2 = axes[1]
    ax2.plot(lat, alb, color="tab:green", lw=1.8)
    ax2.set_xlabel("Latitude (°)"); ax2.set_ylabel("Albedo")
    ax2.set_title("Annual-mean surface / TOA albedo")
    ax2.set_ylim(0, 0.75); ax2.set_xlim(-90, 90); ax2.set_xticks(range(-90, 91, 30))

    # --- Energy balance ---
    ax3 = axes[2]
    ax3.plot(lat, sw,  color="tab:purple", lw=1.8, label="SW absorbed")
    ax3.plot(lat, olr, color="tab:orange", lw=1.8, ls="--", label="OLR")
    ax3.set_xlabel("Latitude (°)"); ax3.set_ylabel("W m⁻²")
    ax3.set_title("Annual-mean energy balance")
    ax3.legend(fontsize=8); ax3.set_xlim(-90, 90); ax3.set_xticks(range(-90, 91, 30))
    ax3.text(0.97, 0.05,
             f"Global mean:\nSW = {np.mean(sw):.1f} W m⁻²\nOLR = {np.mean(olr):.1f} W m⁻²",
             transform=ax3.transAxes, ha="right", va="bottom", fontsize=8,
             bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))

    plt.tight_layout()
    for ext in ("png", "pdf"):
        out = os.path.join(tag, f"{tag}.{ext}")
        plt.savefig(out, dpi=150, bbox_inches="tight", format=ext)
        print(f"Saved {out}")
    plt.close()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Usage: python3 plot.py <tag>\nExample: python3 plot.py ben1")
    plot_benchmark(sys.argv[1])
