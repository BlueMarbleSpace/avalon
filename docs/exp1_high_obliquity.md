# Experiment 1, high obliquity, and the ice-onset threshold

**Summary.** AVALON uses a **0 °C ice-onset threshold** (`T_ice = 0.0`, the model
default). With it, the warm-start sweep (exp1) gives a **snowball at low instellation
for every obliquity**, consistent with the FILLET ensemble. This note records why the
threshold matters and what an earlier −10 °C choice did.

## What −10 °C did (the original outlier)

With a −10 °C Budyko threshold, exp1 produced a snowball at low obliquity but stayed
*ice-free* at low instellation once obliquity exceeded ~45°:

| Obliquity | −10 °C result at S = 0.8 |
|---|---|
| 0°–40° | snowball |
| 50°–90° | ice-free |

This was **genuine bistability**, not a code bug (cold-starting the same cases gave
snowball; warm-starting found a second stable branch). At high obliquity the poles
receive most of the annual insolation, so the *equator* is the cold trap; on the warm
branch it equilibrates at ≈ −6.5 °C — below 0 °C but above −10 °C. A −10 °C onset
therefore leaves it ice-free, while the ensemble (and physical intuition) expects a
snowball there.

## Why 0 °C fixes it

The equatorial cold trap sits in the −10 … 0 °C window, so raising the ice-onset
threshold to 0 °C freezes it and collapses the warm branch. Result: exp1 is now
**snowball for all obliquity at S ≤ 0.825**, and the spurious high-obliquity ice-free
tongue is gone. The disagreement with the ensemble was an *ice-onset-temperature*
difference all along, not a numerical defect.

## Consequences of the 0 °C choice (recorded for transparency)

`T_ice` sets both the ice-albedo onset and the diagnostic ice line, so moving it from
−10 °C to 0 °C cools and expands ice in every case:

| Case | −10 °C | 0 °C (current default) |
|---|---|---|
| exp1, S = 0.8 | ice-free for ε ≥ 50° | **snowball, all ε** |
| Benchmark 1 (tuned) | Tglob 288 K, ice edge ~73°N | Tglob 288 K (re-tuned α_ocean=0.223), **ice edge ~50°N** |
| Benchmark 2 (ε=23.5) | ice-free, 300.0 K | **ice cap at 64°, 295.6 K** |
| Benchmark 3 (ε=60) | ice-free, 300.0 K | ice-free, 300.0 K (unchanged) |

Two things to keep in mind:

1. **Benchmark 1's ice line is now ~50°N**, more equatorward than the ~70°N perennial
   ice line on real Earth. Annual-mean 0 °C occurs near 50° latitude, whereas −10 °C
   (the Budyko value) occurs near 70°. Re-tuning `α_ocean` recovers Tglob = 288 K but
   not a ~70°N edge; recovering both would need joint re-tuning (e.g. also raising the
   diffusion `D`). Flag this if pre-industrial ice realism matters for ben1.

2. **Benchmark 2 now has ice caps** where it was ice-free. Confirm the FILLET ensemble
   also uses a 0 °C (or warmer) onset; if the ensemble used −10 °C for the benchmarks,
   ben2 will now read as an outlier in the other direction.
