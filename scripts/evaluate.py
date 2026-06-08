"""Valutazione quantitativa: unmixing fasoriale vs fitting NNLS del decadimento.

Genera, su dati sintetici con ground truth, il MAE delle frazioni in funzione del
numero di fotoni (livello di rumore) per i due metodi, misura i tempi e salva:
- results/metrics.json
- results/phasor_plot.png   (phasor plot dei pixel + cerchio universale)
- results/mae_vs_photons.png

Esegui:  uv run python scripts/evaluate.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from phasor_flim.baseline import fit_fractions
from phasor_flim.metrics import fraction_mae
from phasor_flim.phasor import phasor_image, universal_circle
from phasor_flim.synth import make_synthetic_flim
from phasor_flim.unmix import reference_phasors, unmix_fractions

RESULTS = Path(__file__).resolve().parents[1] / "results"


def run() -> dict:
    photon_levels = [300, 800, 2000, 5000]
    rows = []
    for photons in photon_levels:
        flim = make_synthetic_flim(
            shape=(48, 48), total_photons=photons, poisson=True, seed=0
        )
        refs = reference_phasors(flim.lifetimes, flim.freq_mhz)

        t0 = time.perf_counter()
        g, s = phasor_image(flim.decay, flim.t, flim.freq_mhz)
        est_phasor = unmix_fractions(g, s, refs)
        t_phasor = time.perf_counter() - t0

        t0 = time.perf_counter()
        est_fit = fit_fractions(flim.decay, flim.t, flim.lifetimes)
        t_fit = time.perf_counter() - t0

        rows.append(
            {
                "photons": photons,
                "mae_phasor": fraction_mae(est_phasor, flim.fractions),
                "mae_fit": fraction_mae(est_fit, flim.fractions),
                "time_phasor_s": round(t_phasor, 4),
                "time_fit_s": round(t_fit, 4),
            }
        )
        print(
            f"photons={photons:5d} | MAE phasor={rows[-1]['mae_phasor']:.4f} "
            f"fit={rows[-1]['mae_fit']:.4f} | "
            f"t_phasor={t_phasor:.3f}s t_fit={t_fit:.3f}s"
        )

    speedup = float(np.mean([r["time_fit_s"] / max(r["time_phasor_s"], 1e-9) for r in rows]))
    return {"per_level": rows, "mean_fit_vs_phasor_speedup": round(speedup, 1)}


def make_phasor_figure() -> None:
    flim = make_synthetic_flim(shape=(64, 64), total_photons=2000, seed=0)
    g, s = phasor_image(flim.decay, flim.t, flim.freq_mhz)
    refs = reference_phasors(flim.lifetimes, flim.freq_mhz)
    cx, cy = universal_circle()

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(cx, cy, "k-", lw=1, label="cerchio universale")
    ax.scatter(g.ravel(), s.ravel(), s=4, alpha=0.25, c="tab:blue", label="pixel")
    ax.scatter(refs[:, 0], refs[:, 1], c="tab:red", s=80, marker="*", zorder=5,
               label="specie pure")
    for tau, (gx, gy) in zip(flim.lifetimes, refs):
        ax.annotate(f"{tau:.1f} ns", (gx, gy), textcoords="offset points", xytext=(6, 6))
    ax.set_xlabel("g")
    ax.set_ylabel("s")
    ax.set_title("Phasor plot — FLIM sintetico (80 MHz)")
    ax.set_aspect("equal")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(RESULTS / "phasor_plot.png", dpi=130)
    plt.close(fig)


def make_mae_figure(metrics: dict) -> None:
    rows = metrics["per_level"]
    x = [r["photons"] for r in rows]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(x, [r["mae_phasor"] for r in rows], "o-", label="Phasor (fit-free)")
    ax.plot(x, [r["mae_fit"] for r in rows], "s--", label="Fitting NNLS")
    ax.set_xscale("log")
    ax.set_xlabel("fotoni per pixel (≈ inverso del rumore)")
    ax.set_ylabel("MAE frazioni ↓")
    ax.set_title("Accuratezza unmixing vs livello di fotoni")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS / "mae_vs_photons.png", dpi=130)
    plt.close(fig)


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    metrics = run()
    make_phasor_figure()
    make_mae_figure(metrics)
    (RESULTS / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"\nSalvati: {RESULTS / 'metrics.json'}, phasor_plot.png, mae_vs_photons.png")
    print(f"Speedup medio fit/phasor: {metrics['mean_fit_vs_phasor_speedup']}×")


if __name__ == "__main__":
    main()
