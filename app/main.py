"""Phasor FLIM Explorer — app interattiva.

Avvia con:  uv run streamlit run app/main.py

Esplora interattivamente la trasformata fasoriale di dati FLIM sintetici:
- regola rumore (fotoni) e ordine armonico
- visualizza intensità, phasor plot e mappe di frazione stimate
- confronta l'accuratezza (MAE) con la ground truth
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from phasor_flim.metrics import fraction_mae, per_species_mae
from phasor_flim.phasor import phasor_image, universal_circle
from phasor_flim.synth import make_synthetic_flim
from phasor_flim.unmix import reference_phasors, unmix_fractions

st.set_page_config(page_title="Phasor FLIM Explorer", layout="wide")
st.title("🔬 Phasor FLIM Explorer")
st.caption(
    "Analisi fasoriale fit-free di dati FLIM sintetici con ground truth. "
    "Trasformata phasor → unmixing lineare delle frazioni di specie."
)

with st.sidebar:
    st.header("Parametri")
    photons = st.select_slider(
        "Fotoni per pixel (↑ = meno rumore)",
        options=[200, 500, 1000, 2000, 5000, 10000],
        value=2000,
    )
    harmonic = st.radio("Ordine armonico", [1, 2], horizontal=True)
    size = st.select_slider("Dimensione immagine", options=[32, 48, 64, 96], value=64)
    seed = st.number_input("Seed", value=0, step=1)
    st.markdown("---")
    st.markdown("Vite medie specie (ns): **0.6 · 2.4 · 4.0** @ 80 MHz")


@st.cache_data
def compute(photons: int, harmonic: int, size: int, seed: int):
    flim = make_synthetic_flim(
        shape=(size, size), total_photons=photons, poisson=True, seed=int(seed)
    )
    g, s = phasor_image(flim.decay, flim.t, flim.freq_mhz, harmonic=harmonic)
    refs = reference_phasors(flim.lifetimes, flim.freq_mhz, harmonic=harmonic)
    est = unmix_fractions(g, s, refs)
    return flim, g, s, refs, est


flim, g, s, refs, est = compute(photons, harmonic, size, seed)
k = flim.lifetimes.size
mae = fraction_mae(est, flim.fractions)
mae_k = per_species_mae(est, flim.fractions)

c1, c2 = st.columns([1, 1])

with c1:
    st.subheader("Intensità integrata")
    inten = flim.intensity
    fig_i = go.Figure(go.Heatmap(z=inten, colorscale="Gray", showscale=False))
    fig_i.update_layout(height=380, margin=dict(l=0, r=0, t=10, b=0), yaxis_autorange="reversed")
    st.plotly_chart(fig_i, use_container_width=True)

with c2:
    st.subheader("Phasor plot")
    dominant = np.nanargmax(est, axis=-1).astype(float).ravel()
    cx, cy = universal_circle()
    fig_p = go.Figure()
    fig_p.add_trace(go.Scatter(x=cx, y=cy, mode="lines",
                               line=dict(color="black", width=1), name="cerchio universale"))
    fig_p.add_trace(go.Scatter(
        x=g.ravel(), y=s.ravel(), mode="markers",
        marker=dict(size=4, color=dominant, colorscale="Viridis", opacity=0.5),
        name="pixel (colore = specie dominante)",
    ))
    fig_p.add_trace(go.Scatter(
        x=refs[:, 0], y=refs[:, 1], mode="markers+text",
        marker=dict(size=14, color="red", symbol="star"),
        text=[f"{tau:.1f} ns" for tau in flim.lifetimes],
        textposition="top center", name="specie pure",
    ))
    fig_p.update_layout(height=380, margin=dict(l=0, r=0, t=10, b=0),
                        xaxis_title="g", yaxis_title="s")
    fig_p.update_yaxes(scaleanchor="x", scaleratio=1)
    st.plotly_chart(fig_p, use_container_width=True)

st.subheader("Mappe di frazione: stima (phasor) vs ground truth")
cols = st.columns(k)
for i in range(k):
    with cols[i]:
        st.markdown(f"**Specie {flim.lifetimes[i]:.1f} ns** — MAE {mae_k[i]:.3f}")
        fig = go.Figure(go.Heatmap(z=est[..., i], colorscale="Magma", zmin=0, zmax=1,
                                   showscale=False))
        fig.update_layout(height=240, margin=dict(l=0, r=0, t=4, b=0),
                          yaxis_autorange="reversed")
        st.plotly_chart(fig, use_container_width=True)
        fig_gt = go.Figure(go.Heatmap(z=flim.fractions[..., i], colorscale="Magma",
                                      zmin=0, zmax=1, showscale=False))
        fig_gt.update_layout(height=240, margin=dict(l=0, r=0, t=4, b=0),
                             yaxis_autorange="reversed")
        st.caption("ground truth")
        st.plotly_chart(fig_gt, use_container_width=True)

st.metric("MAE medio frazioni (↓ meglio)", f"{mae:.4f}")
st.info(
    "L'unmixing fasoriale è **lineare e fit-free**: nessuna ottimizzazione non "
    "lineare per pixel. Aumenta i fotoni per vedere il MAE scendere."
)
