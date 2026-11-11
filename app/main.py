"""Phasor FLIM Explorer: interactive app.

Run with:  uv run streamlit run app/main.py

Interactively explore the phasor transform of synthetic FLIM data:
- adjust noise (photons) and harmonic order
- view intensity, phasor plot and estimated fraction maps
- compare the accuracy (MAE) against the ground truth
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
    "Fit-free phasor analysis of synthetic FLIM data with ground truth. "
    "Phasor transform → linear unmixing of species fractions."
)

with st.sidebar:
    st.header("Parameters")
    photons = st.select_slider(
        "Photons per pixel (↑ = less noise)",
        options=[200, 500, 1000, 2000, 5000, 10000],
        value=2000,
    )
    harmonic = st.radio("Harmonic order", [1, 2], horizontal=True)
    size = st.select_slider("Image size", options=[32, 48, 64, 96], value=64)
    seed = st.number_input("Seed", value=0, step=1)
    st.markdown("---")
    st.markdown("Species lifetimes (ns): **0.6 · 2.4 · 4.0** @ 80 MHz")


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
    st.subheader("Integrated intensity")
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
                               line=dict(color="black", width=1), name="universal circle"))
    fig_p.add_trace(go.Scatter(
        x=g.ravel(), y=s.ravel(), mode="markers",
        marker=dict(size=4, color=dominant, colorscale="Viridis", opacity=0.5),
        name="pixels (color = dominant species)",
    ))
    fig_p.add_trace(go.Scatter(
        x=refs[:, 0], y=refs[:, 1], mode="markers+text",
        marker=dict(size=14, color="red", symbol="star"),
        text=[f"{tau:.1f} ns" for tau in flim.lifetimes],
        textposition="top center", name="pure species",
    ))
    fig_p.update_layout(height=380, margin=dict(l=0, r=0, t=10, b=0),
                        xaxis_title="g", yaxis_title="s")
    fig_p.update_yaxes(scaleanchor="x", scaleratio=1)
    st.plotly_chart(fig_p, use_container_width=True)

st.subheader("Fraction maps: estimate (phasor) vs ground truth")
cols = st.columns(k)
for i in range(k):
    with cols[i]:
        st.markdown(f"**Species {flim.lifetimes[i]:.1f} ns** — MAE {mae_k[i]:.3f}")
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

st.metric("Mean fraction MAE (↓ better)", f"{mae:.4f}")
st.info(
    "Phasor unmixing is **linear and fit-free**: no per-pixel nonlinear "
    "optimization. Increase the photons to see the MAE go down."
)
