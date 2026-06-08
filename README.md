# 🔬 Phasor FLIM Explorer

> Analisi **fasoriale fit-free** di dati FLIM (Fluorescence Lifetime Imaging Microscopy): trasformata phasor, unmixing grafico delle specie e confronto quantitativo col fitting esponenziale classico — il tutto su dati sintetici con ground truth, eseguibile senza GPU.

![phasor](results/phasor_plot.png)

## Cosa dimostra

L'**approccio fasoriale** trasforma ogni decadimento di fluorescenza in un punto `(g, s)` del piano phasor tramite una singola coppia seno/coseno — **senza alcun fitting non lineare**. Un decadimento mono-esponenziale cade sul *cerchio universale*; una miscela di specie cade nella loro combinazione lineare pesata dalle frazioni di fotoni. Questo rende l'**unmixing** (quante frazioni di ciascuna specie ci sono in ogni pixel) un problema lineare, veloce e robusto al rumore di conteggio fotonico tipico del FLIM.

Il progetto mette a confronto il phasor con il **fitting del decadimento** (NNLS sulla curva intera) usando la stessa informazione a priori, isolando così il vantaggio del *metodo*.

## Pipeline

```
synth  →  phasor  →  unmix  →  metrics
  │         │          │          │
  │         │          │          └─ MAE vs ground-truth
  │         │          └─ frazioni per pixel (lineare, fit-free)
  │         └─ mappe (g, s) + cerchio universale
  └─ FLIM sintetico K-specie con frazioni note + rumore Poisson
            ▲
            └─ baseline: fitting NNLS del decadimento (confronto)
```

## Dataset

**Sintetico, generato localmente** (`scripts/download_data.py` → `make_synthetic_flim`): miscele di K fluorofori con vite medie note (default 0.6 / 2.4 / 4.0 ns a 80 MHz), mappe di frazione spaziali come ground truth e rumore Poissoniano realistico. Nessun problema di licenza; la pipeline è progettata per accettare anche dati FLIM reali (stack `H×W×T`).

> Estendibile a dati FLIM pubblici reali sostituendo il loader; il formato atteso è uno stack `(H, W, T)` + asse dei tempi e frequenza del laser.

## Come eseguire (uv)

```bash
uv sync --extra dev
uv run pytest -q                         # test di correttezza (phasor + unmixing)
uv run python scripts/evaluate.py        # genera metriche e figure in results/
uv run streamlit run app/main.py         # explorer interattivo
```

## Risultati (dati sintetici)

`scripts/evaluate.py` confronta unmixing fasoriale vs fitting NNLS sul MAE delle
frazioni, in funzione del numero di fotoni (rumore). Metriche e figure complete
in `results/` (`metrics.json`, `phasor_plot.png`, `mae_vs_photons.png`).

**MAE delle frazioni ↓** (3 specie, 80 MHz, rumore Poisson):

| Fotoni/pixel | Phasor (fit-free) | Fitting NNLS |
|-------------:|:-----------------:|:------------:|
| 300  | **0.100** | 0.136 |
| 800  | **0.071** | 0.096 |
| 2000 | **0.050** | 0.065 |
| 5000 | **0.037** | 0.042 |

Il phasor è più accurato a **ogni** livello di rumore ed è in media **~32× più veloce** del fitting NNLS — pur usando la stessa informazione a priori (le vite di riferimento).

![mae](results/mae_vs_photons.png)

## Note metodologiche e limiti

- Le **vite di riferimento** sono assunte note (sia per il phasor sia per il fitting): confronto equo che isola il metodo, non l'informazione.
- Su dati reali servono **calibrazione strumentale** (IRF) e correzione dell'armonica; qui i dati sintetici sono già calibrati.
- L'unmixing lineare assume sovrapposizione di componenti discrete; specie con distribuzioni di vita continue richiedono modelli aggiuntivi.

## Riferimenti

- Digman M.A. et al., *The Phasor Approach to Fluorescence Lifetime Imaging Analysis*, Biophysical Journal 94(2):L14–L16, 2008. DOI: 10.1529/biophysj.107.120154
- Ranjit S. et al., *Fit-free analysis of fluorescence lifetime imaging data using the phasor approach*, Nature Protocols 13:1979–2004, 2018. DOI: 10.1038/s41596-018-0026-5

## Licenza

Codice: MIT (vedi `LICENSE`).
