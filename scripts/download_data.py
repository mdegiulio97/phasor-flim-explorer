"""Genera il dataset FLIM sintetico e lo salva in data/ (idempotente).

I dati sono sintetici (nessuna licenza esterna). Lo stack e la ground truth sono
salvati in un singolo .npz. Rilanciare lo script non rigenera se il file esiste.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from phasor_flim.synth import make_synthetic_flim

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
OUT = DATA_DIR / "synthetic_flim.npz"


def main() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    if OUT.exists():
        print(f"Dataset già presente: {OUT}")
        return
    flim = make_synthetic_flim(seed=0)
    np.savez_compressed(
        OUT,
        decay=flim.decay,
        t=flim.t,
        fractions=flim.fractions,
        lifetimes=flim.lifetimes,
        freq_mhz=flim.freq_mhz,
    )
    print(f"Salvato dataset sintetico: {OUT}  (shape decay={flim.decay.shape})")


if __name__ == "__main__":
    main()
