"""Generate the synthetic FLIM dataset and save it to data/ (idempotent).

The data are synthetic (no external license). The stack and the ground truth are
saved in a single .npz file. Re-running the script does not regenerate the file if
it already exists.
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
        print(f"Dataset already present: {OUT}")
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
    print(f"Saved synthetic dataset: {OUT}  (shape decay={flim.decay.shape})")


if __name__ == "__main__":
    main()
