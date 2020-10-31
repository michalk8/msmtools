from typing import Tuple

import numpy as np


def get_known_input(Tc: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    assert not np.allclose(Tc, 0.0), "Tc doesn't seem to be a count matrix."
    assert Tc.dtype == np.float64, "Expected double precision"
    assert Tc.shape[0] == Tc.shape[1], "Expected a square matrix."

    row = np.sum(Tc, axis=1) > 0.0001
    P = Tc.copy()
    Ts = Tc[row, :]
    P[row, :] = Ts @ np.diag(np.true_divide(1.0, np.sum(Ts, axis=0)))

    sd = np.sum(Tc, axis=1)
    sd = np.true_divide(sd, np.sum(sd))

    return P, sd
