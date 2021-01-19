from typing import Tuple

import numpy as np


def get_known_input(Tc: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    assert not np.allclose(Tc, 0.0), "Tc doesn't seem to be a count matrix."
    assert Tc.dtype == np.float64, "Expected double precision"
    assert Tc.shape[0] == Tc.shape[1], "Expected a square matrix."

    row = np.sum(Tc, axis=1) > 0.0001
    P = Tc.copy()
    Ts = Tc[row, :]
    P[row, :] = np.diag(np.true_divide(1.0, np.sum(Ts, axis=1))) @ Ts

    sd = np.sum(Tc, axis=1)
    sd = np.true_divide(sd, np.sum(sd))

    return P, sd


def mu(mu: int):
    return np.array(
        [
            [1000, 100, 100, 10, 0, 0, 0, 0, 0],
            [100, 1000, 100, 0, 0, 0, 0, 0, 0],
            [100, 100, 1000, 0, mu, 0, 0, 0, 0],
            [10, 0, 0, 1000, 100, 100, 10, 0, 0],
            [0, 0, mu, 100, 1000, 100, 0, 0, 0],
            [0, 0, 0, 100, 100, 1000, 0, mu, 0],
            [0, 0, 0, 10, 0, 0, 1000, 100, 100],
            [0, 0, 0, 0, 0, mu, 100, 1000, 100],
            [0, 0, 0, 0, 0, 0, 100, 100, 1000],
        ],
        dtype=np.float64,
    )
