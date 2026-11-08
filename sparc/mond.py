"""MOND acceleration, shared by every script that compares against MOND."""
import numpy as np


def g_mond(g_bar, g_dagger=1.2e-10):
    """Total acceleration from the baryonic one: g_tot = g_bar / (1 - exp(-sqrt(g_bar / g_dagger))). Same units for both, g_bar >= 0.

    At g_bar = 0 the expression is 0/0; its limit sqrt(g_bar * g_dagger) -> 0 is returned.
    """
    g_bar = np.asarray(g_bar, dtype=float)
    denominator = -np.expm1(-np.sqrt(g_bar / g_dagger))
    return np.divide(g_bar, denominator, out=np.zeros_like(g_bar), where=g_bar > 0)
