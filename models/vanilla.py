"""Vanilla dark photon model"""

from . import _general


import numpy as np


def rate_Ap_ee_per_eps2(mass_ap):
    """The decay rate of A' to two electrons

    Take from https://arxiv.org/pdf/2005.01515
    Eq 3.2
    """
    r = (_general.electron_mass/mass_ap)**2
    return (
        _general.alpha/3.0
        * mass_ap
        * np.sqrt(1 - 4*r)
        * (1 + 2*r)
    )