"""base physics functions and constants useful for other models

Attributes
----------
c : float
    speed of light in mm/s
hbar : float
    reduced Planck's constant in MeV s
electron_mass: float
    mass of electron in MeV
alpha: float
    fine structure constant in the low energy limit
"""

import math

c = 3.00e11 #mm/s
hbar = 6.58e-22  # MeV*sec
electron_mass = 0.511 # MeV
alpha = 1.0 / 137.0

    
def Beta(x, y):
    """used in some decay rates, it is helpful to isolate this beta function here

    $$
        \beta(x,y) = (1+y^2-x^2-2y)(1+y^2-x^2+2y)
    $$
    """
    return (
        1
        + math.pow(y, 2)
        - math.pow(x, 2)
        - 2 * y
    ) * (
        1
        + math.pow(y, 2)
        - math.pow(x, 2)
        + 2 * y
    )


def ctau(rate):
    """Calculate a decay length from a decay rate

    We use c [mm/s] and hbar [MeV s] so the input rate should be
    in units of MeV and the output decay length will be in mm.

    $$
        c\tau = \frac{c \hbar}{\Gamma}
    $$
    """
    return (c * hbar / rate)
