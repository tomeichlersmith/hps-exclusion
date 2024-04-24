"""base physics functions and constants useful for other models"""


c = 3.00e11 #mm/s
hbar = 6.58e-22  # MeV*sec
electron_mass = 0.511 # MeV
alpha = 1.0 / 137.0

    
def Beta(x, y):
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
    return (c * hbar / rate)