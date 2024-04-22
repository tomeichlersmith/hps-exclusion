"""different radiative fraction estimates from different sources"""

from ._polynomial import polynomial

alic_2016_simps = polynomial(
    -1.04206e-1, 9.992547e-3, -1.99437e-4, 1.83534e-6, -7.93138e-9, 1.30456e-11
)