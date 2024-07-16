"""different radiative fraction estimates from different sources"""

from ._polynomial import polynomial

alic_2016_simps_old = polynomial(
    -1.04206e-1, 9.992547e-3, -1.99437e-4, 1.83534e-6, -7.93138e-9, 1.30456e-11
)
"""Alic produced with KF tracking on 2016 background files"""

alic_2016_simps = polynomial(
    0.10541434, -0.0011737697, 7.4487930e-06, -1.6766332e-08
)
"""Updated fit after tuning of selection, July 2024"""

tom_2016_vanilla_rereach = polynomial(
    -2.12037976e-01, 1.58881667e+01, -3.23546649e+02, 3.03753504e+03,
    -1.33484525e+04, 2.20606661e+04,
    x_units = 1000.
)
"""Tom produced with KF tracking on 2016 background files

copied in 4/23/2024"""