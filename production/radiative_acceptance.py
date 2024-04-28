"""different radiative acceptnace estimates taken from different sources"""

from ._polynomial import polynomial

alic_2016_simps = polynomial(
    -7.35934e-01, 9.75402e-02, -5.22599e-03, 1.47226e-04,
    -2.41435e-06, 2.45015e-08, -1.56938e-10, 6.19494e-13,
    -1.37780e-15, 1.32155e-18
)
