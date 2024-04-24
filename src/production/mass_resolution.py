"""Different mass resolution fits taken from different sources"""

from ._polynomial import polynomial

alic_2016_simps = polynomial(1.06314, 3.45955e-02, -6.62113e-5)
"""Copied from Alic's 2016 SIMPs reach scripts"""

tom_2016_vanilla_rereach = polynomial(9.10174583e-01, 2.94057744e-02, 8.63423108e-05)
"""Tom produced with KF trackingon 2016 displaced vanilla A' signal files

copied in 4/24/2024"""