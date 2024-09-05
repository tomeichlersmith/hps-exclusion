"""Different mass resolution fits taken from different sources"""

from ._polynomial import polynomial

alic_2016_simps_unsmeared = polynomial(1.06314, 3.45955e-02, -6.62113e-5)
"""Copied from Alic's 2016 SIMPs reach scripts"""

alic_2016_simps = polynomial(.75739851, 0.031621002, 5.2949672e-05)
"""Deduced after momentum smearing in hpstr was validated, July 2024"""

tom_2016_vanilla_rereach = polynomial(9.10174583e-01, 2.94057744e-02, 8.63423108e-05)
"""Tom produced with KF trackingon 2016 displaced vanilla A' signal files

copied in 4/24/2024"""

matt_l1l1 = polynomial(0.9348, 0.05442, -5.784e-4, 5.852e-6, -1.724e-8)
"""Eq 22 from Matt's 2016 Vertexing Note"""

tom_2016_simps_l1l1 = polynomial(0.48400635, 0.04214241)
"""Momentum smeared and rad-matched within Psum SR for L1L1 vertices, 2024/09/05"""

tom_2016_simps_l1l2 = polynomial(0.81237597, 0.04217314)
"""Momentum smeared and rad-matched within Psum SR for L1L2 vertices, 2024/09/05"""
