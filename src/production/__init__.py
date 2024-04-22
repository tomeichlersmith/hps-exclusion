"""estimate total signal production


    import production
    signal_yield_per_eps2 = production.from_calculators(
        production.radiative_fraction.alic_2016_simps,
        production.trident_differential_production.lut_estimate(
            '/path/to/reference.root:root/tree',
            production.mass_resolution.alic_2016_simps,
        ),
        production.radiative_acceptance.alic_2016_simps
    )

"""

import numpy as np

from . import mass_resolution
from . import radiative_acceptance
from . import radiative_fraction
from . import trident_differential_production

def from_calculators(
    radiative_fraction,
    trident_differential_production,
    radiative_acceptance = None
):
    """Construct a function which takes an input dark photon mass
    and outputs the estimate total number of signal events using
    the input calculators


    Parameters
    ----------
    radiative_fraction: Callable
        function that calculates the radiative fraction given a dark photon mass
    trident_differential_production: Callable
        function that calculates the trident differential production given
        a dark photon mass
    radiative_acceptance: Callable, optional
        function that calculates the radiative acceptance given a dark photon mass
        This factor will just be omitted if it is not given
    """

    doc = """Total signal yield given all of the production parameters except 
    epsilon^2

    We **do not** include the factor of epsilon^2. This is helpful since, in 
    exclusion calculations, we are often iterative over several values of 
    epsilon^2 so factoring it out would mean we can pre-calculate the hard stuff
    and then just scale the result by epsilon^2 when choosing a specific epsilon
    value. The actual total signal production would be the return value of this
    function multiplied by epsilon^2 for a specific choice of epsilon.

    Parameters
    ----------
    mass: float
        dark photon mass in the same units as the calculators

    Returns
    -------
    float
        estimate of signal event yield per epsilon^2
    """

    if radiative_acceptance is None:
        def _impl(mass):
            return (
                (3. * (137./2.) * np.pi)
                * mass * radiative_fraction(mass)
                * trident_differential_production(mass)
            )
        _impl.__doc__ = doc
        return _impl
    else:
        def _impl(mass):
            return (
                (3. * (137./2.) * np.pi)
                * mass * radiative_fraction(mass)
                * trident_differential_production(mass)
                / radiative_acceptance(mass)
            )
        _impl.__doc__ = doc
        return _impl

