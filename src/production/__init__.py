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

from ._trident_differential_production import TridentDifferentialProduction

from . import mass_resolution
from . import radiative_acceptance
from . import radiative_fraction


def from_calculators(
    rad_frac,
    tdp,
    rad_acc = None
):
    """Construct a function which takes an input dark photon mass
    and outputs the estimate total number of signal events using
    the input calculators


    Parameters
    ----------
    rad_frac: Callable
        function that calculates the radiative fraction given a dark photon mass
    tdp: Callable
        function that calculates the trident differential production given
        a dark photon mass
    rad_acc: Callable, optional
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

    if isinstance(rad_frac,str):
        if getattr(radiative_fraction, rad_frac, None) is None:
            options = [
                n
                for n in dir(radiative_fraction)
                if not n.startswith('_') and n != 'polynomial'
            ]
            raise ValueError(f'{rad_frac} is not one of the radiative_fraction options defined in this package ({options})')
        rad_frac = getattr(radiative_fraction, rad_frac)


    if rad_acc is None:
        def _impl(mass):
            return (
                (3. * (137./2.) * np.pi)
                * mass * rad_frac(mass)
                * tdp(mass)
            )
        _impl.__doc__ = doc
        return _impl
    else:
        def _impl(mass):
            return (
                (3. * (137./2.) * np.pi)
                * mass * rad_frac(mass)
                * tdp(mass)
                / rad_acc(mass)
            )
        _impl.__doc__ = doc
        return _impl

