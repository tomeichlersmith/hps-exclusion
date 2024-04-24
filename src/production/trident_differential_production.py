"""trident differential production estimation

In order to estimate the trident differential production,
we count the number of events within a control region that fall
within a mass resolution window of the mass we want to estimate
(and then divide by the width of that window). This requires
some input data or simulation in order to do this event counting
and requires an estimate of the mass resolution.
"""

import uproot
import awkward as ak

def lut_estimate(
    reference,
    mass_range,
    *,
    mass_window_width = 1.0,
    mass_branch = 'unc_vtx_mass'
):
    """construct a function that estimates the trident differential production
    using the input reference data file and mass resolution calculator.

    The Look Up Table (LUT) is loaded by a function attached to the constructed
    estimate function and is only called when needed (i.e. on the first call
    to the estimate itself). This is important to note because if you are switching
    the file that you want to use to make the estimate, you will probably also
    want to test that the loading functions as expected. I would suggest making
    a simple plot with the estimator as a check.

        import matplotlib.pyplot
        masses = np.arange(40,200,5)
        tdp = trident_differential_production.estimate(
            '/path/to/file.root',
            masses
        )
        plt.plot(
            masses,
            [tdp_est(m) for m in masses]
        )

    Parameters
    ----------
    reference: files inpput to [uproot.concatenate](https://uproot.readthedocs.io/en/latest/uproot.behaviors.TBranch.concatenate.html)
        Specification of ROOT TTree that you want to use as the reference for
        calculating the LUT
    mass_range: Iterable | list
        range of production masses to sample for in MeV
    mass_window_width: float, optional
        width of a mass window to sum over in MeV
    mass_conversion: float, optional
        conversion between dark photon mass (whose production is proportional
        to trident differential production) and mass input to mass resolution
    mass_branch: str, optional
        name of branch in input reference TTree to interpret as the vtx mass
    """

    def _lut_loader():
        dNdm_by_mass = {}
        bkgd_CR = uproot.concatenate(
            reference,
            expressions = [ mass_branch ]
        )
        for mass in mass_range:
            dNdm_by_mass[mass] = ak.sum(
                (bkgd_CR[mass_branch]*1000 > (mass - mass_window_width/2))&
                (bkgd_CR[mass_branch]*1000 < (mass + mass_window_width/2))
            )/mass_window_width
        return dNdm_by_mass

    
    def _estimate_impl(mass):
        """look up the differential production corresponding to the input dark vector mass

        Right now, we error out if the dark vector mass (in MeV and cast to an int) is not found
        in the lookup table loaded and attached to this function. I could imagine a future where
        we instead do some interpolation to avoid having a super large table in memory.
        """
    
        # load the LUT on the first call of this function
        # this means the first call might take some time but subsequent calls will use
        # the now in-memory LUT
        if getattr(_estimate_impl, '__lut__', None) is None:
            _estimate_impl.__lut__ = _estimate_impl.__lut_loader()
        
        if mass in _estimate_impl.__lut__:
            return _estimate_impl.__lut__[mass]
        # right now, erroring-out if choosing a different mass, but
        # we could do some interpolation if desired
        raise ValueError(
            f'The mass {mass} is not found in the trident differential production look-up table.'
        )

    _estimate_impl.__lut_loader = _lut_loader
    return _estimate_impl
