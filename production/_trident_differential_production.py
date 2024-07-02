"""trident differential production estimation

In order to estimate the trident differential production,
we count the number of events within a control region that fall
within a mass resolution window of the mass we want to estimate
(and then divide by the width of that window). This requires
some input data or simulation in order to do this event counting
and requires an estimate of the mass resolution.
"""

from dataclasses import dataclass
import pathlib
import sys
import pickle


import numpy as np
import uproot


_cache_location = (
    pathlib.Path(__file__).parent.resolve()
    / 'trident_differential_production_cache.pkl'
)


@dataclass
class TridentDifferentialProduction:
    """estimate the trident differential production
    using reference data files to construct the dN/dm distribution

    See Also
    --------
    load: for how the dN/dm distribution is constructed from the files
    __call__: for how the values are take from the distribution given a mass

    One can check the distribution directly by plotting it as a simple histogram

        import mplhep
        mplhep.histplot(tdp.distribution)

    Attributes
    ----------
    bin_edges: np.array
        edges of bins defining dN/dm distribution
    bin_values: np.array
        values of dN/dm distribution in those bins
    """
    
    
    bin_edges: np.array
    bin_values: np.array


    @classmethod
    def delete_cache(_cls):
        """Delete the cache file, necessary when changing arguments to load

        Hopefully, when I have time, I can implement a functional hashing algorithm
        so that this re-creation of the cache file is done automatically.
        """
        _cache_location.unlink(missing_ok=True)


    @classmethod
    def load(
        cls,
        reference,
        mass_maximum_MeV,
        *,
        mass_window_width = 1.0,
        mass_branch = 'unc_vtx_mass',
        mass_branch_to_MeV = 1000.,
        cr_cut = None,
    ):
        """estimate the trident differential production
        using the input reference data file to construct the dN/dm distribution
    
        Parameters
        ----------
        reference: files input to [uproot.concatenate](https://uproot.readthedocs.io/en/latest/uproot.behaviors.TBranch.concatenate.html)
            Specification of ROOT TTree that you want to use as the reference for
            filling the histogram modeling the distribution
        mass_maximum_MeV: float
            maximum invariant mass to extend the distribution to in MeV
        mass_window_width: float, optional
            width of a mass window to sum over in MeV
            default is 1.0
        mass_branch: str, optional
            name of branch in input reference TTree to interpret as the vtx mass
            default is 'unc_vtx_mass'
        mass_branch_to_MeV: float, optional
            conversion factor to multiply values of mass_branch to get them into units of MeV
            default is 1000.0
        """
        if _cache_location.is_file():
            with open(_cache_location, 'rb') as f:
                return pickle.load(f)
        bkgd_CR = uproot.concatenate(
            reference,
            expressions = [ mass_branch ],
            cut = cr_cut,
            library = 'np'
        )
        # shifting by half window width so the bin centers are
        # np.arange(0.0, mass_maximum_MeV+mass_window_width, mass_window_width)
        counts, bin_edges = np.histogram(
            bkgd_CR[mass_branch]*mass_branch_to_MeV,
            bins = np.arange(
                -mass_window_width/2,
                mass_maximum_MeV+3*mass_window_width/2,
                mass_window_width
            )
        )
        widths = bin_edges[1:]-bin_edges[:-1]

        tdp = cls(
            bin_edges = bin_edges,
            bin_values = counts/widths,
        )
        
        with open(_cache_location, 'wb') as f:
            pickle.dump(tdp, f)
        
        return tdp


    def __call__(self, mass):
        """look up the differential production corresponding to the input dark photon mass"""
        bin_indices = np.digitize(mass, bins = self.bin_edges)-1
        return self.bin_values[bin_indices]


    @property
    def distribution(self):
        """return the distribution histogram as if it was a numpy histogram tuple"""
        return self.bin_values, self.bin_edges
