"""SIMP Theory Parameters calculating decay rates and branching ratios

Example
-------
First define the parameters you want and then use the object holding
these parameters to do the calculations

    simpeqs = simp.Parameters(
        alpha_dark = 0.01,
        mass_ratio_Ap_to_Vd = 1.66,
        mass_ratio_Ap_to_Pid = 3.0,
        ratio_mPi_to_fPi = 4*3.14159
    )
    simpeqs.br(simpeqs.rate_Vrho_pi, 100)
    simpeqs.ctau(simpeqs.rate_Vd_decay_2l_per_eps2(100))

"""

import math
class Parameters:

    c = 3.00e11 #mm/s
    hbar = 6.58e-22  # MeV*sec
    electron_mass = 0.511 # MeV
    alpha = 1.0 / 137.0

    def __init__(
        self, *,
        alpha_dark = 0.01,
        mass_ratio_Ap_to_Vd = 1.66,
        mass_ratio_Ap_to_Pid = 3.0,
        ratio_mPi_to_fPi = 4*math.pi,
    ):
        self.alpha_dark = alpha_dark
        self.mass_ratio_Ap_to_Vd = mass_ratio_Ap_to_Vd
        self.mass_ratio_Ap_to_Pid = mass_ratio_Ap_to_Pid
        self.ratio_mPi_to_fPi = ratio_mPi_to_fPi

    
    def _masses(self, m, ap = False, vd = False, pid = False):
        """with an input mass scale calculate the three dark sector masses

        one of ap, vd, or pid must be True and it corresponds to which
        of the dark sector masses the input mass scale is.
        """

        # check to make sure exactly one is true
        # https://stackoverflow.com/a/16801605
        i = iter([ap, vd, pid])
        if not (any(i) and not any(i)):
            # using the same iterator variable means
            # that this is true if no True have been found
            # or if more than one was found (any returns after
            # first find of True)
            raise ValueError(f'Need to provide exactly one of ap, vd, or pid, gave {[ap, vd, pid]}')
        # exactly one is true, we are good to go
        m_ap, m_vd, m_pid = None, None, None
        if ap:
            m_ap = m
            m_vd = m_ap/self.mass_ratio_Ap_to_Vd
            m_pid = m_ap/self.mass_ratio_Ap_to_Pid
        elif vd:
            m_vd = m
            m_ap = self.mass_ratio_Ap_to_Vd*m_vd
            m_pid = m_ap/self.mass_ratio_Ap_to_Pid
        elif pid:
            m_pid = m
            m_ap = self.mass_ratio_Ap_to_Pid*m_pid
            m_vd = m_ap/self.mass_ratio_Ap_to_Vd
        return m_ap, m_vd, m_pid
                
    
    def rate_Ap_ee_per_eps2(self, mass_ap):
        r = Parameters.electron_mass/mass_ap
        coeff1 = (Parameters.alpha)/3.0
        coeff2 = (1.0 - 4.0*(r**2))**(0.5)
        coeff3 = (1.0 + 2.0*(r**2))*mass_ap
        return coeff1*coeff2*coeff3


    def rate_2pi(self, m_V):
        m_Ap, m_V, m_pi = self._masses(m_V, vd=True)
        coeff = (2.0 * self.alpha_dark / 3.0) * m_Ap
        pow1 = math.pow((1 - (4 * m_pi * m_pi / (m_Ap * m_Ap))), 3 / 2.0)
        pow2 = math.pow(((m_V * m_V) / ((m_Ap * m_Ap) - (m_V * m_V))), 2)
        return coeff * pow1 * pow2

    
    def rate_2V(self, m_V):
        if self.mass_ratio_Ap_to_Vd < 2.0:
            return 0.0
        raise NotImplemented
        m_Ap, m_V, m_pi = self._masses(m_V, vd=True)
        return alpha_dark / 6.0 * m_Ap * SimpEquations.f(r)

    
    @staticmethod
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


    def _rate_Ap_decay_per_Tv(self, m_V):
        """decay rate of V where the dependence on the outgoing topology
        (the "Tv") is left out"""
        m_Ap, m_V, m_pi = self._masses(m_V, vd=True)
        return (
            self.alpha_dark / (192.0 * math.pow(math.pi, 4))
            * math.pow(self.mass_ratio_Ap_to_Pid, 2)
            * math.pow(self.mass_ratio_Ap_to_Pid/self.mass_ratio_Ap_to_Vd, 2)
            * math.pow(self.ratio_mPi_to_fPi, 4)
            * m_Ap
            * math.pow(
                Parameters.Beta(
                    1/self.mass_ratio_Ap_to_Pid,
                    1/self.mass_ratio_Ap_to_Vd
                ),
                3 / 2.0
            )
        )
    

    def rate_Vrho_pi(self, m_V):
        return (3.0/4.0)*self._rate_Ap_decay_per_Tv(m_V)
    

    def rate_Vphi_pi(self, m_V):
        return (3.0/2.0)*self._rate_Ap_decay_per_Tv(m_V)


    def rate_Vcharged_pi(self, m_V):
        return (18 - (3.0/2.0 + 3.0/4.0))*self._rate_Ap_decay_per_Tv(m_V)
    
    
    def total_rate(self, m_V):
        r = (
            self.rate_Vrho_pi(m_V)
            + self.rate_Vphi_pi(m_V)
            + self.rate_Vcharged_pi(m_V)
            + self.rate_2pi(m_V)
        )
        if self.mass_ratio_Ap_to_Vd > 2.0:
            r += self.rate_2V(m_V)
        return r

    
    def br(self, rate_func, m_V):
        """calculate branching ratio for a given decay rate

        Example
        -------
        
            s = simp.Parameters()
            br_Vphi_pi = s.br(s.rate_Vphi_pi, m_V)

        """
        return rate_func(m_V)/self.total_rate(m_V)


    def rate_Vd_decay_2l_per_eps2(self, m_V, rho):
        m_Ap, m_V, m_pid = self._masses(m_V, vd=True)
        f_pi = m_pid/self.ratio_mPi_to_fPi
        coeff = (16 * math.pi * self.alpha_dark * Parameters.alpha * f_pi**2) / (3 * m_V**2)
        return (
            coeff
            * (m_V**2 / (m_Ap**2 - m_V**2))**2
            * (1 - (4 * Parameters.electron_mass**2 / m_V**2))**0.5
            * (1 + (2 * Parameters.electron_mass**2 / m_V**2))
            * m_V
            * (2 if rho else 1)
        )

    
    @staticmethod
    def ctau(rate):
        return (Parameters.c * Parameters.hbar / rate)
