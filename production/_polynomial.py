"""construct a polynomial from the input coefficients"""

def polynomial(
    *coefficients,
    x_units = None
):
    """construct a polynomial from the input coefficients

    The coeffiecients are in order meaning the power they are
    multiplied by corresponds to the indexy they are in the
    array.
    
    Parameters
    ----------
    *coefficents: list[float | int]
        coefficients in order for the power series polynomial
    x_units: float | int | None
        optionally change the units of x by dividing all x in
        the series by this value

    Returns
    -------
    Callable:
        function that acts as the power series using the input
        coefficients

    Example
    -------
    If we wanted to define a function that calculates 2 + 3x + 4x^2
    
        f = polynomial(2, 3, 4)
    
    If our fit was constructed in GeV but our mass is in MeV, we would
    want to change the x_units to be 1000..

        f = polynomail(2, 3, 4, x_units = 1000.)
    """

    if x_units is None:
        def _series_impl(x):
            return sum([
                coefficient * x**power
                for power, coefficient in enumerate(coefficients)
            ])
        return _series_impl
    else:
        def _series_impl(x):
            return sum([
                coefficient * (x/x_units)**power
                for power, coefficient in enumerate(coefficients)
            ])
        return _series_impl