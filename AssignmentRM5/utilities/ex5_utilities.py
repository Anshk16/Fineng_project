import numpy as np
import pandas as pd
import datetime as dt
from typing import Tuple, List

from scipy.integrate import quad

from utilities.ex1_utilities import (
    year_frac_act_x,
)

def affine_trick(
    s: dt.datetime,
    today: dt.datetime,
    pricing_grid: List[dt.datetime],
    mean_reversion: float,
    sigma: float,
    new_discount: pd.Series,
) -> Tuple[pd.Series, pd.Series]:
    """
    Affine trick: Exploits the affine structure of the Hull-White model. Output the functions
    A(s, t_j) and C(s, t_j) pre-computed on xthe pricing grid s.t.
    B(s, t_j) = A(s, t_j) * exp(-C(s, t_j) * x(s)).

    Parameters:
        s (dt.datetime): Date w.r.t. which the computations are made.
        pricing_grid (List[dt.datetime]): Pricing grid.
        mean_reversion (float): Hull-white mean reversion speed.
        sigma (float): Hull-white interest rate volatility.
        new_discount (pd.Series): Discount factors curve.

    Returns:
        Tuple[pd.Series, pd.Series]: Tuple with the precomputed functions A(t, t_j) and C(t, t_j).
    """

    A = pd.Series(index=pricing_grid)
    C = pd.Series(index=pricing_grid)

    delta_s = year_frac_act_x(today,s, 365)
    delta = np.array([year_frac_act_x(s, pricing_grid[i], 365) for i in range(len(pricing_grid))])

    for t in range(len(pricing_grid)):
        if pricing_grid[t] > s:
            C.iloc[t] = (1 - np.exp( - mean_reversion * delta[t])) / mean_reversion
            integral = quad(lambda u: - (sigma / mean_reversion * 1 - np.exp( - mean_reversion * u)) ** 2
                           + (sigma / mean_reversion * 1 - np.exp( - mean_reversion * (u + delta[t]))) ** 2,
                            0, delta_s)[0]
            index = pricing_grid.index(s)
            A.iloc[t] = (new_discount[t] / new_discount[index]) * np.exp( - 0.5 * integral)
        else:
            A.iloc[t] = 1
            C.iloc[t] = 0

    return A, C