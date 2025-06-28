"""
Mathematical Engineering - Financial Engineering, FY 2024-2025
Risk Management - Exercise 3: Equity Portfolio VaR/ES and Counterparty Risk
"""
import math
from enum import Enum
import numpy as np
import pandas as pd
from QuantLib import Payoff
import datetime as dt
from pywin.mfc.docview import ListView
from scipy.stats import norm
from typing import Optional, Tuple, Union, Iterable, List
from numpy.typing import NDArray

from utilities.bootstrap import Discounts
from utilities.ex1_utilities import year_frac_act_x


class OptionType(Enum):
    """
    Types of options.
    """

    CALL = "call"
    PUT = "put"


def black_scholes_option_pricer(
        S: float,
        K: float,
        ttm: float,
        r: float,
        sigma: float,
        d: float,
        option_type: OptionType = OptionType.PUT,
        return_delta_gamma: bool = False,
) -> Union[float, Tuple[float, float, float]]:
    """
    Return the price (and possibly delta and gamma) of an option according to the Black-Scholes
    formula.

    Parameters:
        S (float): Current stock price.
        K (float): Strike price.
        ttm (float): Time to maturity.
        r (float): Risk-free rate.
        sigma (float): Implied volatility.
        d (float): Dividend yield.
        option_type (OptionType, {'put', 'call'}): Option type, default to put.
        return_delta_gamma (bool): If True the option delta and gamma are returned.

    Returns:
        Union[float, Tuple[float, float, float]]: Option price (and possibly delta and gamma).
    """

    d1 = (np.log(S / K) + (r - d + sigma ** 2 / 2) * ttm) / (sigma * np.sqrt(ttm))  # ERROR 1
    d2 = d1 + sigma * np.sqrt(ttm)

    if option_type == OptionType.CALL:
        if return_delta_gamma:
            return (
                S * np.exp(-d * ttm) * norm.cdf(d1)
                - K * np.exp(-r * ttm) * norm.cdf(d2),
                np.exp(-d * ttm) * norm.cdf(d1),
                np.exp(-d * ttm) * norm.pdf(d1) / (S * sigma * ttm ** (0.5)),
            )
        else:
            return S * np.exp(-d * ttm) * norm.cdf(d1) - K * np.exp(
                -r * ttm
            ) * norm.cdf(d2)
    elif option_type == OptionType.PUT:
        if return_delta_gamma:
            return (
                K * np.exp(-r * ttm) * norm.cdf(-d2)
                - S * np.exp(-d * ttm) * norm.cdf(-d1),
                -np.exp(-d * ttm) * norm.cdf(-d1),
                -np.exp(d * ttm) * norm.pdf(-d1) * (-1 / (S * sigma * ttm ** (0.5))),
            )
        else:
            return K * np.exp(-r * ttm) * norm.cdf(-d2) - S * np.exp(
                -d * ttm
            ) * norm.cdf(-d1)
    else:
        raise ValueError("Invalid option type.")


def principal_component_analysis(
        matrix: NDArray[np.float64],
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Given a matrix, returns the eigenvalues vector and the eigenvectors matrix.
    """

    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    # Sorting from greatest to lowest the eigenvalues and the eigenvectors
    sort_indices = eigenvalues.argsort()[::-1]

    return eigenvalues[sort_indices], eigenvectors[:, sort_indices]


def gaussian_var_es(
        mu: pd.Series,
        sigma: pd.DataFrame,
        alpha: float,
        weights: pd.Series,
        ptf_notional: float = 1e6,
        delta: float = 1,
) -> Tuple[float, float]:
    """
    Return VaR and ES computed via Gaussian parametric approach according to the following formulas:
        VaR_{alpha} = delta * mu + sqrt(delta) * sigma * VaR^{std}_{alpha}, where
            VaR^{std}_{alpha} = N^{-1}(alpha) and N is the standard normal cumulative distribution
            function.
        ES_{alpha} = delta * mu + sqrt(delta) * sigma * ES^{std}_{alpha}, where
            ES^{std}_{alpha} = phi(N^{-1}(alpha)) / (1 - alpha) and phi is the standard normal
            probability density function.

    Parameters:
        mu (pd.Series): Series of mean returns.
        sigma (pd.DataFrame): Returns covariance matrix.
        alpha (float): Confidence level.
        weights (pd.Series): Portfolio weights (considered unchanged).
        ptf_notional (float): Portfolio notional, default to 1MM.
        delta (float): Scaling factor, default to 1 i.e. no adjusment is performed.

    Returns:
        Tuple[float, float]: VaR and ES.
    """

    VaR_std = norm.ppf(alpha)
    ES_std = norm.pdf(norm.ppf(alpha)) / (1 - alpha)

    VaR = ptf_notional * (delta * weights @ mu + np.sqrt(delta) * np.sqrt(weights @ sigma @ weights) * VaR_std)

    ES = ptf_notional * (delta * weights @ mu + np.sqrt(delta) * np.sqrt(weights @ sigma @ weights) * ES_std)


    return VaR, ES


def hs_var_es(
        returns: pd.DataFrame,
        alpha: float,
        weights: pd.Series,
        ptf_notional: float = 1e6,
        delta: float = 1,
        lambda_: Optional[float] = None,
) -> Tuple[float, float]:
    """
    Return VaR and ES computed via possibly weighted historical simulation:

    Parameters:
        returns (pd.DataFrame): Returns.
        alpha (float): Confidence level.
        weights (pd.Series): Portfolio weights (considered unchanged).
        ptf_notional (float): Portfolio notional, default to 1MM.
        delta (float): Scaling factor, default to 1 i.e. no adjusment is performed.
        lambda (Optional[float]): Decay factor for weighted historical simulation, default to None
            i.e. standard historical simulation is performed.

    Returns:
        Tuple[float, float]: VaR and ES.
    """

    n = len(returns)
    loss = -ptf_notional * returns @ weights
    loss = np.sort(loss)[::-1]

    if lambda_ is None:
        VaR = loss[math.floor(n * (1 - alpha))]
        ES = loss[:math.floor(n * (1 - alpha)) + 1].mean()
    else:
        # Weighted historical simulation
        C = (1 - lambda_) / (1 - lambda_ ** n)
        W_s = C * np.array([lambda_ ** (n - j) for j in range(1, n + 1)])
        W_s = np.sort(W_s)[::-1]  # Sort weights to match sorted losses

        combined = np.column_stack((loss, W_s))
        combined_sorted = combined[combined[:, 0].argsort()[::-1]]
        sorted_loss = combined_sorted[:, 0]
        sorted_weights = combined_sorted[:, 1]

        cum_weights = np.cumsum(sorted_weights)
        flag = np.argmax(cum_weights > (1 - alpha)) if np.any(cum_weights > (1 - alpha)) else len(cum_weights)

        VaR = sorted_loss[flag]
        ES = np.sum(sorted_loss[:flag + 1] * sorted_weights[:flag + 1]) / cum_weights[flag]

    return VaR * delta, ES * delta

def plausility_check(
        returns: pd.DataFrame,
        weights: pd.Series,
        alpha: float,
        ptf_notional: float = 1e6,
        delta: float = 1,
) -> float:
    """
    Perform plausibility check on a portfolio VaR estimating its order of magnitude.

    Parameters:
        returns (pd.DataFrame): Returns.
        weights (pd.Series): Portfolio weights.
        alpha (float): Confidence level.
        ptf_notional (float): Portfolio notional, default to 1MM.
        delta (float): Scaling factor, default to one, i.e. no scaling is performed.

    Returns:
        float: Portfolio VaR order of magnitude.
    """

    sVaR = (
            -ptf_notional
            * weights
            * returns.quantile(q=[1 - alpha, alpha], axis=0).T.abs().sum(axis=1)
            / 2
    )

    return np.sqrt(delta * np.dot(sVaR, np.dot(returns.corr(), sVaR)))


def simulated_stock_MC(
        returns: pd.Series,
        stock_price: float,
        simulations: int = 10000,
        Ndays: int = 10,
) -> float:
    """
        Perform plausibility check on a portfolio VaR estimating its order of magnitude.

        Parameters:
            returns (pd.Series): Returns.
            stock_price (float): Initial stock price.
            simulations (int): Number of Monte Carlo simulations.
            Ndays (int): Number of days.

    Returns:
        float: simulated stock prices
    """
    random_vector = np.zeros(Ndays)
    return_10dd = np.zeros(simulations)

    for i in range(simulations):
        for j in range(Ndays):
            k = np.random.randint(0,len(returns - 1))
            random_vector[j] = returns[k] #10 casual log_returns
        return_10dd[i] = sum(random_vector) #log_return at 10 days

    return stock_price * np.exp(return_10dd), return_10dd


def cliquet_option_pricer(
        S0: float,
        discount_factors: Iterable[float],
        payoff_dates_payments: List[dt.datetime],
        survival_probabilities: List[float],
        sigma: float,
        notional: float,
) -> float:
    """
    Return the price (and possibly delta and gamma) of a Cliquet option.

    Parameters:
        S0 (float): Current stock price.
        discount_factors Iterable[float]: Discount factors.
        payoff_dates_payments List[dt.datetime]: Dates of option payoff.
        survival_probabilities List[float] : Survival probabilities
        sigma (float): Implied volatility.
        notional (float): Notional of the option.

    Returns:
        Union[float, Tuple[float, float, float]]: Option price (and possibly delta and gamma).
    """
    # Computation of risk-free rate
    Npayoffs = len(discount_factors)
    r = np.zeros(Npayoffs)
    delta_t = np.zeros(Npayoffs)
    for i in range(Npayoffs):
        delta_t[i] = year_frac_act_x(payoff_dates_payments[0], payoff_dates_payments[i + 1], 365)
        r[i] = -np.log(discount_factors[i]) / delta_t[i]

    # Monte Carlo simulation
    M = int(1e5)  # Number of simulations
    Payoff = np.zeros((M, Npayoffs))

    Stock_price = np.zeros((M, Npayoffs + 1))
    Stock_price[:0] = S0

    for i in range(Npayoffs):
        for j in range(M):
            Z = np.random.normal()
            Stock_price[j, i + 1] = S0 * np.exp((r[i] - 0.5 * sigma ** 2) * delta_t[i] + sigma * np.sqrt(delta_t[i]) * Z)
            Payoff[j, i] = np.maximum(0, Stock_price[j, i] - Stock_price[j, i + 1])

    # Calculate discounted payoffs
    Discounted_Payoff = np.zeros(Npayoffs)
    for i in range(Npayoffs):
        Discounted_Payoff[i] = np.mean(Payoff[:, i]) * survival_probabilities[i] * discount_factors[i]

    return sum(Discounted_Payoff) * notional
