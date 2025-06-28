"""
Mathematical Engineering - Financial Engineering, FY 2024-2025
Risk Management - Exercise 2: Corporate Bond Portfolio
"""

from enum import Enum
import scipy
import calendar
from typing import Iterable, Union, List, Tuple
from scipy.interpolate import interp1d,CubicSpline
from scipy.stats import norm
from scipy.integrate import quad, trapezoid, simpson

from typing import Union
import numpy as np
import pandas as pd
import datetime as dt
from utilities.ex1_utilities import (
    year_frac_act_x,
    business_date_offset,
    date_series,
    bootstrap,
    get_discount_factor_by_zero_rates_linear_interp,year_frac_30_360
)


def bond_cash_flows(
    ref_date: Union[dt.date, pd.Timestamp],
    expiry: Union[dt.date, pd.Timestamp],
    coupon_rate: float,
    coupon_freq: int,
    notional: float = 1.0,
) -> pd.Series:
    """
    Calculate the cash flows of a bond.

    Parameters:
    ref_date (Union[dt.date, pd.Timestamp]): Reference date.
    expiry (Union[dt.date, pd.Timestamp]): Bond's expiry date.
    coupon_rate (float): Coupon rate.
    coupon_freq (int): Coupon frequency in payments per years.
    notional (float): Notional amount.

    Returns:
        pd.Series: Bond cash flows.
    """

    # Payment dates
    cash_flows_dates = []
    payment_dt = expiry
    counter = 1
    while payment_dt > ref_date:
        cash_flows_dates.append(payment_dt)

        payment_dt = business_date_offset(
            expiry, month_offset=(-12 * counter) // coupon_freq
        )
        counter += 1

    cash_flows_dates = cash_flows_dates.sort()

    # Coupon payments
    cash_flows = pd.Series(
        data=[notional * coupon_rate / coupon_freq] * len(cash_flows_dates),
        index=cash_flows_dates,
    )

    # Notional payment
    cash_flows[expiry] += notional

    return cash_flows


def defaultable_bond_dirty_price_from_intensity(
    ref_date: Union[dt.date, pd.Timestamp],
    expiry: Union[dt.date, pd.Timestamp],
    coupon_rate: float,
    coupon_freq: int,
    dates: List[dt.datetime],
    recovery_rate: float,
    intensity: Union[float, pd.Series],
    discount_factors: pd.Series,
    flag: int,
    notional: float = 1.0,
) -> float:
    """
    Calculate the dirty price of a defaultable bond neglecting the recovery of the coupon payments.

    Parameters:
    ref_date (Union[dt.date, pd.Timestamp]): Reference date.
    expiry (Union[dt.date, pd.Timestamp]): Bond's expiry date.
    coupon_rate (float): Coupon rate.
    coupon_freq (int): Coupon frequency in payments a years.
    dates (List[dt.datetime]): List of dates for discount factor lookup.
    recovery_rate (float): Recovery rate.
    intensity (Union[float, pd.Series]): Intensity, can be the average intensity (float) or a
        piecewise constant function of time (pd.Series).
    discount_factors (pd.Series): Discount factors.
    notional (float): Notional amount.
    flag(int): integral mode: 0 quadratic, 1 trapezoid, 2 Simpson, 3 intensity is constant

    Returns:
        float: Dirty price of the bond.
    """

    discountdates = date_series(ref_date, expiry, coupon_freq)

    discounts = [
        get_discount_factor_by_zero_rates_linear_interp(ref_date, k, dates, discount_factors)
        for k in discountdates
    ]

    deltas = [year_frac_30_360(discountdates[i - 1],discountdates[i]) for i in range(1, len(discountdates))]
    deltas2 = [year_frac_30_360(discountdates[0],discountdates[i]) for i in range(1, len(discountdates))]
    l = len(discounts)
    survival_probabilities = np.zeros(l + 1)
    price_defaultable_zcb_with_zero_recovery = np.zeros(l + 1)
    survival_probabilities[0] = 1 # by definition
    price_defaultable_zcb_with_zero_recovery[0] = 1 # by definition

    for k in range(1,l):

        if flag == 0:
            survival_probabilities[k] = np.exp( - scipy.integrate.quad(intensity,ref_date,discountdates[k]))
        
        elif flag == 1:
            survival_probabilities[k] = np.exp( - scipy.integrate.trapezoid(intensity,ref_date,discountdates[k]))

        elif flag == 2:
            survival_probabilities[k] = np.exp( - scipy.integrate.simpson(intensity,ref_date,discountdates[k]))
        
        elif flag == 3:
            survival_probabilities[k] = np.exp( - intensity * deltas2[k-1])

        else: 
            return None
        
        price_defaultable_zcb_with_zero_recovery[k] = survival_probabilities[k]*discounts[k]

    
    e = discounts[1:]*np.array([survival_probabilities[k-1] - survival_probabilities[k] for k in range(1,l)])
    dirty_price = coupon_rate * sum(deltas*price_defaultable_zcb_with_zero_recovery[1:]) + price_defaultable_zcb_with_zero_recovery[-1] + recovery_rate * sum(e)
    
    return dirty_price*notional 


def defaultable_bond_dirty_price_from_z_spread(
    ref_date: Union[dt.date, pd.Timestamp],
    expiry: Union[dt.date, pd.Timestamp],
    coupon_rate: float,
    coupon_freq: int,
    dates: List[dt.datetime],
    z_spread: float,
    discount_factors: pd.Series,
    flag: int,
    notional: float = 1.0,
) -> float:
    """
    Calculate the dirty price of a defaultable bond from the Z-spread.

    Parameters:
    ref_date (Union[dt.date, pd.Timestamp]): Reference date.
    expiry (Union[dt.date, pd.Timestamp]): Bond's expiry date.
    coupon_rate (float): Coupon rate.
    coupon_freq (int): Coupon frequency in payments a years.
    dates (List[dt.datetime]): List of dates for discount factor lookup.
    z_spread (float): Z-spread.
    discount_factors (pd.Series): Discount factors.
    flag(int): integral mode: 0 quadratic, 1 trapezoid, 2 Simpson, else intensity is constant
    notional (float): Notional amount.

    Returns:
        float: Dirty price of the bond.
    """

    discountdates = date_series(ref_date, expiry, coupon_freq)

    discounts = [
        get_discount_factor_by_zero_rates_linear_interp(ref_date, k, dates, discount_factors)
        for k in discountdates
    ]

    deltas = [year_frac_30_360(discountdates[i - 1], discountdates[i]) for i in range(1, len(discountdates))]
    deltas2 = [year_frac_30_360(discountdates[0],discountdates[i]) for i in range(1, len(discountdates))]
    l = len(discounts)
    discounts_cap = np.zeros(l)
    discounts_cap[0] = 1  # by definition

    for k in range(1, l):

        if flag == 0:
            discounts_cap[k] = discounts[k] * np.exp(- scipy.integrate.quad(z_spread, ref_date, discountdates[k]))

        elif flag == 1:
            discounts_cap[k] = discounts[k] * np.exp(- scipy.integrate.trapezoid(z_spread, ref_date, discountdates[k]))

        elif flag == 2:
            discounts_cap[k] = discounts[k] * np.exp(- scipy.integrate.simpson(z_spread, ref_date, discountdates[k]))

        elif flag == 3:
            discounts_cap[k] = discounts[k] * np.exp(- z_spread * deltas2[k-1])
        else: 
            return None



    
    dirty_price = coupon_rate * sum(deltas*discounts_cap[1:]) + discounts_cap[-1]
    
    return dirty_price * notional



    


def defaultable_bond_dirty_price_from_intensity_piecewise(
    ref_date: Union[dt.date, pd.Timestamp],
    expiry: Union[dt.date, pd.Timestamp],
    coupon_rate: float,
    coupon_freq: int,
    dates: List[dt.datetime],
    recovery_rate: float,
    prev_lambda: float,
    intensity: Union[float, pd.Series],
    discount_factors: pd.Series,
    notional: float = 1.0,
) -> float:
    """
    Compute dirty price of a defaultable bond with piecewise default intensities.

    Parameters:
    ref_date (Union[dt.date, pd.Timestamp]): Reference date.
    expiry (Union[dt.date, pd.Timestamp]): Bond's expiry date.
    coupon_rate (float): Coupon rate.
    coupon_freq (int): Coupon frequency in payments a years.
    dates (List[dt.datetime]): List of dates for discount factor lookup.
    recovery_rate (float): Recovery rate.
    prev_lambda (float): Previous lambda value.
    intensity (Union[float, pd.Series]): Intensity, can be the average intensity (float) or a
        piecewise constant function of time (pd.Series).
    discount_factors (pd.Series): Discount factors.
    notional (float): Notional amount.

    Returns:
        float: Dirty price of the bond.
    """
    # Coupon bonds
    discount_dates = date_series(ref_date, expiry, coupon_freq)
    
    # Discount factors
    discounts = [
        get_discount_factor_by_zero_rates_linear_interp(ref_date, k, dates, discount_factors)
        for k in discount_dates
    ]
    
    # Deltas
    deltas = [year_frac_30_360(discount_dates[i - 1], discount_dates[i]) for i in range(1, len(discount_dates))]
    
    # Exponential arguments
    exp_arg = np.zeros(2 * coupon_freq)
    for i in range(coupon_freq):
        exp_arg[i] = -deltas[i] * prev_lambda
    for i in range(coupon_freq, 2 * coupon_freq):
        exp_arg[i] = -deltas[i] * intensity
    
    # Equivalent of the sum2 of the previous function
    temp = discounts[1] * (1 - np.exp(exp_arg[0])) + sum(
        discounts[i+1] * (np.exp(np.sum(exp_arg[:i])) - np.exp(np.sum(exp_arg[:i + 1])))
        for i in range(1, len(deltas))
    )
    
    # Price computation
    price = (
        coupon_rate * sum(deltas[i] * discounts[i + 1] * np.exp(np.sum(exp_arg[:i + 1])) for i in range(len(deltas))) +
        np.exp(np.sum(exp_arg[:])) * discounts[-1] +
        recovery_rate * temp
    )
    
    return price * notional


    
