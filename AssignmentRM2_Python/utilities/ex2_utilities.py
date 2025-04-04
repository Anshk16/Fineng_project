"""
Mathematical Engineering - Financial Engineering, FY 2024-2025
Risk Management - Exercise 2: Corporate Bond Portfolio
"""

from typing import Union
import numpy as np
import pandas as pd
import datetime as dt
import math
from yearfrac import yearfrac, mod
from ex1_utilities import (
    year_frac_act_x,
    get_discount_factor_by_zero_rates_linear_interp,
    date_series
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
    cash_flows_dates = date_series(ref_date, expiry, coupon_freq)
    
    yfrac = [
        yearfrac(d1, d2, mod.EU_30_360) 
        for d1, d2 in zip(cash_flows_dates[:-1], cash_flows_dates[1:])
    ]

    # Coupon payments
    cash_flows = pd.Series(
        data= notional * coupon_rate * np.array(yfrac),
        index=cash_flows_dates[1:],
    )

    # Notional payment
    cash_flows[expiry] += notional

    return cash_flows


def defaultable_bond_dirty_price_from_intensity(
    ref_date: Union[dt.date, pd.Timestamp],
    expiry: Union[dt.date, pd.Timestamp],
    coupon_rate: float,
    coupon_freq: int,
    recovery_rate: float,
    intensity: Union[float, pd.Series],
    discount_factors: pd.Series,
    notional: float = 1.0,
) -> float:
    """
    Calculate the dirty price of a defaultable bond neglecting the recovery of the coupon payments.

    Parameters:
    ref_date (Union[dt.date, pd.Timestamp]): Reference date.
    expiry (Union[dt.date, pd.Timestamp]): Bond's expiry date.
    coupon_rate (float): Coupon rate.
    coupon_freq (int): Coupon frequency in payments a years.
    recovery_rate (float): Recovery rate.
    intensity (Union[float, pd.Series]): Intensity, can be the average intensity (float) or a
        piecewise constant function of time (pd.Series).
    discount_factors (pd.Series): Discount factors.
    notional (float): Notional amount.

    Returns:
        float: Dirty price of the bond.
    """

    cash_flow = bond_cash_flows(ref_date, expiry, coupon_rate, coupon_freq, notional)

    yfrac = [
        year_frac_act_x(ref_date, d, 365)
        for d in cash_flow.index
    ]
    
    discounts = [
        get_discount_factor_by_zero_rates_linear_interp(
            ref_date,
            d,
            discount_factors.index,
            discount_factors.values
        )
        for d in cash_flow.index
    ]

    prob = [1.0] + [math.exp(-intensity * yf) for yf in yfrac]

    price = sum(
        cash_flow.values[i] * discounts[i] * prob[i+1] + recovery_rate * discounts[i] * (prob[i] - prob[i+1]) * notional
        for i in range(len(discounts))
    )

    return price


def defaultable_bond_dirty_price_from_z_spread(
    ref_date: Union[dt.date, pd.Timestamp],
    expiry: Union[dt.date, pd.Timestamp],
    coupon_rate: float,
    coupon_freq: int,
    z_spread: float,
    discount_factors: pd.Series,
    notional: float = 1.0,
) -> float:
    """
    Calculate the dirty price of a defaultable bond from the Z-spread.

    Parameters:
    ref_date (Union[dt.date, pd.Timestamp]): Reference date.
    expiry (Union[dt.date, pd.Timestamp]): Bond's expiry date.
    coupon_rate (float): Coupon rate.
    coupon_freq (int): Coupon frequency in payments a years.
    z_spread (float): Z-spread.
    discount_factors (pd.Series): Discount factors.
    notional (float): Notional amount.

    Returns:
        float: Dirty price of the bond.
    """

    cash_flow = bond_cash_flows(ref_date, expiry, coupon_rate, coupon_freq, notional)

    yfrac = [
        year_frac_act_x(ref_date, d, 365)
        for d in cash_flow.index
    ]
    
    discounts = [
        get_discount_factor_by_zero_rates_linear_interp(
            ref_date,
            d,
            discount_factors.index,
            discount_factors.values
        )
        for d in cash_flow.index
    ]

    prob = [1.0] + [math.exp(-z_spread * yf) for yf in yfrac]

    price = sum(
        cash_flow.values[i] * discounts[i] * prob[i+1]
        for i in range(len(discounts))
    )

    return price



def defaultable_bond_dirty_price_from_intensity_and_previous_lambda(
    ref_date: Union[dt.date, pd.Timestamp],
    expiry: Union[dt.date, pd.Timestamp],
    coupon_rate: float,
    coupon_freq: int,
    recovery_rate: float,
    intensity: Union[float, pd.Series],
    discount_factors: pd.Series,
    prev_intensity: float,
    prev_expiry: Union[dt.date, pd.Timestamp],
    notional: float = 1.0,
) -> float:
    """
    Calculate the dirty price of a defaultable bond neglecting the recovery of the coupon payments.

    Parameters:
    ref_date (Union[dt.date, pd.Timestamp]): Reference date.
    expiry (Union[dt.date, pd.Timestamp]): Bond's expiry date.
    coupon_rate (float): Coupon rate.
    coupon_freq (int): Coupon frequency in payments a years.
    recovery_rate (float): Recovery rate.
    intensity (Union[float, pd.Series]): Intensity, can be the average intensity (float) or a
        piecewise constant function of time (pd.Series).
    discount_factors (pd.Series): Discount factors.
    notional (float): Notional amount.

    Returns:
        float: Dirty price of the bond.
    """

    # 1) Calcoliamo i flussi (cedole + rimborso)
    cash_flow = bond_cash_flows(ref_date, expiry, coupon_rate, coupon_freq, notional)

    # 2) Creiamo la lista dei year fraction dei flussi (rispetto a ref_date)
    yfrac = [year_frac_act_x(ref_date, d, 365) for d in cash_flow.index]

    # 3) Calcoliamo il year fraction da ref_date a prev_expiry
    yfrac_prev_expiry = year_frac_act_x(ref_date, prev_expiry, 365)

    # 4) Calcoliamo i discount factor via interpolazione
    discounts = [
        get_discount_factor_by_zero_rates_linear_interp(
            ref_date, d, discount_factors.index, discount_factors.values
        )
        for d in cash_flow.index
    ]

    # 5) Calcoliamo le probabilità di sopravvivenza a ciascuna data
    prob = [1.0]  # la sopravvivenza iniziale è 1
    for yf_current, flow_date in zip(yfrac, cash_flow.index):
        if yf_current <= yfrac_prev_expiry:
            # Tutto il periodo fino alla data del flusso è coperto da prev_intensity
            p_surv = math.exp(-prev_intensity * yf_current)
        else:
            # Spezzetto: fino a prev_expiry si usa prev_intensity,
            # poi, da prev_expiry a flow_date, si usa intensity.
            yf_additional = year_frac_act_x(prev_expiry, flow_date, 365)
            p_surv = math.exp( - (prev_intensity * yfrac_prev_expiry + intensity * yf_additional))
        prob.append(p_surv)

    price = sum(
        cash_flow.values[i] * discounts[i] * prob[i+1] + recovery_rate * discounts[i] * (prob[i] - prob[i+1]) * notional
        for i in range(len(discounts))
    )

    return price