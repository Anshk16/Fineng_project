"""
Mathematical Engineering - Financial Engineering, FY 2024-2025
Risk Management - Exercise 1: Hedging a Swaption Portfolio
"""

from enum import Enum
import numpy as np
import pandas as pd
import datetime as dt
import calendar
from scipy.stats import norm
from typing import Iterable, Union, List, Tuple
from yearfrac import yearfrac, mod


# Define an enumeration for the two types of swaptions
class SwapType(Enum):
    """
    Types of swaptions.
    """
    RECEIVER = "receiver"  # Option to receive fixed rate payments
    PAYER = "payer"        # Option to pay fixed rate payments


def year_frac_act_x(t1: dt.datetime, t2: dt.datetime, x: int) -> float:
    """
    Compute the year fraction between two dates using the ACT/x convention.
    
    Parameters:
        t1 (dt.datetime): First date.
        t2 (dt.datetime): Second date.
        x (int): Number of days in a year (commonly 365).

    Returns:
        float: Year fraction between the two dates.
    """
    # Calculate the difference in days and divide by x
    return (t2 - t1).days / x


def from_discount_factors_to_zero_rates(
    dates: Union[List[float], pd.DatetimeIndex],
    discount_factors: Iterable[float],
) -> List[float]:
    """
    Compute the zero rates from the discount factors.

    Parameters:
        dates (Union[List[float], pd.DatetimeIndex]): List of year fractions or dates.
        discount_factors (Iterable[float]): List of discount factors.

    Returns:
        List[float]: List of zero rates.
    """
    effDates, effDf = dates, discount_factors
    # If dates are given as a DatetimeIndex, convert them to year fractions using ACT/365
    if isinstance(effDates, pd.DatetimeIndex):
        effDates = [
            year_frac_act_x(effDates[i - 1], effDates[i], 365)
            for i in range(1, len(effDates))
        ]
        # Adjust discount factors by excluding the first value to match the year fractions
        effDf = discount_factors[1:]

    # Calculate zero rates using the formula: -ln(discount factor) / time
    return -np.log(np.array(effDf)) / np.array(effDates)


def get_discount_factor_by_zero_rates_linear_interp(
    reference_date: Union[dt.datetime, pd.Timestamp],
    interp_date: Union[dt.datetime, pd.Timestamp],
    dates: Union[List[dt.datetime], pd.DatetimeIndex],
    discount_factors: Iterable[float],
) -> float:
    """
    Given a list of discount factors, return the discount factor at a given date by linear interpolation.
    
    Parameters:
        reference_date (Union[dt.datetime, pd.Timestamp]): Reference date.
        interp_date (Union[dt.datetime, pd.Timestamp]): Date at which to interpolate the discount factor.
        dates (Union[List[dt.datetime], pd.DatetimeIndex]): List of dates corresponding to the discount factors.
        discount_factors (Iterable[float]): List of discount factors.

    Returns:
        float: Discount factor at the interpolated date.
    """
    # Ensure dates and discount factors are the same length
    if len(dates) != len(discount_factors):
        raise ValueError("Dates and discount factors must have the same length.")

    # Convert dates to year fractions from the reference date (skip the first date)
    year_fractions = [year_frac_act_x(reference_date, T, 365) for T in dates[1:]]
    # Convert discount factors to zero rates using the helper function
    zero_rates = from_discount_factors_to_zero_rates(year_fractions, discount_factors[1:])
    # Compute the year fraction for the interpolation date
    inter_year_frac = year_frac_act_x(reference_date, interp_date, 365)
    # Linearly interpolate the zero rate for the target year fraction
    rate = np.interp(inter_year_frac, year_fractions, zero_rates)
    # Convert back to a discount factor using the exponential function
    return np.exp(-inter_year_frac * rate)


def business_date_offset(
    base_date: Union[dt.date, pd.Timestamp],
    year_offset: int = 0,
    month_offset: int = 0,
    day_offset: int = 0,
) -> Union[dt.date, pd.Timestamp]:
    """
    Return the closest following business date to a reference date after applying the specified offset.

    Parameters:
        base_date (Union[dt.date, pd.Timestamp]): The starting date.
        year_offset (int): Number of years to add.
        month_offset (int): Number of months to add.
        day_offset (int): Number of days to add.

    Returns:
        Union[dt.date, pd.Timestamp]: Adjusted date moved to the closest following business day if needed.
    """
    # Adjust the year and month by converting the offset months into years and months
    total_months = base_date.month + month_offset - 1
    year, month = divmod(total_months, 12)
    year += base_date.year + year_offset
    month += 1

    # Try to adjust the day; if the day is invalid (e.g., Feb 30), use the last valid day of the month
    day = base_date.day
    try:
        adjusted_date = base_date.replace(year=year, month=month, day=day) + dt.timedelta(days=day_offset)
    except ValueError:
        # Determine the last day of the month
        last_day_of_month = calendar.monthrange(year, month)[1]
        adjusted_date = base_date.replace(year=year, month=month, day=last_day_of_month) + dt.timedelta(days=day_offset)

    # If the adjusted date falls on a weekend, shift it to the next business day
    if adjusted_date.weekday() == 5:  # Saturday
        adjusted_date += dt.timedelta(days=2)
    elif adjusted_date.weekday() == 6:  # Sunday
        adjusted_date += dt.timedelta(days=1)

    return adjusted_date


def date_series(
    t0: Union[dt.date, pd.Timestamp], t1: Union[dt.date, pd.Timestamp], freq: int
) -> Union[List[dt.date], List[pd.Timestamp]]:
    """
    Generate a list of dates from t0 to t1 inclusive with a specified frequency (number of dates per year).

    Parameters:
        t0 (Union[dt.date, pd.Timestamp]): Start date.
        t1 (Union[dt.date, pd.Timestamp]): End date.
        freq (int): Number of dates per year.

    Returns:
        List of dates from t0 to t1.
    """
    # Start the series with the initial date
    dates = [t0]
    # Continue generating dates using business_date_offset until t1 is reached or exceeded
    while dates[-1] < t1:
        dates.append(business_date_offset(t0, month_offset=len(dates) * 12 // freq))
    # Remove any date that overshoots t1
    if dates[-1] > t1:
        dates.pop()
    # Ensure the final date is exactly t1
    if dates[-1] != t1:
        dates.append(t1)

    return dates


def swaption_price_calculator(
    S0: float,
    strike: float,
    ref_date: Union[dt.date, pd.Timestamp],
    expiry: Union[dt.date, pd.Timestamp],
    underlying_expiry: Union[dt.date, pd.Timestamp],
    sigma_black: float,
    freq: int,
    discount_factors: pd.Series,
    swaption_type: SwapType = SwapType.RECEIVER,
    compute_delta: bool = False,
) -> Union[float, Tuple[float, float]]:
    """
    Calculate the price (and optionally the delta) of a swaption using the Black model.

    Parameters:
        S0 (float): Forward swap rate.
        strike (float): Swaption strike price.
        ref_date (Union[dt.date, pd.Timestamp]): Valuation date.
        expiry (Union[dt.date, pd.Timestamp]): Swaption expiry date.
        underlying_expiry (Union[dt.date, pd.Timestamp]): Expiry date of the underlying forward starting swap.
        sigma_black (float): Implied volatility for the swaption.
        freq (int): Frequency of fixed leg payments per year.
        discount_factors (pd.Series): Series of discount factors indexed by date.
        swaption_type (SwapType): Type of swaption (receiver or payer).
        compute_delta (bool): Flag to compute delta (sensitivity), though only receiver delta is implemented.

    Returns:
        Tuple containing the swaption price and its delta.
    """
    # Generate the payment dates for the fixed leg of the underlying swap
    fixed_leg_schedule = date_series(expiry, underlying_expiry, freq)

    # Calculate the time to expiry from the reference date
    time_to_mat = year_frac_act_x(ref_date, expiry, 365)
    
    # Calculate d1 and d2 parameters for the Black formula
    d1 = 1 / (sigma_black * np.sqrt(time_to_mat)) * np.log(S0 / strike) + 0.5 * sigma_black * np.sqrt(time_to_mat)
    d2 = d1 - sigma_black * np.sqrt(time_to_mat)

    # Interpolate discount factors for each payment date in the fixed leg schedule
    discounts = [
        get_discount_factor_by_zero_rates_linear_interp(ref_date, i, discount_factors.index, discount_factors.values)
        for i in fixed_leg_schedule
    ]
    
    # Compute forward discount factors (excluding the first discount factor)
    fwd_discount = [discounts[i+1] / discounts[0] for i in range(len(discounts)-1)]

    # Calculate the year fractions for the fixed leg periods using the EU 30/360 convention
    yf = [
        yearfrac(fixed_leg_schedule[i-1], fixed_leg_schedule[i], mod.EU_30_360)
        for i in range(1, len(fixed_leg_schedule))
    ]
    # Compute the basis point value (BPV) as the weighted sum of the forward discount factors
    bpv = sum([yf[i] * fwd_discount[i] for i in range(len(yf))])
    
    # Calculate the swaption price using the Black formula for swaptions
    swaption_price = discounts[0] * bpv * (S0 * norm.cdf(d1) - strike * norm.cdf(d2))

    # For a receiver swaption, adjust the formula accordingly and compute delta
    if swaption_type == SwapType.RECEIVER:
        swaption_price = discounts[0] * bpv * (strike * norm.cdf(-d2) - S0 * norm.cdf(-d1))
        swaption_delta = discounts[0] * bpv * (norm.cdf(d1) - 1)
    
    # Note: If compute_delta were enabled for payer, additional logic would be required.
    return swaption_price, swaption_delta


def irs_proxy_duration(
    ref_date: dt.date,
    swap_rate: float,
    fixed_leg_payment_dates: List[dt.date],
    discount_factors: pd.Series,
) -> float:
    """
    Compute the duration of an interest rate swap, approximated using a fixed coupon bond.

    Parameters:
        ref_date (dt.date): Valuation date.
        swap_rate (float): Swap rate.
        fixed_leg_payment_dates (List[dt.date]): List of fixed leg payment dates.
        discount_factors (pd.Series): Series of discount factors indexed by date.

    Returns:
        float: Duration (sensitivity to interest rate changes) of the swap.
    """
    # Calculate the present value of the first coupon payment
    IB_bond = swap_rate * yearfrac(ref_date, fixed_leg_payment_dates[0], mod.EU_30_360) * \
              get_discount_factor_by_zero_rates_linear_interp(
                  discount_factors.index[0], fixed_leg_payment_dates[0], discount_factors.index, discount_factors.values
              )
    # Sum up the present values for subsequent coupon payments
    for i in range(1, len(fixed_leg_payment_dates)):
        yfrac = yearfrac(fixed_leg_payment_dates[i-1], fixed_leg_payment_dates[i], mod.EU_30_360)
        IB_bond += yfrac * swap_rate * \
                   get_discount_factor_by_zero_rates_linear_interp(
                       discount_factors.index[0], fixed_leg_payment_dates[i], discount_factors.index, discount_factors.values
                   )
    # Add the final principal repayment
    IB_bond += get_discount_factor_by_zero_rates_linear_interp(
        discount_factors.index[0], fixed_leg_payment_dates[-1], discount_factors.index, discount_factors.values
    )

    # Calculate the weighted sum of time factors (duration numerator)
    sum = swap_rate * yearfrac(ref_date, fixed_leg_payment_dates[0], mod.EU_30_360) * \
          get_discount_factor_by_zero_rates_linear_interp(
              discount_factors.index[0], fixed_leg_payment_dates[0], discount_factors.index, discount_factors.values
          ) * yearfrac(ref_date, fixed_leg_payment_dates[0], mod.EU_30_360)
    for i in range(1, len(fixed_leg_payment_dates)):
        yfrac = yearfrac(fixed_leg_payment_dates[i-1], fixed_leg_payment_dates[i], mod.EU_30_360)
        sum += yfrac * swap_rate * \
               get_discount_factor_by_zero_rates_linear_interp(
                   discount_factors.index[0], fixed_leg_payment_dates[i], discount_factors.index, discount_factors.values
               ) * yearfrac(ref_date, fixed_leg_payment_dates[i], mod.EU_30_360)
    sum += get_discount_factor_by_zero_rates_linear_interp(
        discount_factors.index[0], fixed_leg_payment_dates[-1], discount_factors.index, discount_factors.values
    ) * yearfrac(ref_date, fixed_leg_payment_dates[-1], mod.EU_30_360)

    # The duration is the weighted sum divided by the bond price (IB_bond)
    duration = sum / IB_bond

    return duration


def swap_par_rate(
    fixed_leg_schedule: List[dt.datetime],
    discount_factors: pd.Series, 
    fwd_start_date: dt.datetime | None = None,
) -> float:
    """
    Calculate the swap par rate, i.e., the fixed rate that makes the net present value of the swap zero.
    If a forward start date is provided, the function returns a forward swap rate.

    Parameters:
        fixed_leg_schedule (List[dt.datetime]): List of fixed leg payment dates.
        discount_factors (pd.Series): Series of discount factors indexed by date.
        fwd_start_date (dt.datetime | None): Optional forward start date.

    Returns:
        float: The swap par rate.
    """
    today = discount_factors.index[0]
    
    # Calculate the discount factor at the forward start date or use 1 if not provided
    discount_factor_t0 = get_discount_factor_by_zero_rates_linear_interp(
        today, fwd_start_date, discount_factors.index, discount_factors.values
    ) if fwd_start_date is not None else 1
    
    # Calculate the discount factor for the final payment date
    discount_factor_tN = get_discount_factor_by_zero_rates_linear_interp(
        discount_factors.index[0], fixed_leg_schedule[-1], discount_factors.index, discount_factors.values
    )
    
    # Calculate the basis point value (BPV) of the swap
    if fwd_start_date is not None:
        bpv = yearfrac(fwd_start_date, fixed_leg_schedule[0], mod.EU_30_360) * \
              get_discount_factor_by_zero_rates_linear_interp(
                  discount_factors.index[0], fixed_leg_schedule[0], discount_factors.index, discount_factors.values
              )
        for i in range(1, len(fixed_leg_schedule)):
            year_frac_val = yearfrac(fixed_leg_schedule[i - 1], fixed_leg_schedule[i], mod.EU_30_360)
            discount_factor_i = get_discount_factor_by_zero_rates_linear_interp(
                discount_factors.index[0], fixed_leg_schedule[i], discount_factors.index, discount_factors.values
            )
            bpv += year_frac_val * discount_factor_i
    else:
        bpv = yearfrac(today, fixed_leg_schedule[0], mod.EU_30_360) * \
              get_discount_factor_by_zero_rates_linear_interp(
                  discount_factors.index[0], fixed_leg_schedule[0], discount_factors.index, discount_factors.values
              )
        for i in range(1, len(fixed_leg_schedule)):
            year_frac_val = yearfrac(fixed_leg_schedule[i - 1], fixed_leg_schedule[i], mod.EU_30_360)
            discount_factor_i = get_discount_factor_by_zero_rates_linear_interp(
                discount_factors.index[0], fixed_leg_schedule[i], discount_factors.index, discount_factors.values
            )
            bpv += year_frac_val * discount_factor_i

    # Return the par rate computed from the difference in discount factors divided by BPV
    return (discount_factor_t0 - discount_factor_tN) / bpv


def swap_mtm(
    swap_rate: float,
    fixed_leg_schedule: List[dt.datetime],
    discount_factors: pd.Series,
    swap_type: SwapType = SwapType.PAYER,
) -> float:
    """
    Compute the mark-to-market (MTM) value of a swap based on the fixed leg cash flows and discount factors.
    
    Parameters:
        swap_rate (float): The fixed swap rate.
        fixed_leg_schedule (List[dt.datetime]): List of fixed leg payment dates.
        discount_factors (pd.Series): Series of discount factors indexed by date.
        swap_type (SwapType): Swap type (payer or receiver).

    Returns:
        float: The swap mark-to-market value.
    """
    today = discount_factors.index[0]
    # Calculate the basis point value (BPV) for the fixed leg starting with the first payment
    bpv = yearfrac(today, fixed_leg_schedule[0], mod.EU_30_360) * \
          get_discount_factor_by_zero_rates_linear_interp(
              discount_factors.index[0], fixed_leg_schedule[0], discount_factors.index, discount_factors.values
          )
    # Sum the BPV for each subsequent fixed leg payment
    for i in range(1, len(fixed_leg_schedule)):
        year_frac_val = yearfrac(fixed_leg_schedule[i - 1], fixed_leg_schedule[i], mod.EU_30_360)
        discount_factor_i = get_discount_factor_by_zero_rates_linear_interp(
            discount_factors.index[0], fixed_leg_schedule[i], discount_factors.index, discount_factors.values
        )
        bpv += year_frac_val * discount_factor_i

    # Compute the present value of the floating leg as the difference from 1 to the discount factor at the last payment date
    P_term = get_discount_factor_by_zero_rates_linear_interp(
        discount_factors.index[0], fixed_leg_schedule[-1], discount_factors.index, discount_factors.values
    )
    float_leg = 1.0 - P_term
    # Compute the value of the fixed leg
    fixed_leg = swap_rate * bpv

    # Choose a multiplier based on swap type (payer vs receiver)
    if swap_type == SwapType.RECEIVER:
        multiplier = 1
    elif swap_type == SwapType.PAYER:
        multiplier = -1
    else:
        raise ValueError("Unknown swap type.")

    # Return the mark-to-market value, applying the multiplier to adjust for the swap type
    return multiplier * (fixed_leg - float_leg)
