"""
Mathematical Engineering - Financial Engineering, FY 2024-2025
Risk Management - Exercise 1: Hedging a Swaption Portfolio
"""

from enum import Enum
import numpy as np
import pandas as pd
import datetime as dt
import calendar
from typing import Iterable, Union, List, Union, Tuple
import QuantLib as ql
from scipy.interpolate import interp1d,CubicSpline
from scipy.stats import norm

__all__ = [
    "business_date_offset",
    "swaption_price_calculator",
    "date_series",
    "irs_proxy_duration",
    "swap_par_rate",
    "SwapType",
    "bootstrap",  
]

class SwapType(Enum):
    """
    Types of swaptions.
    """

    RECEIVER = "receiver"
    PAYER = "payer"


def year_frac_act_x(t1: dt.datetime, t2: dt.datetime, x: int) -> float:
    """
    Compute the year fraction between two dates using the ACT/x convention.

    Parameters:
        t1 (dt.datetime): First date.
        t2 (dt.datetime): Second date.
        x (int): Number of days in a year.

    Returns:
        float: Year fraction between the two dates.
    """

    return (t2 - t1).days / x

def year_frac_30_360(t1: dt.datetime, t2: dt.datetime) ->float:
    d1, m1, y1 = t1.day, t1.month, t1.year
    d2, m2, y2 = t2.day, t2.month, t2.year


    if m1 != 2 and d1 == 31:
        d1=30
    if m2 != 2 and d2==31:
        d2=30

    day_count = (360 * (y2-y1) + 30 * (m2-m1) + (d2-d1))


    return day_count /360.0

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
    if isinstance(effDates, pd.DatetimeIndex):
        effDates = [
            year_frac_act_x(effDates[i - 1], effDates[i], 365)
            for i in range(1, len(effDates))
        ]
        effDf = discount_factors[1:]

    return -np.log(np.array(effDf)) / np.array(effDates)



def get_discount_factor_by_zero_rates_linear_interp(
    reference_date: Union[dt.datetime, pd.Timestamp],
    interp_date: Union[dt.datetime, pd.Timestamp],
    dates: Union[List[dt.datetime], pd.DatetimeIndex],
    discount_factors: Iterable[float],
) -> float:
    """
    Given a list of discount factors, return the discount factor at a given date by linear
    interpolation.

    Parameters:
        reference_date (Union[dt.datetime, pd.Timestamp]): Reference date.
        interp_date (Union[dt.datetime, pd.Timestamp]): Date at which the discount factor is
            interpolated.
        dates (Union[List[dt.datetime], pd.DatetimeIndex]): List of dates.
        discount_factors (Iterable[float]): List of discount factors.

    Returns:
        float: Discount factor at the interpolated date.
    """

    if len(dates) != len(discount_factors):
        raise ValueError("Dates and discount factors must have the same length.")

    year_fractions = [year_frac_act_x(reference_date, T, 365) for T in dates[1:]]
    zero_rates = from_discount_factors_to_zero_rates(
        year_fractions, discount_factors[1:]
    )
    inter_year_frac = year_frac_act_x(reference_date, interp_date, 365)
    rate = np.interp(inter_year_frac, year_fractions, zero_rates)
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
        base_date (Union[dt.date, pd.Timestamp]): Reference date.
        year_offset (int): Number of years to add.
        month_offset (int): Number of months to add.
        day_offset (int): Number of days to add.

    Returns:
        Union[dt.date, pd.Timestamp]: Closest following business date to ref_date once the specified
            offset is applied.
    """

    # Adjust the year and month
    total_months = base_date.month + month_offset - 1
    year, month = divmod(total_months, 12)
    year += base_date.year + year_offset
    month += 1

    # Adjust the day and handle invalid days
    day = base_date.day
    try:
        adjusted_date = base_date.replace(
            year=year, month=month, day=day
        ) + dt.timedelta(days=day_offset)
    except ValueError:
        # Set to the last valid day of the adjusted month
        last_day_of_month = calendar.monthrange(year, month)[1]
        adjusted_date = base_date.replace(
            year=year, month=month, day=last_day_of_month
        ) + dt.timedelta(days=day_offset)

    # Adjust to the closest business day
    if adjusted_date.weekday() == 5:  # Saturday
        adjusted_date += dt.timedelta(days=2)
    elif adjusted_date.weekday() == 6:  # Sunday
        adjusted_date += dt.timedelta(days=1)

    return adjusted_date


def date_series(
    t0: Union[dt.date, pd.Timestamp], t1: Union[dt.date, pd.Timestamp], freq: int
) -> Union[List[dt.date], List[pd.Timestamp]]:
    """
    Return a list of dates from t0 to t1 inclusive with frequency freq, where freq is specified as
    the number of dates per year.
    """

    dates = [t0]
    while dates[-1] < t1:
        dates.append(business_date_offset(t0, month_offset=len(dates) * 12 // freq))
    if dates[-1] > t1:
        dates.pop()
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
    dates: List[dt.datetime],
    discount_factors: pd.Series,
    swaption_type: SwapType = SwapType.RECEIVER,
    compute_delta: bool = False,
) -> Union[float, Tuple[float, float]]:
    """
    Return the swaption price defined by the input parameters.

    Parameters:
        S0 (float): Forward swap rate.
        strike (float): Swaption strike price.
        ref_date (Union[dt.date, pd.Timestamp]): Value date.
        expiry (Union[dt.date, pd.Timestamp]): Swaption expiry date.
        underlying_expiry (Union[dt.date, pd.Timestamp]): Underlying forward-starting swap expiry.
        sigma_black (float): Swaption implied volatility.
        freq (int): Number of times a year the fixed leg pays the coupon.
        dates (List[dt.datetime]): List of dates for discount factor lookup.
        discount_factors (pd.Series): Discount factors indexed by dates.
        swaption_type (SwapType): Swaption type, default to receiver.
        compute_delta (bool): Whether to compute delta, default to False.

    Returns:
        Union[float, Tuple[float, float]]: Swaption price (and possibly delta).
    """
    
    x = year_frac_act_x(ref_date, expiry,365)
    d1 = (1 / (sigma_black * np.sqrt(x))) * np.log(S0 / strike) + 0.5 * sigma_black * np.sqrt(x)
    d2 = d1 - sigma_black * np.sqrt(x)

    discountdates = date_series(expiry, underlying_expiry, freq)
    
    discounts = [
        get_discount_factor_by_zero_rates_linear_interp(ref_date, k, dates, discount_factors)
        for k in discountdates
    ]
    
    bpv = sum(
        year_frac_30_360(discountdates[k - 1], discountdates[k]) * discounts[k]
        for k in range(1, len(discountdates))
    )
    
    if swaption_type == SwapType.RECEIVER:
        price =  bpv * (strike * norm.cdf(-d2) - S0 * norm.cdf(-d1))
    else:  # Payer swaption
        price =  bpv * (S0 * norm.cdf(d1) - strike * norm.cdf(d2))

    if compute_delta:
        if swaption_type == SwapType.RECEIVER:
            delta = - bpv * norm.cdf(-d1)
        else:  # Payer swaption
            delta =  bpv * norm.cdf(d1)
    return price, delta



def irs_proxy_duration(
    ref_date: dt.date,
    swap_rate: float,
    fixed_leg_payment_dates: List[dt.date],
    discount_factors: pd.Series,
    dates : List[dt.date],
) -> float:
    """
    Approximate the rate sensitivity (duration) of an interest rate swap using the 
    duration formula of a fixed coupon bond.

    Parameters:
        ref_date (dt.date): Reference date.
        swap_rate (float): Swap rate.
        fixed_leg_payment_dates (List[dt.date]): Fixed leg payment dates.
        discount_factors (pd.Series): Discount factors indexed by date.

    Returns:
        float: Swap duration.
    """
    
    B=[]
    Delta=[]
    Delta1=[]
    
    Sum=0
    for i in range(1,10):
        B.append(get_discount_factor_by_zero_rates_linear_interp(ref_date,fixed_leg_payment_dates[i],dates,discount_factors))
        Delta.append(year_frac_30_360(fixed_leg_payment_dates[i-1],fixed_leg_payment_dates[i]))
        Delta1.append(year_frac_30_360(fixed_leg_payment_dates[0],fixed_leg_payment_dates[i]))
        
        Sum = Sum + B[i-1]*Delta[i-1]*Delta1[i-1]*swap_rate 
    
    B.append(get_discount_factor_by_zero_rates_linear_interp(ref_date,fixed_leg_payment_dates[9],dates,discount_factors))
    Delta1.append(year_frac_30_360(fixed_leg_payment_dates[0],fixed_leg_payment_dates[9]))
    
    return Sum + Delta1[9]*B[9]


def swap_par_rate(
    refdate : dt.datetime,
    fixed_leg_schedule: List[dt.datetime],
    discount_factors: pd.Series,
    fwd_start_date: dt.datetime | None = None,
    discountdates : List[dt.datetime] | None = None,
) -> float:
    """
    Given a fixed leg payment schedule and the discount factors, return the swap par rate. If a
    forward start date is provided, a forward swap rate is returned.

    Parameters:
        fixed_leg_schedule (List[dt.datetime]): Fixed leg payment dates.
        discount_factors (pd.Series): Discount factors.
        fwd_start_date (dt.datetime | None): Forward start date, default to None.

    Returns:
        float: Swap par rate.
    """
    ### !!! MODIFY AS APPROPRIATE !!!
    discount_factor_t0 = get_discount_factor_by_zero_rates_linear_interp(refdate,fwd_start_date,discountdates,discount_factors) if fwd_start_date is not None else 1
    discount_factor_tN = get_discount_factor_by_zero_rates_linear_interp(refdate,fixed_leg_schedule[-1],discountdates,discount_factors)
    discounts = []
    for k in fixed_leg_schedule : 
        x=get_discount_factor_by_zero_rates_linear_interp(refdate,k,discountdates,discount_factors)
        discounts.append(x)
    bpv = 0
    for k in range(1,len(discounts)):
        bpv+=year_frac_30_360(fixed_leg_schedule[k-1],fixed_leg_schedule[k])*discounts[k]
    return (discount_factor_t0 - discount_factor_tN) / bpv  # error 5


def swap_mtm(
    swap_rate: float,
    refdate : dt.datetime,
    fixed_leg_schedule: List[dt.datetime],
    discount_factors: pd.Series,
    dates : List[dt.datetime],
    swap_type: SwapType = SwapType.PAYER,
) -> float:
    """
    Given a swap rate, a fixed leg payment schedule and the discount factors, return the swap
    mark-to-market.

    Parameters:
        swap_rate (float): Swap rate.
        fixed_leg_schedule (List[dt.datetime]): Fixed leg payment dates.
        discount_factors (pd.Series): Discount factors.
        swap_type (SwapType): Swap type, either 'payer' or 'receiver', default to 'payer'.

    Returns:
        float: Swap mark-to-market.
    """

    # Single curve framework, returns price and basis point value
    discounts = np.array([
    get_discount_factor_by_zero_rates_linear_interp(refdate, k, dates, discount_factors)
    for k in fixed_leg_schedule
])
    bpv = sum(
        year_frac_30_360(fixed_leg_schedule[k - 1], fixed_leg_schedule[k]) * discounts[k]
        for k in range(1, len(fixed_leg_schedule))
    )  # !!! MODIFY AS APPROPRIATE !!!
    P_term = get_discount_factor_by_zero_rates_linear_interp(refdate,fixed_leg_schedule[-1],dates,discount_factors)
    float_leg = 1.0 - P_term
    fixed_leg = swap_rate * bpv

    if swap_type == SwapType.RECEIVER:
        multiplier = -1
    elif swap_type == SwapType.PAYER:
        multiplier = 1
    else:
        raise ValueError("Unknown swap type.")

    return float(multiplier) * (float_leg - fixed_leg)










# bootstrapping section
def bootstrap(file:str,shift)  :
    format_data = "%Y-%m-%dT%H:%M:%S.%f"
    dates, rates = read_excel_data(file, format_data)
    SettlementDate = dates["settlement"]
    DeposDates = dates["depos"]
    FuturesDates = dates["futures"]
    SwapsDates = dates["swaps"]
    DeposRates = np.array(rates["depos"])+shift
    FuturesRates = np.array(rates["futures"])+shift
    SwapsRates = np.array(rates["swaps"])+shift
    n_years = int((SwapsDates[-1]-SettlementDate).days/365.25)
    qlSettlementDate = ql.Date(SettlementDate.day, SettlementDate.month, SettlementDate.year)
    qlDeposDates = [ql.Date(date.day, date.month, date.year) for date in DeposDates]
    qlFuturesDates = [[ql.Date(date[0].day, date[0].month, date[0].year),
                    ql.Date(date[1].day, date[1].month, date[1].year)] for date in FuturesDates]
    qlSwapsDates = [ql.Date(date.day, date.month, date.year) for date in SwapsDates]
    FirstDay = ql.Date(1,2,2023)
    FestCalendar = ql.Italy()
    calendar = NextBusinessDays(FirstDay,FestCalendar,n_years)
    Dates = [qlSettlementDate]
    Discounts = [1]  # First DCF is 1
    ContDepos = 0
    while ContDepos + 1 < len(qlDeposDates) and qlDeposDates[ContDepos] <= qlFuturesDates[0][0]:
        ContDepos = ContDepos +1
    Dates += qlDeposDates[:ContDepos]
    MidDepos = np.zeros((len(DeposRates),1))
    MidDepos = np.mean(DeposRates,axis=1)
    for j in range(ContDepos):
        temp = ql.Actual360().yearFraction((qlSettlementDate),qlDeposDates[j])
        DiscountTemp = 1 / (1 + temp * MidDepos[j])
        Discounts.append(DiscountTemp)
    Discounts = [float(x) for x in Discounts]
    ContFutures = 0
    while ContFutures + 1 < len(qlFuturesDates) and qlFuturesDates[ContFutures][1] <= qlSwapsDates[0]:
        ContFutures = ContFutures +1
    SelectedFutures = FuturesRates[:ContFutures, :]
    MidFutures =  np.mean(SelectedFutures, axis=1)
    ForwardDiscounts = []
    for j in range(ContFutures):
        StartDate = qlFuturesDates[j][0]
        EndDate = qlFuturesDates[j][1]
        temp = ql.Actual360().yearFraction(StartDate, EndDate) # Calculate the year fraction between the two dates
        DiscountTemp = 1 / (1 + temp * MidFutures[j])
        ForwardDiscounts.append(DiscountTemp)
    ForwardDiscounts = [float(x) for x in ForwardDiscounts]
    def zRatesInterp(dates, discounts, interp_dates):
        m = len(interp_dates)
        DCF_t0_ti = np.zeros(m)
        dates_num = np.array([date.serialNumber() for date in dates])
        interp_dates_num = np.array([date.serialNumber() for date in interp_dates])
        for i in range(m):
            if interp_dates[i] in dates:
                DCF_t0_ti[i] = discounts[dates.index(interp_dates[i])]
            elif interp_dates_num[i] < max(dates_num) and interp_dates_num[i] > min(dates_num):
                y = -np.log(discounts[1:]) / np.array([ql.Actual365Fixed().yearFraction(dates[0], date) for date in dates[1:]])
                interp_y = interp1d(dates_num[1:], y)(interp_dates_num[i])
                DCF_t0_ti[i] = np.exp(-interp_y * ql.Actual365Fixed().yearFraction(dates[0], interp_dates[i]))
            else:
                y = -np.log(discounts[1:]) / np.array([ql.Actual365Fixed().yearFraction(dates[0], date) for date in dates[1:]])
                interp_y = interp1d(dates_num[1:], y, kind='previous', fill_value="extrapolate")(interp_dates_num[i])
                DCF_t0_ti[i] = np.exp(-interp_y * ql.Actual365Fixed().yearFraction(dates[0], interp_dates[i]))
        return DCF_t0_ti
    if qlFuturesDates[0][0] == qlDeposDates[ContDepos]:
        index = dates.index(qlFuturesDates[0][0])
        DCF_t0_ti = Discounts[index]
    else:
        InterpDate = qlFuturesDates[0][0]
        NewDeposDates = qlDeposDates[ContDepos]
        Dates.append(NewDeposDates)
        MidDeposPrime = np.mean(DeposRates[ContDepos])
        temp = ql.Actual360().yearFraction(Dates[0], NewDeposDates)
        DiscountToAdd = 1 / (1 + temp * MidDeposPrime)
        Discounts.append(DiscountToAdd)
        Discounts = [float(x) for x in Discounts]
        DCF_t0_ti = zRatesInterp(Dates, Discounts, [InterpDate])
    Dates.append(qlFuturesDates[0][1])
    Discounts.append(DCF_t0_ti[0] * ForwardDiscounts[0])
    Discounts = [float(x) for x in Discounts]
    LenDiscounts = len(Discounts)
    LenDates = LenDiscounts
    Dates.extend([None] * (ContFutures - 1))
    Discounts.extend([0.0] * (ContFutures - 1))
    Discounts = [float(x) for x in Discounts]
    for i in range(1, ContFutures):
        t_i = qlFuturesDates[i][0]  # ti
        t_i_plus = qlFuturesDates[i][1]  # ti+1
        DCF_t0_ti = zRatesInterp(Dates[:LenDates + i - 1], Discounts[:LenDiscounts + i - 1], [t_i])
        DCF_t0_ti_plus = DCF_t0_ti[0] * ForwardDiscounts[i]
        Dates[LenDates + i - 1] = t_i_plus
        Discounts[LenDiscounts + i - 1] = DCF_t0_ti_plus
        Discounts = [float(x) for x in Discounts]
    nYearsSwaps = int((SwapsDates[-1] - SettlementDate).days / 365.25)
    CalendarSwaps = [None] * (nYearsSwaps + 1)
    CalendarSwaps[0] = qlSettlementDate
    CalendarSwaps = NextBusinessDays(qlSettlementDate-1,FestCalendar = ql.Italy(), n_years=nYearsSwaps+1)
    ContSwaps = 0
    while CalendarSwaps[ContSwaps] < qlSwapsDates[0]:
        ContSwaps += 1
    BPV = 0.0
    for j in range(ContSwaps - 1):
        Bj = zRatesInterp(Dates, Discounts, [CalendarSwaps[j + 1]])[0]
        BPV += Bj * ql.Thirty360(ql.Thirty360.BondBasis).yearFraction(CalendarSwaps[j], CalendarSwaps[j + 1])
    AvgSwaps = 0.5 * (SwapsRates[:, 0] + SwapsRates[:, 1])
    Avg_y = -np.log(AvgSwaps) / np.array([ql.Actual365Fixed().yearFraction(qlSettlementDate, date) for date in qlSwapsDates])
    InterpolatedSwaps = CubicSpline([date.serialNumber() for date in qlSwapsDates],AvgSwaps)([date.serialNumber() for date in CalendarSwaps[ContSwaps:]])
    LenDates = len(Dates)
    LenDiscounts = LenDates
    Dates.extend([None] * (len(CalendarSwaps) - ContSwaps))
    Discounts.extend([0.0] * (len(CalendarSwaps) - ContSwaps))
    Discounts = [float(x) for x in Discounts]
    for i in range(ContSwaps, len(CalendarSwaps)):
        Delta = ql.Thirty360(ql.Thirty360.BondBasis).yearFraction(CalendarSwaps[i-1], CalendarSwaps[i])
        Dates[LenDates + i - ContSwaps] = CalendarSwaps[i]
        Bi = (1 - InterpolatedSwaps[i - ContSwaps] * BPV) / (1 + InterpolatedSwaps[i - ContSwaps] * Delta)
        Discounts[LenDiscounts + i - ContSwaps] = Bi
        BPV += Bi * Delta
    Discounts = [float(x) for x in Discounts]
    discounts = np.array(Discounts)  # Assuming Discounts is a list
    dates = np.array(Dates)  # Assuming Dates is a list of QuantLib Date objects
    yearfractions_zrates = np.array([ql.Actual365Fixed().yearFraction(dates[0], d) for d in dates])
    zRates = np.zeros(len(discounts))
    zRates[1:] = -1*np.divide(np.log(discounts[1:]) , yearfractions_zrates[1:])
    zRates *= 100  # Convert to percentage
    zRates[0] = 0
    def qld(ql_dates):
        return [dt.datetime(d.year(), d.month(), d.dayOfMonth()) for d in dates]
    return qld(dates) , zRates , discounts



def read_excel_data(filename : str, format_data:str) -> dict:
    settlement = pd.read_excel(filename, sheet_name=0, usecols="E", skiprows=6, nrows=1).values[0][0]
    settlement_date = pd.Timestamp(settlement).to_pydatetime()  # Convert to datetime
    depos_dates = pd.read_excel(filename, sheet_name=0, usecols="D", skiprows=9, nrows=6).values.flatten()
    depos_dates = [pd.Timestamp(date).to_pydatetime() for date in depos_dates]  # Convert to datetime
    futures_dates = pd.read_excel(filename, sheet_name=0, usecols="Q:R", skiprows=10, nrows=9).values
    futures_dates = np.array([[pd.Timestamp(date).to_pydatetime() for date in row] for row in futures_dates])  # Convert to datetime
    swap_dates = pd.read_excel(filename, sheet_name=0, usecols="D", skiprows=37, nrows=17).values.flatten()
    swap_dates = [pd.Timestamp(date).to_pydatetime() for date in swap_dates]  # Convert to datetime
    dates = {'settlement': settlement_date, 'depos': depos_dates,'futures': futures_dates,'swaps': swap_dates}
    depos_rates = pd.read_excel(filename, sheet_name=0, usecols="E:F", skiprows=9, nrows=6).values / 100
    futures_rates = pd.read_excel(filename, sheet_name=0, usecols="E:F", skiprows=26, nrows=9).values
    futures_rates = (100 - futures_rates) / 100
    swap_rates = pd.read_excel(filename, sheet_name=0, usecols="E:F", skiprows=37, nrows=17).values / 100
    rates = {
        'depos': depos_rates,
        'futures': futures_rates,
        'swaps': swap_rates
    }
    return dates, rates

def NextBusinessDays(FirstDay, FestCalendar, n_years):
    """
    Creates a vector of the next business day for the settlement date in each of the next `n_years`.
    Parameters:
        FirstDay: The starting date (e.g., a specific date like October 1).
        FestCalendar: The calendar to use for business day rules (e.g., ql.Italy()).
        n_years: The number of years to consider.
    """
    calendar = []
    for i in range(n_years):
        CurrentDate = FirstDay + ql.Period(i, ql.Years)
        NextBusyDay = FestCalendar.advance(CurrentDate, ql.Period(1, ql.Days), ql.Following, False)
        calendar.append(NextBusyDay)
    return calendar


def bootstrapBucket(file:str,shift)  :
    format_data = "%Y-%m-%dT%H:%M:%S.%f"
    dates, rates = read_excel_data(file, format_data)
    SettlementDate = dates["settlement"]
    DeposDates = dates["depos"]
    FuturesDates = dates["futures"]
    SwapsDates = dates["swaps"]
    DeposRates = np.array(rates["depos"])
    FuturesRates = np.array(rates["futures"])
    SwapsRates = np.array(rates["swaps"])+shift
    n_years = int((SwapsDates[-1]-SettlementDate).days/365.25)
    qlSettlementDate = ql.Date(SettlementDate.day, SettlementDate.month, SettlementDate.year)
    qlDeposDates = [ql.Date(date.day, date.month, date.year) for date in DeposDates]
    qlFuturesDates = [[ql.Date(date[0].day, date[0].month, date[0].year),
                    ql.Date(date[1].day, date[1].month, date[1].year)] for date in FuturesDates]
    qlSwapsDates = [ql.Date(date.day, date.month, date.year) for date in SwapsDates]
    FirstDay = ql.Date(1,2,2023)
    FestCalendar = ql.Italy()
    calendar = NextBusinessDays(FirstDay,FestCalendar,n_years)
    Dates = [qlSettlementDate]
    Discounts = [1]  # First DCF is 1
    ContDepos = 0
    while ContDepos + 1 < len(qlDeposDates) and qlDeposDates[ContDepos] <= qlFuturesDates[0][0]:
        ContDepos = ContDepos +1
    Dates += qlDeposDates[:ContDepos]
    MidDepos = np.zeros((len(DeposRates),1))
    MidDepos = np.mean(DeposRates,axis=1)
    for j in range(ContDepos):
        temp = ql.Actual360().yearFraction((qlSettlementDate),qlDeposDates[j])
        DiscountTemp = 1 / (1 + temp * MidDepos[j])
        Discounts.append(DiscountTemp)
    Discounts = [float(x) for x in Discounts]
    ContFutures = 0
    while ContFutures + 1 < len(qlFuturesDates) and qlFuturesDates[ContFutures][1] <= qlSwapsDates[0]:
        ContFutures = ContFutures +1
    SelectedFutures = FuturesRates[:ContFutures, :]
    MidFutures =  np.mean(SelectedFutures, axis=1)
    ForwardDiscounts = []
    for j in range(ContFutures):
        StartDate = qlFuturesDates[j][0]
        EndDate = qlFuturesDates[j][1]
        temp = ql.Actual360().yearFraction(StartDate, EndDate) # Calculate the year fraction between the two dates
        DiscountTemp = 1 / (1 + temp * MidFutures[j])
        ForwardDiscounts.append(DiscountTemp)
    ForwardDiscounts = [float(x) for x in ForwardDiscounts]
    def zRatesInterp(dates, discounts, interp_dates):
        m = len(interp_dates)
        DCF_t0_ti = np.zeros(m)
        dates_num = np.array([date.serialNumber() for date in dates])
        interp_dates_num = np.array([date.serialNumber() for date in interp_dates])
        for i in range(m):
            if interp_dates[i] in dates:
                DCF_t0_ti[i] = discounts[dates.index(interp_dates[i])]
            elif interp_dates_num[i] < max(dates_num) and interp_dates_num[i] > min(dates_num):
                y = -np.log(discounts[1:]) / np.array([ql.Actual365Fixed().yearFraction(dates[0], date) for date in dates[1:]])
                interp_y = interp1d(dates_num[1:], y)(interp_dates_num[i])
                DCF_t0_ti[i] = np.exp(-interp_y * ql.Actual365Fixed().yearFraction(dates[0], interp_dates[i]))
            else:
                y = -np.log(discounts[1:]) / np.array([ql.Actual365Fixed().yearFraction(dates[0], date) for date in dates[1:]])
                interp_y = interp1d(dates_num[1:], y, kind='previous', fill_value="extrapolate")(interp_dates_num[i])
                DCF_t0_ti[i] = np.exp(-interp_y * ql.Actual365Fixed().yearFraction(dates[0], interp_dates[i]))
        return DCF_t0_ti
    if qlFuturesDates[0][0] == qlDeposDates[ContDepos]:
        index = dates.index(qlFuturesDates[0][0])
        DCF_t0_ti = Discounts[index]
    else:
        InterpDate = qlFuturesDates[0][0]
        NewDeposDates = qlDeposDates[ContDepos]
        Dates.append(NewDeposDates)
        MidDeposPrime = np.mean(DeposRates[ContDepos])
        temp = ql.Actual360().yearFraction(Dates[0], NewDeposDates)
        DiscountToAdd = 1 / (1 + temp * MidDeposPrime)
        Discounts.append(DiscountToAdd)
        Discounts = [float(x) for x in Discounts]
        DCF_t0_ti = zRatesInterp(Dates, Discounts, [InterpDate])
    Dates.append(qlFuturesDates[0][1])
    Discounts.append(DCF_t0_ti[0] * ForwardDiscounts[0])
    Discounts = [float(x) for x in Discounts]
    LenDiscounts = len(Discounts)
    LenDates = LenDiscounts
    Dates.extend([None] * (ContFutures - 1))
    Discounts.extend([0.0] * (ContFutures - 1))
    Discounts = [float(x) for x in Discounts]
    for i in range(1, ContFutures):
        t_i = qlFuturesDates[i][0]  # ti
        t_i_plus = qlFuturesDates[i][1]  # ti+1
        DCF_t0_ti = zRatesInterp(Dates[:LenDates + i - 1], Discounts[:LenDiscounts + i - 1], [t_i])
        DCF_t0_ti_plus = DCF_t0_ti[0] * ForwardDiscounts[i]
        Dates[LenDates + i - 1] = t_i_plus
        Discounts[LenDiscounts + i - 1] = DCF_t0_ti_plus
        Discounts = [float(x) for x in Discounts]
    nYearsSwaps = int((SwapsDates[-1] - SettlementDate).days / 365.25)
    CalendarSwaps = [None] * (nYearsSwaps + 1)
    CalendarSwaps[0] = qlSettlementDate
    CalendarSwaps = NextBusinessDays(qlSettlementDate-1,FestCalendar = ql.Italy(), n_years=nYearsSwaps+1)
    ContSwaps = 0
    while CalendarSwaps[ContSwaps] < qlSwapsDates[0]:
        ContSwaps += 1
    BPV = 0.0
    for j in range(ContSwaps - 1):
        Bj = zRatesInterp(Dates, Discounts, [CalendarSwaps[j + 1]])[0]
        BPV += Bj * ql.Thirty360(ql.Thirty360.BondBasis).yearFraction(CalendarSwaps[j], CalendarSwaps[j + 1])
    AvgSwaps = 0.5 * (SwapsRates[:, 0] + SwapsRates[:, 1])
    Avg_y = -np.log(AvgSwaps) / np.array([ql.Actual365Fixed().yearFraction(qlSettlementDate, date) for date in qlSwapsDates])
    InterpolatedSwaps = CubicSpline([date.serialNumber() for date in qlSwapsDates],AvgSwaps)([date.serialNumber() for date in CalendarSwaps[ContSwaps:]])
    LenDates = len(Dates)
    LenDiscounts = LenDates
    Dates.extend([None] * (len(CalendarSwaps) - ContSwaps))
    Discounts.extend([0.0] * (len(CalendarSwaps) - ContSwaps))
    Discounts = [float(x) for x in Discounts]
    for i in range(ContSwaps, len(CalendarSwaps)):
        Delta = ql.Thirty360(ql.Thirty360.BondBasis).yearFraction(CalendarSwaps[i-1], CalendarSwaps[i])
        Dates[LenDates + i - ContSwaps] = CalendarSwaps[i]
        Bi = (1 - InterpolatedSwaps[i - ContSwaps] * BPV) / (1 + InterpolatedSwaps[i - ContSwaps] * Delta)
        Discounts[LenDiscounts + i - ContSwaps] = Bi
        BPV += Bi * Delta
    Discounts = [float(x) for x in Discounts]
    discounts = np.array(Discounts)  # Assuming Discounts is a list
    dates = np.array(Dates)  # Assuming Dates is a list of QuantLib Date objects
    yearfractions_zrates = np.array([ql.Actual365Fixed().yearFraction(dates[0], d) for d in dates])
    zRates = np.zeros(len(discounts))
    zRates[1:] = -1*np.divide(np.log(discounts[1:]) , yearfractions_zrates[1:])
    zRates *= 100  # Convert to percentage
    zRates[0] = 0
    def qld(ql_dates):
        return [dt.datetime(d.year(), d.month(), d.dayOfMonth()) for d in dates]
    return qld(dates) , zRates , discounts
