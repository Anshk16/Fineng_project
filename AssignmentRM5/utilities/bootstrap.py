from datetime import datetime
import numpy as np
import pandas as pd
import QuantLib as ql
from scipy.interpolate import interp1d,CubicSpline
import matplotlib.pyplot as plt

def read_excel_data(filename, format_data):
    # Read settlement date
    settlement = pd.read_excel(filename, sheet_name=0, usecols="E", skiprows=6, nrows=1).values[0][0]
    settlement_date = pd.Timestamp(settlement).to_pydatetime()  # Convert to datetime

    # Read deposit dates
    depos_dates = pd.read_excel(filename, sheet_name=0, usecols="D", skiprows=9, nrows=6).values.flatten()
    depos_dates = [pd.Timestamp(date).to_pydatetime() for date in depos_dates]  # Convert to datetime

    # Read futures dates
    futures_dates = pd.read_excel(filename, sheet_name=0, usecols="Q:R", skiprows=10, nrows=9).values
    futures_dates = np.array([[pd.Timestamp(date).to_pydatetime() for date in row] for row in futures_dates])  # Convert to datetime

    # Read swap dates
    swap_dates = pd.read_excel(filename, sheet_name=0, usecols="D", skiprows=37, nrows=17).values.flatten()
    swap_dates = [pd.Timestamp(date).to_pydatetime() for date in swap_dates]  # Convert to datetime

    # Store dates in a dictionary
    dates = {
        'settlement': settlement_date,
        'depos': depos_dates,
        'futures': futures_dates,
        'swaps': swap_dates
    }

    # Read deposit rates
    depos_rates = pd.read_excel(filename, sheet_name=0, usecols="E:F", skiprows=9, nrows=6).values / 100

    # Read futures rates and adjust them
    futures_rates = pd.read_excel(filename, sheet_name=0, usecols="E:F", skiprows=26, nrows=9).values
    futures_rates = (100 - futures_rates) / 100

    # Read swap rates
    swap_rates = pd.read_excel(filename, sheet_name=0, usecols="E:F", skiprows=37, nrows=17).values / 100

    # Store rates in a dictionary
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

    # Initialize an empty list to store the business days
    calendar = []

    # Iterate over the next `n_years`
    for i in range(n_years):
        # Calculate the date for the current year
        CurrentDate = FirstDay + ql.Period(i, ql.Years)

        # Find the next business day for the current date
        # ql.Following ensures the result is adjusted to the next business day
        NextBusyDay = FestCalendar.advance(CurrentDate, ql.Period(1, ql.Days), ql.Following, False)

        # Append the next business day to the list
        calendar.append(NextBusyDay)

    return calendar

def zRatesInterp(dates, discounts, interp_dates):
    """
    Interpolate discount factors for given dates.

    Parameters:
    - dates: List of Dates for which discounts are known.
    - discounts: List of discount factors corresponding to the dates.
    - interp_dates: List of Dates for which discount factors need to be interpolated.

    Returns:
    - DCF_t0_ti: Array of interpolated discount factors.
    """
    m = len(interp_dates)
    DCF_t0_ti = np.zeros(m)


    # Convert dates to serial numbers for interpolation
    dates_num = np.array([date.serialNumber() for date in dates])
    interp_dates_num = np.array([date.serialNumber() for date in interp_dates])

    for i in range(m):
        if interp_dates[i] in dates:
            # If the interpolation date is already in the dates list
            DCF_t0_ti[i] = discounts[dates.index(interp_dates[i])]

        elif interp_dates_num[i] < max(dates_num) and interp_dates_num[i] > min(dates_num):
            # If the interpolation date is between two known dates
            y = -np.log(discounts[1:]) / np.array([ql.Actual365Fixed().yearFraction(dates[0], date) for date in dates[1:]])

            # Linear interpolation
            interp_y = interp1d(dates_num[1:], y)(interp_dates_num[i])

            DCF_t0_ti[i] = np.exp(-interp_y * ql.Actual365Fixed().yearFraction(dates[0], interp_dates[i]))

        else:
            # If the interpolation date is outside the range of known dates, extrapolate
            y = -np.log(discounts[1:]) / np.array([ql.Actual365Fixed().yearFraction(dates[0], date) for date in dates[1:]])

            # Extrapolate using the last known value
            interp_y = interp1d(dates_num[1:], y, kind='previous', fill_value="extrapolate")(interp_dates_num[i])
            DCF_t0_ti[i] = np.exp(-interp_y * ql.Actual365Fixed().yearFraction(dates[0], interp_dates[i]))

    return DCF_t0_ti

format_data = "%Y-%m-%dT%H:%M:%S.%f"
dates, rates = read_excel_data('MktData_CurveBootstrap.xls', format_data)

def bootstrap(dates,rates):

    # We import settlement date
    SettlementDate = dates["settlement"]

    # We import Depos dates
    DeposDates = dates["depos"]

    # We import Futures dates
    FuturesDates = dates["futures"]

    # We import Swaps dates
    SwapsDates = dates["swaps"]


    # We import Depos rates
    DeposRates = rates["depos"]

    # We import Futures dates
    FuturesRates = rates["futures"]

    # We import Swaps dates
    SwapsRates = rates["swaps"]



    # We compute the number of years between settlement date and last swap
    n_years = int((SwapsDates[-1]-SettlementDate).days/365.25)


    # Settings in order to work with the QuantLib package
    qlSettlementDate = ql.Date(SettlementDate.day, SettlementDate.month, SettlementDate.year)
    qlDeposDates = [ql.Date(date.day, date.month, date.year) for date in DeposDates]
    qlFuturesDates = [[ql.Date(date[0].day, date[0].month, date[0].year),
                       ql.Date(date[1].day, date[1].month, date[1].year)] for date in FuturesDates]
    qlSwapsDates = [ql.Date(date.day, date.month, date.year) for date in SwapsDates]

    FirstDay = ql.Date(1,2,2023)
    FestCalendar = ql.Italy()

    calendar = NextBusinessDays(FirstDay,FestCalendar,n_years)
    # Print the results
    # print("Next business days for the given date in the next", n_years, "years:")
    # for i, day in enumerate(calendar, start=1):
    #     print(f"Year {i}: {day}")


    # We initialize the dates and discounts outputs
    Dates = [qlSettlementDate]
    Discounts = [1]  # First DCF is 1



    # DEPOS

    # Now we focus on the depos, we want to consider only depos whose expiry is
    # smaller than the first settle of the futures
    ContDepos = 0

    while ContDepos + 1 < len(qlDeposDates) and qlDeposDates[ContDepos] <= qlFuturesDates[0][0]:
        ContDepos = ContDepos +1

    # We add those dates to our output vector dates
    Dates += qlDeposDates[:ContDepos]

    # Mid prices of depos
    MidDepos = np.zeros((len(DeposRates),1))
    MidDepos = np.mean(DeposRates,axis=1)



    # We update also discounts output vector with act/360



    for j in range(ContDepos):
        temp = ql.Actual360().yearFraction((qlSettlementDate),qlDeposDates[j])
        DiscountTemp = 1 / (1 + temp * MidDepos[j])
        Discounts.append(DiscountTemp)

    # Convert np.float64 to native Python float
    Discounts = [float(x) for x in Discounts]


    # FUTURES


    # Now we focus on the futures, again we consider only the futures whose
    #expiry is smaller than the first settle of the swaps

    ContFutures = 0

    while ContFutures + 1 < len(qlFuturesDates) and qlFuturesDates[ContFutures][1] <= qlSwapsDates[0]:
        ContFutures = ContFutures +1

    # We compute the mid prices of futures L(t0,ti,ti+1)
    SelectedFutures = FuturesRates[:ContFutures, :]
    MidFutures =  np.mean(SelectedFutures, axis=1)

    # We also compute the forward disount factors B(t0,ti,ti+1), again with
    # act/360



    ForwardDiscounts = []
    for j in range(ContFutures):
        StartDate = qlFuturesDates[j][0]
        EndDate = qlFuturesDates[j][1]

        temp = ql.Actual360().yearFraction(StartDate, EndDate) # Calculate the year fraction between the two dates

        DiscountTemp = 1 / (1 + temp * MidFutures[j])
        ForwardDiscounts.append(DiscountTemp)

    # Convert np.float64 to native Python float
    ForwardDiscounts = [float(x) for x in ForwardDiscounts]



    # First future: we check if its settle date is the same date as the last
    #depos, if yes they have the same DCF (B(t0,ti)) and we can calculate
    #B(t0,ti+1)=B(t0,ti)*B(t0,ti,ti+1), otherwise we obtain the B(t0,ti) by
    #interpolating with the depos we didn't consider


    if qlFuturesDates[0][0] == qlDeposDates[ContDepos]:
        index = dates.index(qlFuturesDates[0][0])
        DCF_t0_ti = Discounts[index]
    else:

        # Initialize the interpolation date
        InterpDate = qlFuturesDates[0][0]

        NewDeposDates = qlDeposDates[ContDepos]


        Dates.append(NewDeposDates)


        # Compute the mid price of the new depos
        MidDeposPrime = np.mean(DeposRates[ContDepos])


        # Compute the discount for the new depos
        temp = ql.Actual360().yearFraction(Dates[0], NewDeposDates)
        DiscountToAdd = 1 / (1 + temp * MidDeposPrime)

        # Add the new discount to the discounts list
        Discounts.append(DiscountToAdd)
        Discounts = [float(x) for x in Discounts]

        # Lastly, we interpolate the DCF_t0_ti by interpolating the corresponding zero rates
        DCF_t0_ti = zRatesInterp(Dates, Discounts, [InterpDate])

    # Print the result
    print("DCF_t0_ti:", DCF_t0_ti)


    # Update dates and discounts with the expiry for the first future and the
    # B(t0,ti+1) = B(t0,ti) * B(t0,ti,ti+1)


    Dates.append(qlFuturesDates[0][1])
    Discounts.append(DCF_t0_ti[0] * ForwardDiscounts[0])
    Discounts = [float(x) for x in Discounts]







    # Preallocate the remaining dates and discounts
    LenDiscounts = len(Discounts)
    LenDates = LenDiscounts

    # Extend the lists to accommodate the remaining futures
    Dates.extend([None] * (ContFutures - 1))
    Discounts.extend([0.0] * (ContFutures - 1))
    Discounts = [float(x) for x in Discounts]



    # Loop through the remaining futures
    for i in range(1, ContFutures):

        # Initialize settle and expiry for each future
        t_i = qlFuturesDates[i][0]  # ti
        t_i_plus = qlFuturesDates[i][1]  # ti+1

        # Compute DCF_t0_ti for each i-th future, i.e., for the settle
        DCF_t0_ti = zRatesInterp(Dates[:LenDates + i - 1], Discounts[:LenDiscounts + i - 1], [t_i])

        # Compute DCF_t0_ti_plus for each i-th future, i.e., for the expiry
        DCF_t0_ti_plus = DCF_t0_ti[0] * ForwardDiscounts[i]

        # Update dates and discounts
        Dates[LenDates + i - 1] = t_i_plus
        Discounts[LenDiscounts + i - 1] = DCF_t0_ti_plus
        Discounts = [float(x) for x in Discounts]




    # SWAPS

    # Build a complete set of swaps

    # Compute the number of years between settlement date and last swap
    nYearsSwaps = int((SwapsDates[-1] - SettlementDate).days / 365.25)


    # Initialize a vector of n+1 zeros for all the years between 2023 and 2073
    CalendarSwaps = [None] * (nYearsSwaps + 1)

    # Initialize the first value of the calendar as the settlement date
    CalendarSwaps[0] = qlSettlementDate

    CalendarSwaps = NextBusinessDays(qlSettlementDate-1,FestCalendar = ql.Italy(), n_years=nYearsSwaps+1)



    # Only consider swaps from the first given swaps, the rest we are not going to add them to the discounts
    ContSwaps = 0

    while CalendarSwaps[ContSwaps] < qlSwapsDates[0]:
        ContSwaps += 1


    # Initialize the BPV value
    BPV = 0.0

    for j in range(ContSwaps - 1):
        Bj = zRatesInterp(Dates, Discounts, [CalendarSwaps[j + 1]])[0]
        BPV += Bj * ql.Thirty360(ql.Thirty360.BondBasis).yearFraction(CalendarSwaps[j], CalendarSwaps[j + 1])





    # Now we spline interpolate the full set of swaps
    AvgSwaps = 0.5 * (SwapsRates[:, 0] + SwapsRates[:, 1])
    Avg_y = -np.log(AvgSwaps) / np.array([ql.Actual365Fixed().yearFraction(qlSettlementDate, date) for date in qlSwapsDates])



    # Interpolate the swaps



    # Interpolate the swap rates using a cubic spline
    InterpolatedSwaps = CubicSpline([date.serialNumber() for date in qlSwapsDates],AvgSwaps)([date.serialNumber() for date in CalendarSwaps[ContSwaps:]])


    # Preallocate memory for dates and discounts
    LenDates = len(Dates)
    LenDiscounts = LenDates

    # Extend the lists to accommodate the remaining swaps
    Dates.extend([None] * (len(CalendarSwaps) - ContSwaps))
    Discounts.extend([0.0] * (len(CalendarSwaps) - ContSwaps))
    Discounts = [float(x) for x in Discounts]


    # Now we compute the rest of the DCF's
    for i in range(ContSwaps, len(CalendarSwaps)):

        # Time interval between ti-1 and ti
        Delta = ql.Thirty360(ql.Thirty360.BondBasis).yearFraction(CalendarSwaps[i-1], CalendarSwaps[i])

        # Update dates
        Dates[LenDates + i - ContSwaps] = CalendarSwaps[i]

        # Compute the discount factor Bi
        Bi = (1 - InterpolatedSwaps[i - ContSwaps] * BPV) / (1 + InterpolatedSwaps[i - ContSwaps] * Delta)

        # Update discounts
        Discounts[LenDiscounts + i - ContSwaps] = Bi

        # Update BPV
        BPV += Bi * Delta

    # Convert np.float64 to native Python float
    Discounts = [float(x) for x in Discounts]


    # Print the final dates and discounts
    print("Final Dates:", Dates)
    print("Final Discounts:", Discounts)

    return Dates,Discounts

Dates,Discounts=bootstrap(dates,rates)
discounts = np.array(Discounts)  # Assuming Discounts is a list
dates = np.array(Dates)  # Assuming Dates is a list of QuantLib Date objects


# Compute zero rates
discounts_temp = discounts[1:]
zRates=np.zeros(len(discounts))
dates_temp = dates[1:]
zRates[0] = 0
zRates[1:] = -np.log(discounts_temp) / np.array([ql.Actual365Fixed().yearFraction(dates[0], d) for d in dates_temp])
zRates *= 100  # Convert to percentage


print("Final Zero Rates:", zRates)





YearFractions_zrates = np.array([ql.Actual365Fixed().yearFraction(dates[0], d) for d in dates])



# Plotting
fig, ax1 = plt.subplots()

# Plot discount factors on the left y-axis
color = 'tab:blue'
ax1.set_xlabel('Year Fraction')
ax1.set_ylabel('Discount Factors', color=color)
ax1.plot(YearFractions_zrates, np.array(discounts), '-bo', label='DCF')
ax1.tick_params(axis='y', labelcolor=color)

# Adjust the y-axis limits for discount factors
ax1.set_ylim(0.3, 1)  # Set discount factors range between 1 and 0.3

# Create a second y-axis for zero rates
ax2 = ax1.twinx()
color = 'tab:red'
ax2.set_ylabel('Zero Rates', color=color)
ax2.plot(np.array(YearFractions_zrates), np.array(zRates), '-ro', label='ZR')
ax2.tick_params(axis='y', labelcolor=color)

# Adjust the y-axis limits for zero rates
ax2.set_ylim(1.8, 3.6)  # Set zero rates range between 1.8% and 3.6%

# Add legend, title, and show plot
fig.legend(loc="upper left", bbox_to_anchor=(0.1, 0.9))
plt.title('Bootstrap DCF vs ZR')
plt.grid()
plt.show()



