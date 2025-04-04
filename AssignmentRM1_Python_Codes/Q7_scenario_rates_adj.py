from dataclasses import dataclass
import pandas as pd
from dateutil.relativedelta import relativedelta

# Data class to store various dates information including settlement and corresponding DataFrames.
@dataclass
class DatesSet:
    settle: str               # Settlement date as a string.
    depos: pd.DataFrame       # DataFrame containing deposit dates and related information.
    future: pd.DataFrame      # DataFrame containing future expiry dates and related information.
    swap: pd.DataFrame        # DataFrame containing swap dates and related information.

# Data class to store rate information for deposits, futures, and swaps.
@dataclass
class RatesSet:
    depos: pd.DataFrame       # DataFrame containing deposit rates.
    future: pd.DataFrame      # DataFrame containing future rates.
    swap: pd.DataFrame        # DataFrame containing swap rates.

def Q7_scenario_rates_adj(
    rates_set: RatesSet, 
    dates_set: DatesSet,
) -> RatesSet:
    # Convert the settlement date string to a pandas Timestamp.
    settle_date = pd.to_datetime(dates_set.settle)
    
    # Calculate target dates: 10 and 15 years after the settlement date.
    date_10 = settle_date + relativedelta(years=10)
    date_15 = settle_date + relativedelta(years=15)
    
    # Update deposit rates:
    for i in range(len(rates_set.depos)):
        # Get the deposit's settle date from the DataFrame.
        d = dates_set.depos['Settle Dates'].iloc[i]
        # If the deposit date matches the 15-year target, add 0.01 to the 'Mid' rate.
        if d == date_15:
            rates_set.depos["Mid"].iloc[i] += 0.01
        # If the deposit date matches the 10-year target, subtract 0.01 from the 'Mid' rate.
        elif d == date_10:
            rates_set.depos["Mid"].iloc[i] -= 0.01

    # Update future rates:
    for i in range(len(rates_set.future)):
        # Get the future's expiry date from the DataFrame.
        d = dates_set.future['Expiry'].iloc[i]
        # If the expiry date matches the 15-year target, add 0.01 to the 'Mid' rate.
        if d == date_15:
            rates_set.future["Mid"].iloc[i] += 0.01
        # If the expiry date matches the 10-year target, subtract 0.01 from the 'Mid' rate.
        elif d == date_10:
            rates_set.future["Mid"].iloc[i] -= 0.01

    # Update swap rates:
    for i in range(len(rates_set.swap)):
        # Get the swap date from the DataFrame.
        d = dates_set.swap['Swap Dates'].iloc[i]
        # If the swap date matches the 15-year target, add 0.01 to the 'Mid' rate.
        if d == date_15:
            rates_set.swap["Mid"].iloc[i] += 0.01
        # If the swap date matches the 10-year target, subtract 0.01 from the 'Mid' rate.
        elif d == date_10:
            rates_set.swap["Mid"].iloc[i] -= 0.01
            
    # Return the adjusted rates set.
    return rates_set
