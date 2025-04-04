from dataclasses import dataclass
import pandas as pd
import copy
from dateutil.relativedelta import relativedelta
from yearfrac import yearfrac, mod
from typing import List

# Define a data structure for storing various date-related DataFrames along with the settlement date.
@dataclass
class DatesSet:
    settle: str             # Settlement date as a string.
    depos: pd.DataFrame     # DataFrame containing deposit-related dates and data.
    future: pd.DataFrame    # DataFrame containing future-related dates and data.
    swap: pd.DataFrame      # DataFrame containing swap-related dates and data.

# Define a data structure for storing different rate DataFrames.
@dataclass
class RatesSet:
    depos: pd.DataFrame     # DataFrame containing deposit rates.
    future: pd.DataFrame    # DataFrame containing future rates.
    swap: pd.DataFrame      # DataFrame containing swap rates.

def shift_rates_set(
    rates_set: RatesSet, 
    dates_set: DatesSet, 
    bucket_years: List[int]
) -> List[RatesSet]:
    """
    Applies a shift on the RatesSet for each bucket specified in bucket_years,
    where each bucket represents a number of years relative to the settle date.
    Internally, the new dates are computed by adding the respective number of years
    to dates_set.settle, but these computed dates are not returned.

    Parameters:
        rates_set (RatesSet): Object containing the DataFrames for depos, future, and swap.
        dates_set (DatesSet): Object containing the settle date (as a string) and corresponding DataFrames.
        bucket_years (List[int]): List of integers representing the number of years to add to the settle date.

    Returns:
        List[RatesSet]: A list of RatesSet objects (one for each bucket; currently, the original rates_set
                        is returned as a placeholder with modified rates).
    """
    shifted_rates_list = []  # List to store the shifted RatesSet objects.
    computed_dates = []      # List to store computed new dates based on bucket_years.
    
    # Convert the settlement date string to a pandas Timestamp.
    settle_date = pd.to_datetime(dates_set.settle)
    
    # Compute new dates by adding each bucket (in years) to the settlement date.
    for bucket in bucket_years:
        new_date = settle_date + relativedelta(years=bucket)
        computed_dates.append(new_date)

    # Iterate over each bucket index to apply different shifts.
    for j in range(len(bucket_years)):
        # Create a deep copy of the original rates_set to avoid modifying it in-place.
        ratesSet_bucket = copy.deepcopy(rates_set)
        
        # For the first bucket (j == 0), apply an upward shift to the rates.
        if j == 0:
            # Process deposit rates.
            for i in range(len(ratesSet_bucket.depos)):
                # If the deposit's settle date is before or equal to the first computed date,
                # increase the 'Mid' rate by 0.01.
                if dates_set.depos['Settle Dates'].iloc[i] <= computed_dates[0]:
                    ratesSet_bucket.depos["Mid"].iloc[i] += 0.01
                # If the deposit's settle date is after or equal to the second computed date,
                # do nothing (the line is present but does nothing).
                elif dates_set.depos['Settle Dates'].iloc[i] >= computed_dates[1]:
                    ratesSet_bucket.depos["Mid"].iloc[i]
                # Otherwise, apply a proportional upward shift based on the year.
                else:
                    ratesSet_bucket.depos["Mid"].iloc[i] += 0.01 * (3 - 1/5 * (dates_set.depos['Settle Dates'].iloc[i].year - 2023))
            
            # Process future rates.
            for i in range(len(ratesSet_bucket.future)):
                # If the future's expiry date is before or equal to the first computed date,
                # increase the 'Mid' rate by 0.01.
                if dates_set.future['Expiry'].iloc[i] <= computed_dates[0]:
                    ratesSet_bucket.future["Mid"].iloc[i] += 0.01
                # If the future's expiry date is after or equal to the second computed date, do nothing.
                elif dates_set.future['Expiry'].iloc[i] >= computed_dates[1]:    
                    ratesSet_bucket.future["Mid"].iloc[i]
                # Otherwise, apply a proportional upward shift based on the year.
                else:
                    ratesSet_bucket.future["Mid"].iloc[i] += 0.01 * (3 - 1/5 * (dates_set.future['Expiry'].iloc[i].year - 2023))

            # Process swap rates.
            for i in range(len(ratesSet_bucket.swap)):
                # If the swap's date is before or equal to the first computed date,
                # increase the 'Mid' rate by 0.01.
                if dates_set.swap['Swap Dates'].iloc[i] <= computed_dates[0]:
                    ratesSet_bucket.swap["Mid"].iloc[i] += 0.01
                # If the swap's date is after or equal to the second computed date, do nothing.
                elif dates_set.swap['Swap Dates'].iloc[i] >= computed_dates[1]:
                    ratesSet_bucket.swap["Mid"].iloc[i]
                # Otherwise, apply a proportional upward shift based on the year.
                else:
                    ratesSet_bucket.swap["Mid"].iloc[i] += 0.01 * (3 - 1/5 * (dates_set.swap['Swap Dates'].iloc[i].year - 2023))

        # For subsequent buckets (j != 0), apply a downward shift to the rates.
        else:
            # Process deposit rates.
            for i in range(len(ratesSet_bucket.depos)):
                # If the deposit's settle date is before or equal to the first computed date, do nothing.
                if dates_set.depos['Settle Dates'].iloc[i] <= computed_dates[0]:
                    ratesSet_bucket.depos["Mid"].iloc[i]
                # If the deposit's settle date is after or equal to the second computed date, increase the rate by 0.01.
                elif dates_set.depos['Settle Dates'].iloc[i] >= computed_dates[1]:
                    ratesSet_bucket.depos["Mid"].iloc[i] += 0.01
                # Otherwise, apply a proportional downward shift based on the year.
                else:
                    ratesSet_bucket.depos["Mid"].iloc[i] += 0.01 * (-2 + 1/5 * (dates_set.depos['Settle Dates'].iloc[i].year - 2023))
            
            # Process future rates.
            for i in range(len(ratesSet_bucket.future)):
                # If the future's expiry date is before or equal to the first computed date, do nothing.
                if dates_set.future['Expiry'].iloc[i] <= computed_dates[0]:
                    ratesSet_bucket.future["Mid"].iloc[i]
                # If the future's expiry date is after or equal to the second computed date, increase the rate by 0.01.
                elif dates_set.future['Expiry'].iloc[i] >= computed_dates[1]:
                    ratesSet_bucket.future["Mid"].iloc[i] += 0.01
                # Otherwise, apply a proportional downward shift based on the year.
                else:
                    ratesSet_bucket.future["Mid"].iloc[i] += 0.01 * (-2 + 1/5 * (dates_set.future['Expiry'].iloc[i].year - 2023))

            # Process swap rates.
            for i in range(len(ratesSet_bucket.swap)):
                # If the swap's date is before or equal to the first computed date, do nothing.
                if dates_set.swap['Swap Dates'].iloc[i] <= computed_dates[0]:
                    ratesSet_bucket.swap["Mid"].iloc[i]
                # If the swap's date is after or equal to the second computed date, increase the rate by 0.01.
                elif dates_set.swap['Swap Dates'].iloc[i] >= computed_dates[1]:
                    ratesSet_bucket.swap["Mid"].iloc[i] += 0.01
                # Otherwise, apply a proportional downward shift based on the year.
                else:
                    ratesSet_bucket.swap["Mid"].iloc[i] += 0.01 * (-2 + 1/5 * (dates_set.swap['Swap Dates'].iloc[i].year - 2023))

        # Append the shifted RatesSet for this bucket to the list.
        shifted_rates_list.append(ratesSet_bucket)
    
    # Return the list of shifted RatesSet objects.
    return shifted_rates_list
