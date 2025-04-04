# Importing the libraries
### !!! IMPORT USEFUL LIBRARIES HERE !!! ###
import pandas as pd
import copy
import math
import warnings
from bootstrap import bootstrap
from bucket_rates import shift_rates_set
from readExcelData import readExcelData
from Q7_scenario_rates_adj import Q7_scenario_rates_adj
from ex1_utilities import (
    business_date_offset,
    swaption_price_calculator,
    date_series,
    irs_proxy_duration,
    swap_par_rate,
    swap_mtm,
    SwapType,
)

# --------------------- PARAMETERS -------------------------
# Swaption parameters
swaption_maturity_y = 10          # Swaption maturity in years
swaption_maturity_m = 1           # Swaption maturity in months
swaption_tenor_y = 5              # Underlying swap tenor in years
swaption_fixed_leg_freq = 1       # Frequency (payments per year) of the swaption's fixed leg
swaption_type = SwapType.RECEIVER # Swaption type: receiver option
swaption_notional = 700_000_000   # Notional amount for the swaption
sigma_black = 0.7955              # Black swaption volatility

# IRS (Interest Rate Swap) parameters
irs_maturity = 10                 # IRS maturity in years
irs_fixed_leg_freq = 1            # Frequency (payments per year) of the IRS fixed leg
irs_notional = 600_000_000        # Notional amount for the IRS

# --------------------- READ MARKET AND BOOTSTRAP DATA -------------------------
# Read market data from Excel and obtain dates and rates information.
[datesSet, ratesSet] = readExcelData()

# Bootstrap to calculate discount factors based on market data.
dates, discount_factors_appo = bootstrap(datesSet, ratesSet)

# Ensure that the 'Date' column is in datetime format.
dates["Date"] = pd.to_datetime(dates["Date"])

# Create a Series using the 'Discount Factor' column as data and the 'Date' column as the index.
discount_factors = pd.Series(
    data=discount_factors_appo["Discount Factor"].values,
    index=dates["Date"].values
)

# Set the current date as the settlement date from the datesSet.
today = datesSet.settle

# --------------------- Q1: PORTFOLIO MTM COMPUTATION -------------------------
print('##############################################')
print('###############     Q1      ##################\n')

# Compute the swaption expiry date by applying maturity offsets to the current date.
swaption_expiry = business_date_offset(
    today, year_offset=swaption_maturity_y, month_offset=swaption_maturity_m
)

# Compute the underlying swap expiry date by adding the swap tenor to the swaption expiry.
underlying_expiry = business_date_offset(
    today,
    year_offset=swaption_maturity_y + swaption_tenor_y,
    month_offset=swaption_maturity_m,
)

# Generate the fixed leg payment schedule for the underlying swap.
swaption_underlying_fixed_leg_schedule = date_series(
    swaption_expiry, underlying_expiry, swaption_fixed_leg_freq
)

# Calculate the forward swap rate (ATM strike) using the swap par rate function.
fwd_swap_rate = swap_par_rate(
    swaption_underlying_fixed_leg_schedule[1:],
    discount_factors,
    swaption_underlying_fixed_leg_schedule[0]
)
print(f"Forward swap rate: {fwd_swap_rate:.5%}")
print('---')

# --------------------- Q1: SWAPTION PRICE COMPUTATION -------------------------
# For an ATM swaption, the strike is equal to the forward swap rate.
strike = fwd_swap_rate  

# Calculate swaption price and its delta using the Black model.
swaption_price, swaption_delta = swaption_price_calculator(
    fwd_swap_rate,
    strike,
    today,
    swaption_expiry,
    underlying_expiry,
    sigma_black,
    swaption_fixed_leg_freq,
    discount_factors,
    swaption_type,
    compute_delta=True,
)

# Print the computed swaption price (scaled by notional) and delta.
print(f"Swaption price: €{swaption_price * swaption_notional:,.2f}")
print(f"Swaption delta: {swaption_delta:.4f}")
print('---')

# --------------------- Q1: SWAP MTM COMPUTATION -------------------------
# Compute IRS expiry date based on the current date and IRS maturity.
irs_expiry = business_date_offset(today, year_offset=irs_maturity)

# Generate the fixed leg payment dates for the IRS (excluding the first date).
irs_fixed_leg_payment_dates = date_series(today, irs_expiry, irs_fixed_leg_freq)[1:]

# Calculate the IRS par rate.
irs_rate = swap_par_rate(irs_fixed_leg_payment_dates, discount_factors)
print(f"IRS rate: {irs_rate:.5%}")

# Compute the mark-to-market (MtM) of the IRS.
irs_mtm = swap_mtm(irs_rate, irs_fixed_leg_payment_dates, discount_factors)
print(f"IRS MtM: €{irs_mtm:,.2f}")
print('---')
# The IRS MtM is zero if the swap is entered today; thus, the portfolio MtM equals the swaption value.

# Compute the overall portfolio MtM.
ptf_mtm = swaption_notional * swaption_price + irs_notional * irs_mtm
print(f"Portfolio MtM: €{ptf_mtm:,.2f}")

# --------------------- Q2: PORTFOLIO DV01 (PARALLEL SHIFT) -------------------------
print('\n\n##############################################')
print('###############     Q2      ##################\n')

# Create a deep copy of the ratesSet to apply a parallel shift (increase all rates by 1 basis point).
ratesSet_up = copy.deepcopy(ratesSet)

# Apply a 0.01 (1 basis point) upward shift to all 'Mid' rates.
ratesSet_up.depos["Mid"] += 0.01
ratesSet_up.future["Mid"] += 0.01
ratesSet_up.swap["Mid"] += 0.01 

# Re-bootstrap discount factors using the shifted rates.
dates, discount_factors_appo = bootstrap(datesSet, ratesSet_up)

# Ensure that the 'Date' column is in datetime format.
dates["Date"] = pd.to_datetime(dates["Date"])

# Create a Series for the new discount factors.
discount_factors_up = pd.Series(
    data=discount_factors_appo["Discount Factor"].values,
    index=dates["Date"].values
)

# Recalculate the forward swap rate with the shifted discount factors.
fwd_swap_rate_up = swap_par_rate(
    swaption_underlying_fixed_leg_schedule[1:], discount_factors_up, swaption_underlying_fixed_leg_schedule[0]
)

# Calculate the swaption price and delta under the shifted scenario.
swaption_price_up, swaption_delta_up = swaption_price_calculator(
    fwd_swap_rate_up,
    strike,
    today,
    swaption_expiry,
    underlying_expiry,
    sigma_black,
    swaption_fixed_leg_freq,
    discount_factors_up,
    swaption_type,
)

# Compute the IRS MtM under the shifted scenario.
irs_mtm_up = swap_mtm(irs_rate, irs_fixed_leg_payment_dates, discount_factors_up)
print(f"IRS MtM-parallel: €{irs_mtm_up * irs_notional:,.2f}")
print('---')

# Compute DV01 (sensitivity) for swaption and IRS
DV01_swaption = swaption_notional * (swaption_price_up - swaption_price)
DV01_irs = irs_notional * (irs_mtm_up - irs_mtm)
print(f"Swaption DV01-parallel: €{DV01_swaption:,.2f}")
print(f"IRS DV01-parallel: €{DV01_irs:,.2f}")
print('---')

# Compute the overall portfolio DV01 (difference in portfolio MtM after the shock).
ptf_mtm_up = swaption_notional * swaption_price_up + irs_notional * irs_mtm_up
ptf_numeric_dv01 = ptf_mtm_up - ptf_mtm
print(f"Portfolio DV01-parallel: €{ptf_numeric_dv01:,.2f}")

# --------------------- Q3: ANALYTICAL PORTFOLIO DV01 -------------------------
print('\n\n##############################################')
print('###############     Q3      ##################\n')

# Calculate the IRS duration as a proxy for its DV01.
irs_duration = irs_proxy_duration(
    today, irs_rate, irs_fixed_leg_payment_dates, discount_factors
)

# Print the approximated DV01 for the swaption (using its delta) and IRS.
print(f'Swaption DV01-approx: €{swaption_notional * swaption_delta * 1e-4:,.2f}') 
print(f'IRS DV01-approx: €{irs_notional * irs_duration * 1e-4:,.2f}')
print(' ')

# Compute the overall portfolio proxy DV01.
ptf_proxy_dv01 = (
    swaption_notional * swaption_delta + irs_notional * irs_duration
) * 1e-4
print(f"Portfolio proxy DV01: €{ptf_proxy_dv01:,.2f}")

# --------------------- Q4: DELTA HEDGING OF THE SWAPTION (CHANGING IRS NOTIONAL) -------------------------
print('\n\n##############################################')
print('###############     Q4      ##################\n')

min_lot = 1_000_000  # Minimum lot size for IRS notional adjustments

# Compute the required IRS notional adjustment to hedge DV01.
delta_hedge_swap_notional = - ptf_numeric_dv01 / (irs_mtm_up - irs_mtm)
# Round down to the nearest minimum lot.
delta_hedge_swap_notional = math.floor(delta_hedge_swap_notional / min_lot) * min_lot

# Calculate the residual DV01 after hedging.
delta_hedge_dv01 = (
    swaption_notional * swaption_price_up + delta_hedge_swap_notional * irs_mtm_up + irs_notional * irs_mtm_up
) - ptf_mtm - delta_hedge_swap_notional * irs_mtm
print(
    f"With €{delta_hedge_swap_notional:,.0f} swap notional the DV01 is €{delta_hedge_dv01:,.0f}."
)
print(f'Net IRS Notional: €{irs_notional + delta_hedge_swap_notional:,.0f}')

# Using the approximated DV01 (proxy) for hedging.
delta_hedge_swap_notional_proxy = - ptf_proxy_dv01 / (irs_duration * 1e-4)
delta_hedge_swap_notional_proxy = math.floor(delta_hedge_swap_notional_proxy / min_lot) * min_lot

delta_hedge_dv01_proxy = (
    swaption_notional * swaption_price_up + delta_hedge_swap_notional_proxy * irs_mtm_up + irs_notional * irs_mtm_up
) - ptf_mtm - delta_hedge_swap_notional_proxy * irs_mtm
print(
    f"With €{delta_hedge_swap_notional_proxy:,.0f} swap notional the DV01 is €{delta_hedge_dv01_proxy:,.0f}."
)
print(f'Net IRS Notional: €{irs_notional + delta_hedge_swap_notional_proxy:,.0f}')

# --------------------- Q5: COARSE-GRAINED BUCKET DV01 -------------------------
print('\n\n##############################################')
print('###############     Q5      ##################\n')
warnings.simplefilter("ignore")  # Suppress warnings for cleaner output

# Shift the ratesSet using bucket years of 10 and 15 years.
ratesSet_bucket = shift_rates_set(ratesSet, datesSet, [10, 15])
DV01_ptf = 0  # Initialize total portfolio DV01 accumulator

# Iterate over each shifted rates set (bucket)
for i, rate_set in enumerate(ratesSet_bucket, start=1):

    # Re-bootstrap discount factors using the current bucket's rates.
    dates, discount_factors_appo = bootstrap(datesSet, rate_set)
    dates["Date"] = pd.to_datetime(dates["Date"])
    discount_factors_bucket = pd.Series(
        data=discount_factors_appo["Discount Factor"].values,
        index=dates["Date"].values
    )

    # Recalculate the forward swap rate for the bucket.
    fwd_swap_bucket = swap_par_rate(
        swaption_underlying_fixed_leg_schedule[1:],
        discount_factors_bucket, 
        swaption_underlying_fixed_leg_schedule[0]
    )

    # Compute swaption price and delta under the bucket scenario.
    swaption_price_bucket, swaption_delta_bucket = swaption_price_calculator(
        fwd_swap_bucket,
        strike,
        today,
        swaption_expiry,
        underlying_expiry,
        sigma_black,
        swaption_fixed_leg_freq,
        discount_factors_bucket,
        swaption_type,
    )
    
    # Compute IRS MtM under the bucket scenario.
    irs_mtm_bucket = swap_mtm(irs_rate, irs_fixed_leg_payment_dates, discount_factors_bucket)

    # Calculate DV01 for swaption and IRS in this bucket.
    DV01_swaption = swaption_notional * (swaption_price_bucket - swaption_price)
    DV01_irs = irs_notional * (irs_mtm_bucket - irs_mtm)
    print(f"Swaption DV01-parallel bucket {i}: €{DV01_swaption:,.2f}")
    print(f"IRS DV01-parallel bucket {i}: €{DV01_irs:,.2f}")
    print('---')

    # Compute portfolio MtM shock in this bucket.
    ptf_mtm_bucket = swaption_notional * swaption_price_bucket + irs_notional * irs_mtm_bucket
    ptf_numeric_dv01 = ptf_mtm_bucket - ptf_mtm
    print(f"Portfolio DV01-parallel bucket {i}: €{ptf_numeric_dv01:,.2f}")
    print('\n')

    # Accumulate the portfolio DV01 across buckets.
    DV01_ptf += ptf_numeric_dv01

print('---')
print(f'Portfolio Total DV01-parallel: €{DV01_ptf:,.2f}')
warnings.simplefilter("default")  # Reset warnings filter

# --------------------- Q6: DELTA HEDGING WITH TWO IRS -------------------------
print('\n\n##############################################')
print('###############     Q6      ##################\n')

min_lot = 1_000_000

# Recompute IRS DV01 for the base IRS.
DV01_irs = irs_notional * (irs_mtm_up - irs_mtm)
# Compute the notional adjustment for the 10-year IRS hedge.
delta_hedge_swap_notional_10y = - DV01_irs / (irs_mtm_up - irs_mtm)
delta_hedge_swap_notional_10y = math.floor(delta_hedge_swap_notional_10y / min_lot) * min_lot
print(f'Notional IRS 10y for hedging the IRS: €{delta_hedge_swap_notional_10y:,.2f}')

# Recompute swaption DV01.
DV01_swaption = swaption_notional * (swaption_price_up - swaption_price)

# For the 15-year IRS, compute expiry and fixed leg dates.
irs_15y_maturity = 15
irs_15y_expiry = business_date_offset(today, year_offset=irs_15y_maturity)
irs_15y_fixed_leg_payment_dates = date_series(today, irs_15y_expiry, irs_fixed_leg_freq)[1:]

# Compute the 15-year IRS rate and its MtM.
irs_15y_rate = swap_par_rate(irs_15y_fixed_leg_payment_dates, discount_factors)
irs_15y_mtm = swap_mtm(irs_15y_rate, irs_15y_fixed_leg_payment_dates, discount_factors)
irs_15y_mtm_up = swap_mtm(irs_15y_rate, irs_15y_fixed_leg_payment_dates, discount_factors_up)

# Compute the notional adjustment for the 15-year IRS hedge.
delta_hedge_swap_notional_15y = - DV01_swaption / (irs_15y_mtm_up - irs_15y_mtm)
delta_hedge_swap_notional_15y = math.ceil(delta_hedge_swap_notional_15y / min_lot) * min_lot
print(f'Notional IRS 15y for hedging the Swaption: €{delta_hedge_swap_notional_15y:,.2f}')
print('---')

# Compute the overall hedged portfolio DV01 using a formula.
DV01_ptf_hedged_computed = (
    swaption_notional * swaption_price_up + delta_hedge_swap_notional_10y * irs_mtm_up + irs_notional * irs_mtm_up + delta_hedge_swap_notional_15y * irs_15y_mtm_up
) - (ptf_mtm + delta_hedge_swap_notional_10y * irs_mtm + delta_hedge_swap_notional_15y * irs_15y_mtm)

# Compute the hedged DV01 using the bucket approach.
DV01_ptf_hedged = 0
for i, rate_set in enumerate(ratesSet_bucket, start=1):

    dates, discount_factors_appo = bootstrap(datesSet, rate_set)
    dates["Date"] = pd.to_datetime(dates["Date"])
    discount_factors_bucket = pd.Series(
        data=discount_factors_appo["Discount Factor"].values,
        index=dates["Date"].values
    )

    # Compute swaption price under bucket conditions.
    swaption_price_bucket, swaption_delta_bucket = swaption_price_calculator(
        fwd_swap_rate,
        strike,
        today,
        swaption_expiry,
        underlying_expiry,
        sigma_black,
        swaption_fixed_leg_freq,
        discount_factors_bucket,
        swaption_type,
    )
    
    # Compute the 15-year IRS MtM for this bucket.
    irs_15y_mtm_bucket = swap_mtm(irs_15y_rate, irs_15y_fixed_leg_payment_dates, discount_factors_bucket)

    DV01_swaption = swaption_notional * (swaption_price_bucket - swaption_price)
    DV01_irs = delta_hedge_swap_notional_15y * (irs_15y_mtm_bucket - irs_15y_mtm)
    print(f"Swaption DV01-parallel bucket {i}: €{DV01_swaption:,.2f}")
    print(f"IRS DV01-parallel bucket {i}: €{DV01_irs:,.2f}")
    print('---')

    ptf_hedged_numeric_dv01 = DV01_swaption + DV01_irs
    print(f"Portfolio DV01-parallel bucket {i}: €{ptf_hedged_numeric_dv01:,.2f}")
    print('\n')

    DV01_ptf_hedged += ptf_hedged_numeric_dv01

print('---')
print(f'Portfolio Total DV01-parallel computed: €{DV01_ptf_hedged_computed:,.2f}')
print(f'Portfolio Total DV01-parallel bucket: €{DV01_ptf_hedged:,.2f}')

# --------------------- Q7: CURVE STEEPENER SCENARIO -------------------------
print('\n\n##############################################')
print('###############     Q7      ##################\n')

# Apply a curve steepener scenario adjustment on the rates.
warnings.simplefilter("ignore")
shifted_rates = Q7_scenario_rates_adj(ratesSet, datesSet)
warnings.simplefilter("default")

# Re-bootstrap discount factors using the shifted rates.
dates, discount_factors_appo = bootstrap(datesSet, shifted_rates)
dates["Date"] = pd.to_datetime(dates["Date"])
discount_factors_q7 = pd.Series(
    data=discount_factors_appo["Discount Factor"].values,
    index=dates["Date"].values
)

# Base portfolio MtM remains the same.
ptf_mtm = swaption_notional * swaption_price

# Recalculate the forward swap rate using the adjusted discount factors.
fwd_swap_rate_q7 = swap_par_rate(
    swaption_underlying_fixed_leg_schedule[1:],
    discount_factors_q7, 
    swaption_underlying_fixed_leg_schedule[0]
)

# Compute the swaption price and delta under the curve steepener scenario.
swaption_price_q7, swaption_delta_q7 = swaption_price_calculator(
    fwd_swap_rate_q7,
    strike,
    today,
    swaption_expiry,
    underlying_expiry,
    sigma_black,
    swaption_fixed_leg_freq,
    discount_factors_q7,
    swaption_type,
)

# Compute the IRS MtM under the adjusted scenario.
irs_mtm_q7a = swap_mtm(irs_rate, irs_fixed_leg_payment_dates, discount_factors_q7)

# Compute the portfolio MtM using the adjusted swaption price and base IRS notional.
ptf_mtm_q7a = swaption_notional * swaption_price_q7 + irs_notional * irs_mtm_q7a
print(f'Q7.a: €{ptf_mtm_q7a - ptf_mtm:,.2f}')

# Compute the IRS MtM for the 15-year IRS under the adjusted scenario.
irs_mtm_q7b = swap_mtm(irs_15y_rate, irs_15y_fixed_leg_payment_dates, discount_factors_q7)
ptf_mtm_q7b = swaption_notional * swaption_price_q7 + delta_hedge_swap_notional_15y * irs_mtm_q7b
print(f'Q7.b: €{ptf_mtm_q7b - ptf_mtm:,.2f}\n\n')
