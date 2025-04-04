# Importing the libraries
import pandas as pd
import math
import numpy as np
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'utilities'))

from utilities.bootstrap import bootstrap
from utilities.readExcelData import readExcelData
from scipy.optimize import fsolve
from utilities.ex1_utilities import business_date_offset, year_frac_act_x
from utilities.ex2_utilities import (
    defaultable_bond_dirty_price_from_intensity,
    defaultable_bond_dirty_price_from_z_spread,
    defaultable_bond_dirty_price_from_intensity_and_previous_lambda
)

# Se il sistema operativo è Windows usa 'cls', altrimenti usa 'clear'
os.system('cls' if os.name == 'nt' else 'clear')

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


# Parameters
maturity1 = 1  # Maturity in years
maturity2 = 2
notional1 = 1e7
notional2 = 1e7
coupon_rate1 = 0.05
coupon_rate2 = 0.06
coupon_freq1 = 2  # Coupon frequency in payments a years
coupon_freq2 = 2
dirty_price1 = 100
dirty_price2 = 102

rating = "IG"  # Credit rating

expiry1 = business_date_offset(today, year_offset=maturity1)
expiry2 = business_date_offset(today, year_offset=maturity2)

# Q1: Derive the average intensity for the two bonds
print('##############################################')
print('###############     Q1      ##################\n')

recovery_rate = 0.3

h_1y = fsolve(
    lambda intensity: defaultable_bond_dirty_price_from_intensity(
        today,
        expiry1,
        coupon_rate1,
        coupon_freq1,
        recovery_rate,
        intensity[0],
        discount_factors,
        100,
    )
    - dirty_price1,
    x0=0.02,
)[0]

h_2y = fsolve(
    lambda intensity: defaultable_bond_dirty_price_from_intensity(
        today,
        expiry2,
        coupon_rate2,
        coupon_freq2,
        recovery_rate,
        intensity[0],
        discount_factors,
        100,
    )
    - dirty_price2,
    x0=0.02,
)[0]

print(f"Average intensity over {maturity1}y: {h_1y:.5%}")
print(f"Average intensity over {maturity2}y: {h_2y:.5%}")


# Q2: Default probability estimates
print('\n\n##############################################')
print('###############     Q2      ##################\n')

yfrac_1y = year_frac_act_x(today, expiry1, 365)
yfrac_2y = year_frac_act_x(today, expiry2, 365)

# Survival probabilities using h_2y for both bonds sice it contains more information then the first one
surv_prob_1y = math.exp(- h_2y * yfrac_1y) 
surv_prob_2y = math.exp(- h_2y * yfrac_2y)

# Defaul probabilities
default_prob_1y = 1 - surv_prob_1y
default_prob_2y = 1 - surv_prob_2y

print(f"{maturity1}y default probability: {default_prob_1y:.5%}")
print(f"{maturity2}y default probability: {default_prob_2y:.5%}")


# Q3: Z-spread calculation
print('\n\n##############################################')
print('###############     Q3      ##################\n')

z_spread_1y = fsolve(
    lambda z_spread: defaultable_bond_dirty_price_from_z_spread(
        today,
        expiry1,
        coupon_rate1,
        coupon_freq1,
        z_spread[0],
        discount_factors,
        100,
    )
    - dirty_price1,
    x0=0.02,
)[0]

z_spread_2y = fsolve(
    lambda z_spread: defaultable_bond_dirty_price_from_z_spread(
        today,
        expiry2,
        coupon_rate2,
        coupon_freq2,
        z_spread[0],
        discount_factors,
        100,
    )
    - dirty_price2,
    x0=0.02,
)[0]

print(f"Z-spread over {maturity1}y: {z_spread_1y:.5%}")
print(f"Z-spread over {maturity2}y: {z_spread_2y:.5%}")



# Q4: Default probability estimates under the hp. of piecewise constant intensity
print('\n\n##############################################')
print('###############     Q4      ##################\n')

h_1y2y = fsolve(
    lambda intensity: defaultable_bond_dirty_price_from_intensity_and_previous_lambda(
        today,
        expiry2,
        coupon_rate2,
        coupon_freq2,
        recovery_rate,
        intensity[0],
        discount_factors,
        h_1y,
        expiry1,
        100,
    )
    - dirty_price2,
    x0=0.02,
)[0]


deltas_1y = year_frac_act_x(today, expiry1, 365)
deltas_2y = year_frac_act_x(expiry1, expiry2, 365)

# Survival probabilities
surv_prob_1y = math.exp(- h_1y * deltas_1y)
surv_prob_2y = math.exp(- h_1y2y * deltas_2y - h_1y * deltas_1y)

# Defaul probabilities
default_prob_1y = 1 - surv_prob_1y
default_prob_2y = 1 - surv_prob_2y

print(f"h_1y: {h_1y:.5%}")
print(f"h_1y2y: {h_1y2y:.5%}")
print('---')
print(f"{maturity1}y default probability: {default_prob_1y:.5%}")
print(f"{maturity2}y default probability: {default_prob_2y:.5%}")



# Q5:Real world default probability from the rating transition matrix
print('\n\n##############################################')
print('###############     Q5      ##################\n')

# Simplified rating transition matrix at 1y
transition_matrix = pd.DataFrame(
    [[0.73, 0.25, 0.02], [0.35, 0.6, 0.05], [0, 0, 1]],
    index=["IG", "HY", "Def"],
    columns=["IG", "HY", "Def"],
)

# Convert DataFrame to NumPy array for matrix operations
P_1 = transition_matrix.to_numpy()

# Compute 2-year transition matrix (P²)
P_2 = np.linalg.matrix_power(P_1, 2)

print(
    f"One year real world default probability: {transition_matrix.at[rating, 'Def']:.2%}"
)
print(
    f"Two year real world default probability: {P_2[0,2]:.2%}"
)


# Q6: Estimate the default probabilities under a shock scenario of the mid-term survival probability (Scenario1)
print('\n\n##############################################')
print('###############     Q6      ##################\n')

dirty_price1_shock = dirty_price1
dirty_price2_shock = 97.0

h_1y_shock = h_1y
h_1y2y_shock = fsolve(
    lambda intensity: defaultable_bond_dirty_price_from_intensity_and_previous_lambda(
        today,
        expiry2,
        coupon_rate2,
        coupon_freq2,
        recovery_rate,
        intensity[0],
        discount_factors,
        h_1y_shock,
        expiry1,
        100,
    )
    - dirty_price2_shock,
    x0=0.02,
)[0]

# Survival probabilities
surv_prob_1y_shock = math.exp(- h_1y_shock * deltas_1y)
surv_prob_2y_shock = math.exp(- h_1y2y_shock * deltas_2y - h_1y_shock * deltas_1y)

# Defaul probabilities
default_prob_1y_shock = 1 - surv_prob_1y_shock
default_prob_2y_shock = 1 - surv_prob_2y_shock

print(f"{maturity1}y default probability: {default_prob_1y_shock:.5%}")
print(f"{maturity2}y default probability: {default_prob_2y_shock:.5%}")



# Q7: Estimate the default probabilities under a shock scenario on overall creditworthiness (Scenario2)
print('\n\n##############################################')
print('###############     Q7      ##################\n')

dirty_price1_shock2 = 101.0
dirty_price2_shock2 = 103.0

h_1y_shock2 = fsolve(
    lambda intensity: defaultable_bond_dirty_price_from_intensity(
        today,
        expiry1,
        coupon_rate1,
        coupon_freq1,
        recovery_rate,
        intensity[0],
        discount_factors,
        100,
    )
    - dirty_price1_shock2,
    x0=0.02,
)[0]

h_1y2y_shock2 = fsolve(
    lambda intensity: defaultable_bond_dirty_price_from_intensity_and_previous_lambda(
        today,
        expiry2,
        coupon_rate2,
        coupon_freq2,
        recovery_rate,
        intensity[0],
        discount_factors,
        h_1y_shock2,
        expiry1,
        100,
    )
    - dirty_price2_shock2,
    x0=0.02,
)[0]

# Survival probabilities
surv_prob_1y_shock2 = math.exp(- h_1y_shock2 * deltas_1y)
surv_prob_2y_shock2 = math.exp(- h_1y2y_shock2 * deltas_2y - h_1y_shock2 * deltas_1y)

# Defaul probabilities
default_prob_1y_shock2 = 1 - surv_prob_1y_shock2
default_prob_2y_shock2 = 1 - surv_prob_2y_shock2

print(f"{maturity1}y default probability: {default_prob_1y_shock2:.2%}")
print(f"{maturity2}y default probability: {default_prob_2y_shock2:.2%}")
print('\n')


# Q8: Are the conditional default probabilities between the first and the second year derived under the scenario 1
# consistent with the equivalent real-world probabilities derived from the transition matrix?
print('\n\n##############################################')
print('###############     Q8      ##################\n')

surv_prob_1y2y_shock = math.exp(- h_1y2y_shock * deltas_2y)
default_prob_1y2y_shock = 1 - surv_prob_1y2y_shock
print(f"{maturity1}y{maturity2}y default probability: {default_prob_1y2y_shock:.5%}")
