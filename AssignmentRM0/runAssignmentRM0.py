# Import custom functions and necessary libraries.
from readExcelData import readExcelData
from add_Dates import add_Dates, mod
from bootstrap import bootstrap
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
from zeroRates import zeroRates

# Read data from an Excel file using the custom function.
# The function returns two objects: one containing date information and one containing rate data.
[datesSet, ratesSet] = readExcelData()

# Print the retrieved dates and rates to the console.
print("Settlement Date:", datesSet.settle)
print("Depo Dates:\n", datesSet.depos)
print("Future Dates:\n", datesSet.future)
print("Swap Dates:\n", datesSet.swap)

import pandas as pd
pd.set_option('display.float_format', '{:.5f}'.format)

print("Depo Rates:\n", ratesSet.depos)
print("Future Rates:\n", ratesSet.future)
print("Swap Rates:\n", ratesSet.swap)

# Generate annual dates using the settlement date.
# 'years' specifies the number of years (50 in this case) and mod.Modified is a mode parameter.
years = 50
df = add_Dates(datesSet.settle, years, mod.Modified)

# Calculate discount factors using the bootstrap method.
dates, discounts = bootstrap(datesSet, ratesSet)

# Print the aggregated and sorted dates.
print("Aggregated (sorted) dates:")
print(dates)

# Set the display format for floating-point numbers to 12 decimal places and print the discount factors.
print("Aggregated (sorted) discounts:")
pd.set_option('display.float_format', '{:.12f}'.format)
print(discounts)

# First Plot: Bootstrapped Discount Curve
plt.figure(figsize=(12, 8))  # Create a figure of size 12x8 inches.
plt.plot(dates, discounts, marker='o', markersize=6, linestyle='-', linewidth=2, label='Discount Factor')
plt.xlabel('Date', fontsize=14)  # Set the x-axis label.
plt.ylabel('Discount Factor', fontsize=14)  # Set the y-axis label.
plt.title('Bootstrapped Discount Curve', fontsize=16, fontweight='bold')  # Set the plot title.
plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)  # Enable grid lines.
plt.legend(fontsize=12)  # Display the legend.
plt.gcf().autofmt_xdate()  # Auto-format the x-axis dates for better appearance.
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))  # Define the date format.
plt.tight_layout()  # Adjust subplots to fit the figure area.
plt.show()  # Display the discount curve plot.
# Calculate the zero-coupon interest rates based on the discount factors.
df_zero = zeroRates(dates, discounts)

# Second Plot: Zero-Coupon Interest Rates
plt.figure(figsize=(12, 8))  # Create a new figure with the specified size.
# Exclude the first element of the 'dates' DataFrame (assumes the first date is not needed).
plot_dates = dates["Date"].iloc[1:].reset_index(drop=True)
# Convert the 'Zero Rate' column to float (assuming df_zero has already removed the first element).
plot_zero = df_zero["Zero Rate"].astype(float)
plt.plot(plot_dates, plot_zero, marker='o', linestyle='-', linewidth=2, markersize=6, label='Zero Rates')
plt.xlabel('Date', fontsize=12)  # Set the x-axis label.
plt.ylabel('Zero Rate (%)', fontsize=12)  # Set the y-axis label.
plt.title('Zero-Coupon Interest Rates', fontsize=14, fontweight='bold')  # Set the plot title.
plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)  # Enable grid lines.
plt.legend(fontsize=12)  # Display the legend.
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))  # Define the date format for the x-axis.
plt.gcf().autofmt_xdate()  # Auto-format the x-axis dates.
plt.tight_layout()  # Adjust the layout.
plt.show()  # Display the zero-coupon rates plot.