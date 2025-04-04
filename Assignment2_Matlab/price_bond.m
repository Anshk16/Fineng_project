function price = price_bond(dates, discounts, ratesSet)
%PRICE_BOND Computes the price of a bond using discount factors.
%
%   price = price_bond(dates, discounts, ratesSet) calculates the bond price
%   using interpolated discount factors and the average swap rate.
%
%   Inputs:
%   - dates: Vector of known dates corresponding to discount factors.
%   - discounts: Vector of discount factors associated with known dates.
%   - ratesSet: Structure containing swap rate data.
%
%   Output:
%   - price: Computed bond price.

    % Define valuation date (settlement date)
    today = datetime(2023, 02, 02);

    % Generate bond payment dates from settlement date to 7 years ahead
    datesSet_add = Add_dates(today, 7); 

    % Convert bond payment dates to numeric format
    bond_dates = datenum(datesSet_add);

    % Compute year fractions for bond cash flows (Act/360 convention)
    y_frac_swaps = yearfrac(today, bond_dates, 6);

    % Compute the mean swap rate at the 6-year tenor
    r = mean(ratesSet.swaps(6,:));

    % Interpolate discount factors for bond payment dates
    discounts_bd = interpolation_vector(dates, discounts, datesSet_add, today);

    % Compute time intervals between payment dates
    yfrac_comp = y_frac_swaps(2:end) - y_frac_swaps(1:end-1);

    % Compute bond price as the sum of discounted cash flows
    price = r * sum(yfrac_comp .* discounts_bd(2:end)) + discounts_bd(end);

end
