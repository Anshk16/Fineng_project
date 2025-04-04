function zRates = zeroRates(dates, discounts)
%ZERO_RATES Computes zero-coupon interest rates from discount factors.
%
%   zRates = zeroRates(dates, discounts) returns the zero-coupon interest 
%   rates (in percentage) given a set of future dates and corresponding 
%   discount factors.
%
%   Inputs:
%   - dates: A vector of future dates (datetime format) for which 
%     the zero rates need to be calculated.
%   - discounts: A vector of discount factors corresponding to the 
%     given dates.
%
%   Output:
%   - zRates: A vector of zero-coupon interest rates (expressed as 
%     percentages).

    % Reference date for the calculations
    dt = datetime(2023, 02, 02);
    
    % Compute the year fractions between the reference date and each given date
    % The third argument (3) specifies the day count convention (30/360 ISDA)
    y_frac = yearfrac(dt, dates, 3);

    % Compute the zero-coupon rates using the formula:
    % zRates = (- log(discount factor) / year fraction) * 100
    % The multiplication by 100 converts the rate to percentage form
    zRates = (- log(discounts) ./ y_frac) * 100;

end
