function MacD = sensCouponBond(setDate, couponPaymentDates, fixedRate, dates, discounts)
%SENSCOUPONBOND Computes the Macaulay Duration of a coupon bond.
%
%   MacD = sensCouponBond(setDate, couponPaymentDates, fixedRate, dates, discounts)
%   calculates the Macaulay Duration of a coupon bond given the payment dates, 
%   fixed rate, discount factors, and valuation date.
%
%   Inputs:
%   - setDate: Valuation date (datetime or numeric format).
%   - couponPaymentDates: Vector of future coupon payment dates.
%   - fixedRate: Coupon rate of the bond.
%   - dates: Vector of dates corresponding to the given discount factors.
%   - discounts: Discount factors associated with the given dates.
%
%   Output:
%   - MacD: Macaulay Duration of the bond.

    % Ensure both dates are in the same format
    if isdatetime(setDate)
        setDate = datenum(setDate); % Convert setDate to numeric format
    elseif isnumeric(couponPaymentDates)
        couponPaymentDates = datetime(couponPaymentDates, 'ConvertFrom', 'datenum'); % Convert to datetime
    end
    % Now both setDate and couponPaymentDates are in a consistent format

    % Compute year fractions between setDate and coupon payment dates
    yfrac = yearfrac(setDate, couponPaymentDates, 6);

    % Interpolate discount factors for coupon payment dates
    cf_discounts = interpolation_vector(dates, discounts, couponPaymentDates, setDate);

    % Compute Macaulay Duration
    N = sum(fixedRate * yfrac .* cf_discounts) + yfrac(end) * cf_discounts(end);
    D = sum(fixedRate * cf_discounts) + cf_discounts(end);
    MacD = N / D;

end
