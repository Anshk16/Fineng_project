function [DV01, BPV, DV01_z] = sensSwap(setDate, fixedLegPaymentDates, fixedRate, dates, discounts, discounts_DV01)
%SENSSWAP Computes DV01, BPV, and shifted DV01 for an interest rate swap.
%
%   [DV01, BPV, DV01_z] = sensSwap(setDate, fixedLegPaymentDates, fixedRate, 
%   dates, discounts, discounts_DV01) calculates the interest rate swap 
%   sensitivity metrics: DV01 (Dollar Value of 1 basis point), BPV (Basis 
%   Point Value), and DV01_z (DV01 computed via zero rates).
%
%   Inputs:
%   - setDate: Valuation date.
%   - fixedLegPaymentDates: Vector of payment dates for the fixed leg.
%   - fixedRate: Fixed leg interest rate.
%   - dates: Vector of discount factor dates.
%   - discounts: Discount factors for the given dates.
%   - discounts_DV01: Discount factors shifted by 1 basis point.
%
%   Outputs:
%   - DV01: Sensitivity of swap value to a 1 basis point shift in discount factors.
%   - BPV: Present value of a 1 basis point shift in fixed leg payments.
%   - DV01_z: Sensitivity of swap value to a 1 basis point shift in zero rates.

    % Compute discount factors for fixed leg payment dates using interpolation
    fix_discounts = interpolation_vector(dates, discounts, fixedLegPaymentDates, setDate);
    fix_discounts_DV01 = interpolation_vector(dates, discounts_DV01, fixedLegPaymentDates, setDate);

    % Plot discount factor interpolation for validation
    figure
    hold on
    grid on
    plot(dates, discounts, 'b-', 'LineWidth', 2); % Original discounts
    plot(dates, discounts_DV01, 'g--', 'LineWidth', 2); % Shifted discounts
    plot(datenum(fixedLegPaymentDates), fix_discounts, 'r.', 'MarkerSize', 12); % Interpolated discounts
    plot(datenum(fixedLegPaymentDates), fix_discounts_DV01, 'm.', 'MarkerSize', 12); % Interpolated shifted discounts
    hold off
    legend('Original Discounts', 'Discounts DV01', 'Interpolated Discounts', 'Interpolated Discounts DV01', 'Location', 'Best');
    xlabel('Dates');
    ylabel('Discount Factors');
    title('Interpolation Check');
    datetick('x', 'yyyy-mm-dd', 'keeplimits');

    % Compute year fractions for fixed leg payments
    yfrac_fixed_6 = yearfrac(setDate, fixedLegPaymentDates, 6);
    yfrac_fixed_3 = yearfrac(setDate, fixedLegPaymentDates, 3);
    yfrac_fixed_comp = [yfrac_fixed_6(1); yfrac_fixed_6(2:end) - yfrac_fixed_6(1:end-1)];

    % Compute Net Present Value (NPV) for floating and fixed legs
    NPV_float = 1 - fix_discounts(end);
    NPV_fixed = fixedRate * sum(yfrac_fixed_comp .* fix_discounts);
    NPV_float_DV01 = 1 - fix_discounts_DV01(end);
    NPV_fixed_DV01 = fixedRate * sum(yfrac_fixed_comp .* fix_discounts_DV01);
    NPV = NPV_float - NPV_fixed;
    NPV_DV01 = NPV_float_DV01 - NPV_fixed_DV01;

    % Compute DV01 as the absolute difference in NPV after shifting discount factors
    DV01 = abs(NPV_DV01 - NPV);

    % Compute DV01 using zero rates (DV01_z)
    bp = 0.0001; % 1 basis point shift
    zRates = zeroRates(fixedLegPaymentDates, fix_discounts);
    zRates_shift = zRates / 100 + bp;
    discounts_shift = exp(-zRates_shift .* yfrac_fixed_3);
    NPV_z_float = 1 - discounts_shift(end);
    NPV_z_fixed = fixedRate * sum(yfrac_fixed_comp .* discounts_shift);
    NPV_z = NPV_z_float - NPV_z_fixed;
    DV01_z = abs(NPV_z - NPV);

    % Compute BPV as the sum of weighted discount factors multiplied by 1 basis point
    BPV = sum(yfrac_fixed_comp .* fix_discounts) * bp;

end
