function s_asw = AssetSwapSpread(dates, discounts, p_clean, r, issue_date, n_years)
    % This function calculates the Asset Swap Spread (ASW).
    % It determines the spread that equates the present value of fixed
    % cash flows with the floating cash flows.

    formatData = 'dd/mm/yyyy'; % Pay attention to your computer settings

    % Define the settlement date (assumed to be the first date in 'dates')
    settle_date = dates(1);

    % Generate cash flow dates for the fixed leg
    cf_dates = Add_dates_pre(issue_date, n_years);
    
    % Generate floating leg cash flow dates (quarterly payments)
    num_periods = (n_years - 1) * 4; % Each year has 4 quarters
    cf_float_dates = cf_dates(2) + calmonths(0:3:num_periods*3); % Add months in steps of 3
    cf_float_dates = datenum(cf_float_dates);
    cf_float_dates = [dates(1); cf_float_dates'];
    cf_float_dates = datetime(cf_float_dates, 'ConvertFrom', 'datenum'); 
    cf_float_dates = cf_float_dates(2:end); % Remove first date as it is redundant

    % Adjust floating leg cash flow dates to avoid weekends
    for i = 1:length(cf_float_dates)
        if weekday(cf_float_dates(i)) == 1 % If Sunday
            cf_float_dates(i) = cf_float_dates(i) - 2; % Move to Friday
        elseif weekday(cf_float_dates(i)) == 7 % If Saturday
            cf_float_dates(i) = cf_float_dates(i) - 1; % Move to Friday
        end
    end

    % Adjust for Good Friday (March 29, 2024)
    cf_float_dates(5) = cf_float_dates(5) - 1; 

    % Adjust for Good Friday (March 29, 2024) in fixed leg dates
    cf_dates(3) = cf_dates(3) - 1; 

    % Define day count convention index
    convention_index = 2;

    % Compute dirty price (if needed)
    % p_dirty = p_clean + A ??? (accrued interest not considered here)
    p_dirty = p_clean;

    % Compute discount factors for fixed and floating legs
    cf_discounts = interpolation_vector(dates, discounts, cf_dates(2:end), settle_date);
    cf_discounts_float = interpolation_vector(dates, discounts, cf_float_dates, settle_date);

    % Compute year fractions for fixed leg payments
    deltas = [yearfrac(settle_date, cf_dates(2), convention_index); ...
              yearfrac(cf_dates(2:end-1), cf_dates(3:end), convention_index)];

    % Compute year fractions for floating leg payments
    deltas_float = [yearfrac(settle_date, cf_float_dates(1), convention_index); ...
                    yearfrac(cf_float_dates(1:end-1), cf_float_dates(2:end), convention_index)]; 
    
    % Compute the present value of fixed leg cash flows
    C = r * sum(deltas .* cf_discounts) + cf_discounts(end);

    % Compute BPV (Basis Point Value) for the floating leg
    BPV = sum(deltas_float .* cf_discounts_float);

    % Calculate Asset Swap Spread (ASW)
    s_asw = (C - p_dirty) / BPV;
end
