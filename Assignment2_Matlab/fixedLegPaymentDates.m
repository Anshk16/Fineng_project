function cf_dates = fixedLegPaymentDates(start_date, n_years)
%FIXEDLEGPAYMENTDATES Generates a series of annual fixed leg payment dates.
%
%   cf_dates = fixedLegPaymentDates(start_date, n_years) generates an array
%   of yearly payment dates, adjusted for weekends to ensure they fall on 
%   business days (Monday-Friday).
%
%   Inputs:
%   - start_date: The initial payment date (datetime or numeric format).
%   - n_years: Number of years for which payment dates should be generated.
%
%   Output:
%   - cf_dates: A column vector of adjusted yearly payment dates.

    % Convert start_date to datetime if it's in numeric format
    if isnumeric(start_date)
        start_date = datetime(start_date, 'ConvertFrom', 'datenum'); 
    end

    num_dates = n_years + 1;  % Number of dates including the first one

    % Preallocate the datetime array
    cf_dates = NaT(num_dates, 1);

    % Generate yearly payment dates and adjust for weekends
    for j = 1:num_dates
        newDate = start_date + calyears(j - 1); % Add years incrementally
        
        % Adjust if the date falls on a weekend
        dayOfWeek = weekday(newDate);
        if dayOfWeek == 7  % Saturday → Move to Monday
            newDate = newDate + days(2);
        elseif dayOfWeek == 1  % Sunday → Move to Monday
            newDate = newDate + days(1);
        end
        
        % Store the adjusted date
        cf_dates(j) = newDate;
    end

    % Remove the first date to match the expected output format
    cf_dates = cf_dates(2:end);

end
