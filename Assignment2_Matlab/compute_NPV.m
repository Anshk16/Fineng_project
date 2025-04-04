function NPV = compute_NPV(dates, discounts, c, today, start_date, n_years)
%COMPUTE_NPV Computes the Net Present Value (NPV) of a cash flow series.
%
%   NPV = compute_NPV(dates, discounts, c, today, start_date, n_years)
%   calculates the NPV of a series of monthly cash flows over a given
%   number of years, using discount factors obtained via interpolation.
%
%   Inputs:
%   - dates: Vector of dates corresponding to the given discount factors.
%   - discounts: Discount factors associated with the given dates.
%   - c: Initial cash flow amount.
%   - today: Current valuation date.
%   - start_date: Start date of the cash flows.
%   - n_years: Number of years for which cash flows are generated.
%
%   Output:
%   - NPV: The computed Net Present Value.

%% Build cash flow dates vector

% Preallocate the cash flow date vector
cf_dates = NaT(12 * n_years, 1);
cf_dates(1) = start_date;

% Generate monthly cash flow dates and adjust for weekends
for j = 0:(n_years * 12 - 1)
    cf_dates(j+1) = start_date + calmonths(j);
    
    % Adjust if the date falls on a weekend
    if weekday(cf_dates(j+1)) == 7
        cf_dates(j+1) = cf_dates(j+1) + days(2);  % Saturday → Monday
    elseif weekday(cf_dates(j+1)) == 1
        cf_dates(j+1) = cf_dates(j+1) + days(1);  % Sunday → Monday
    end
end

%% Build cash flow vector

% Preallocate the cash flow values vector
cf_values = zeros(1, n_years * 12);

% Assign cash flow values with growth over time
for j = 0:(n_years-1)
    cf_values((j*12+1):(j*12+12)) = c * (1.05^j);
end

%% Build discount factors vector 

% Preallocate the discount factors vector
cf_discounts = zeros(1, n_years * 12);
cf_dates_num = datenum(cf_dates);  % Convert datetime to numeric format

% Perform interpolation for discount factors
for i = 1:(n_years * 12)
    
    % Initialize variables for previous and next discount factor dates
    previous_date = -1;  
    next_date = -1;  
    previous_B = -1;
    next_B = -1;

    % Scan the discount factor dates to find the surrounding points
    for j = 1:length(dates)
        if dates(j) < cf_dates_num(i)
            previous_date = dates(j);  % Update previous date
            previous_B = discounts(j); % Update previous discount factor
        elseif dates(j) > cf_dates_num(i) && next_date == -1
            next_date = dates(j);  % Find the first next date and exit loop
            next_B = discounts(j); % Get the next discount factor
            break;
        end
    end

    % Handle cases where interpolation is not possible
    if previous_date == -1 || next_date == -1
        warning(['Interpolation impossible for cf_date = ', num2str(cf_dates_num(i))]);
        cf_discounts(i) = NaN; % Avoid interpolation errors
    else
        % Perform linear interpolation
        cf_discounts(i) = interpolation(previous_date, next_date, previous_B, next_B, cf_dates_num(i), 1, today);
    end
end

%% Graphical check for interpolation accuracy

figure;
hold on;
grid on;
plot(dates, discounts, 'LineWidth', 1.5, 'DisplayName', 'Original Discounts');
plot(cf_dates_num, cf_discounts, '.', 'MarkerSize', 8, 'DisplayName', 'Interpolated Discounts');
hold off;

% Customize plot appearance
xlabel('Dates', 'FontSize', 12);
ylabel('Discount Factors', 'FontSize', 12);
title('Discount Factor Interpolation', 'FontSize', 14);
legend;
grid on;

%% Computation of NPV

% Compute Net Present Value (NPV) as the sum of discounted cash flows
NPV = sum(cf_values .* cf_discounts);

end
