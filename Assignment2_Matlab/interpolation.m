function discount = interpolation(start_date, end_date, start_B, end_B, date, fwd, today)
%INTERPOLATION Computes interpolated discount factor for a given date.
%
%   discount = interpolation(start_date, end_date, start_B, end_B, date, fwd, today)
%   performs a linear interpolation of discount factors based on their 
%   corresponding zero rates and applies a forward adjustment if necessary.
%
%   Inputs:
%   - start_date: The earlier date in the interpolation range.
%   - end_date: The later date in the interpolation range.
%   - start_B: Discount factor for start_date.
%   - end_B: Discount factor for end_date.
%   - date: The target date for which the discount factor is computed.
%   - fwd: Forward adjustment factor (usually 1, unless specific adjustment is needed).
%   - today: The valuation date.
%
%   Output:
%   - discount: The interpolated discount factor for the given date.

    % Compute year fractions from today to each relevant date (Act/365 convention)
    y_frac_start = yearfrac(today, start_date, 3); 
    y_frac_end = yearfrac(today, end_date, 3);
    y_frac_date = yearfrac(today, date, 3);

    % Compute zero rates (epsilon) for interpolation
    eps_start = -log(start_B) / y_frac_start;
    eps_end = -log(end_B) / y_frac_end;

    % Perform linear interpolation of zero rates
    if date > start_date && date < end_date
        y = interp1([start_date, end_date], [eps_start, eps_end], date, 'linear');
    else
        y = eps_end; % flat extrapolation
    end

    % Compute interpolated discount factor and apply forward adjustment
    discount = exp(-y * y_frac_date) * fwd;

end
