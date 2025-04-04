function disc_interp = interpolation_vector(dates, discounts, dates_interp, today)
%INTERPOLATION_VECTOR Interpolates discount factors for a set of dates.
%
%   disc_interp = interpolation_vector(dates, discounts, dates_interp, today)
%   performs linear interpolation of discount factors for specified dates, 
%   ensuring consistent date formats and handling edge cases where interpolation
%   is not possible.
%
%   Inputs:
%   - dates: Vector of known dates corresponding to discount factors.
%   - discounts: Vector of discount factors associated with known dates.
%   - dates_interp: Vector of dates for which discount factors need to be interpolated.
%   - today: Valuation date used in year fraction calculations.
%
%   Output:
%   - disc_interp: Vector of interpolated discount factors.

    % Ensure all date inputs are in numeric format
    if isdatetime(dates)
        dates = datenum(dates);
    end
    if isdatetime(dates_interp)
        dates_interp = datenum(dates_interp);
    end
    if isdatetime(today)
        today = datenum(today);
    end

    % Number of interpolation dates
    num_interp = length(dates_interp);

    % Preallocate the interpolated discount factors vector
    disc_interp = zeros(num_interp, 1);

    % Iterate through each interpolation date
    for i = 1:num_interp
        interp_date = dates_interp(i);

        % Find the position of the previous and next date in the known dates vector
        idx_prev = find(dates <= interp_date, 1, 'last');  % Last date before interp_date
        idx_next = find(dates >= interp_date, 1, 'first'); % First date after interp_date

        % If the interpolation date exactly matches a known date, use the known discount factor
        if ~isempty(idx_prev) && dates(idx_prev) == interp_date
            disc_interp(i) = discounts(idx_prev);
        else
            % Perform interpolation only if both bounding dates exist
            if ~isempty(idx_prev) && ~isempty(idx_next)
                start_date = dates(idx_prev);
                end_date = dates(idx_next);
                start_B = discounts(idx_prev);
                end_B = discounts(idx_next);

                % Compute interpolated discount factor using the interpolation function
                disc_interp(i) = interpolation(start_date, end_date, start_B, end_B, interp_date, 1, today);
            else
                % Assign NaN if the date is out of range (extrapolation not allowed)
                disc_interp(i) = NaN;
            end
        end
    end
end
