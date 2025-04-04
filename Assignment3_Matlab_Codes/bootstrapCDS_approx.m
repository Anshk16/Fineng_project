function [datesCDS, survProbs, intensities] = bootstrapCDS_approx(datesDF, discounts, datesCDS, spreadsCDS, recovery)
    % This function performs an approximate bootstrap of CDS spreads 
    % to estimate survival probabilities and default intensities.
    % 
    % Inputs:
    % - datesDF: Discount factor dates
    % - discounts: Discount factors corresponding to datesDF
    % - datesCDS: CDS contract maturities
    % - spreadsCDS: CDS spreads in basis points (bps)
    % - recovery: Recovery rate (as a decimal, e.g., 0.4 for 40%)
    %
    % Outputs:
    % - datesCDS: Updated CDS dates
    % - survProbs: Survival probabilities for each CDS date
    % - intensities: Default intensities for each CDS date
    
    % Define settlement date (assumed to be the first element in datesDF)
    settle_date = datesDF(1);
    
    % Interpolate discount factors for CDS maturity dates
    discountsCDS = interpolation_vector(datesDF, discounts, datesCDS(2:end), settle_date);
    
    % Define day count conventions
    convention_index_30_360 = 6; % 30/360 day count convention
    convention_index_365 = 3; % Actual/365 day count convention
    
    % Compute year fractions for CDS tenors using both conventions
    deltas = yearfrac(datesCDS(1:end-1), datesCDS(2:end), convention_index_30_360);
    deltas_365 = yearfrac(datesCDS(1:end-1), datesCDS(2:end), convention_index_365);

    % Initialize survival probabilities
    survProbs = zeros(length(deltas) + 1, 1);
    
    % Convert spreads from basis points (bps) to decimal
    bp = 0.0001;
    spreadsCDS = spreadsCDS * bp;

    % Bootstrap the survival probabilities
    survProbs(1) = 1; % The probability of survival at t=0 is 1
    survProbs(2) = (1 - recovery) / (spreadsCDS(1) * deltas(1) + 1 - recovery);

    for i = 2:length(survProbs)-1
        sum1 = 0;
        % Compute the summation term
        for j = 1:i-1
            sum1 = sum1 + discountsCDS(j) * (survProbs(j) - survProbs(j+1));
        end

        % Compute numerator and denominator for survival probability formula
        N = (1 - recovery) * sum1 + (1 - recovery) * discountsCDS(i) * survProbs(i) ...
            - spreadsCDS(i) * sum(deltas(1:i-1) .* discountsCDS(1:i-1) .* survProbs(2:i));
        D = (1 - recovery) * discountsCDS(i) + spreadsCDS(i) * deltas(i) * discountsCDS(i);
        
        % Compute next survival probability
        survProbs(i+1) = N / D;
    end

    % Compute default intensities from survival probabilities
    intensities = zeros(length(deltas_365), 1);
    for i = 1:length(survProbs)-1
        intensities(i) = -(1 / deltas_365(i)) * log(survProbs(i+1) / survProbs(i));
    end
    
    % Remove the first survival probability to match dimensions
    survProbs = survProbs(2:end);
    
    % Display the results
    format long
    disp('-------- bootstrapCDS_approx --------')
    fprintf('%-20s %-20s\n', 'survProbs', 'intensities');  % Header with alignment
    
    % Print values row by row
    for i = 1:length(survProbs)
        fprintf('%-20.12f %-20.12f\n', survProbs(i), intensities(i));
    end
    disp(' ')

end
