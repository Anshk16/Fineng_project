function [datesCDS, survProbs, intensities] = bootstrapCDS_JT(datesCDS, spreadsCDS, recovery)
    % This function performs bootstrapping of CDS spreads 
    % using the Jarrow-Turnbull (JT) model to estimate survival probabilities
    % and default intensities.
    % 
    % Inputs:
    % - datesCDS: CDS contract maturities
    % - spreadsCDS: CDS spreads in basis points (bps)
    % - recovery: Recovery rate (as a decimal, e.g., 0.4 for 40%)
    %
    % Outputs:
    % - datesCDS: Updated CDS dates
    % - survProbs: Survival probabilities for each CDS date
    % - intensities: Default intensities for each CDS date

    % Define day count convention (Actual/365)
    convention_index_365 = 3;

    % Compute year fractions between settlement date and CDS maturities
    yfrac_365 = yearfrac(datesCDS(1), datesCDS(2:end), convention_index_365);

    % Convert spreads from basis points (bps) to decimal
    bp = 0.0001;
    spreadsCDS = spreadsCDS * bp;

    % Compute default intensities using the Jarrow-Turnbull model
    intensities = spreadsCDS' / (1 - recovery);

    % Compute survival probabilities using the exponential formula
    survProbs = exp(-intensities .* yfrac_365);

    % Display the results
    format long
    disp('-------- bootstrapCDS_JT --------')
    fprintf('%-20s %-20s\n', 'survProbs', 'intensities');  % Header with alignment
    
    % Print values row by row
    for i = 1:length(survProbs)
        fprintf('%-20.12f %-20.12f\n', survProbs(i), intensities(i));
    end
    disp(' ')

end
