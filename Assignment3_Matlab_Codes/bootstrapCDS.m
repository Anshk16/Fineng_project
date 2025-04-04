function [datesCDS, survProbs, intensities] = bootstrapCDS(datesDF, discounts, datesCDS, spreadsCDS, flag, recovery)
    % This function performs a bootstrap of CDS (Credit Default Swap) spreads 
    % to obtain survival probabilities and default intensities.
    %
    % Inputs:
    % - datesDF: Discount factor dates
    % - discounts: Discount factors corresponding to datesDF
    % - datesCDS: CDS contract maturities
    % - spreadsCDS: CDS spreads in basis points
    % - flag: Specifies the type of bootstrap method
    %   - 1: Approximate method
    %   - 2: Exact method
    %   - 3: Jarrow-Turnbull (JT) model
    % - recovery: Recovery rate (as a decimal, e.g., 0.4 for 40%)
    %
    % Outputs:
    % - datesCDS: Updated CDS dates
    % - survProbs: Survival probabilities for each CDS date
    % - intensities: Default intensities for each CDS date

    % Choose the appropriate bootstrapping method based on the flag
    if flag == 1
        % Approximate bootstrapping method
        [datesCDS, survProbs, intensities] = bootstrapCDS_approx(datesDF, discounts, datesCDS, spreadsCDS, recovery);
    elseif flag == 2
        % Exact bootstrapping method
        [datesCDS, survProbs, intensities] = bootstrapCDS_exact(datesDF, discounts, datesCDS, spreadsCDS, recovery);
    elseif flag == 3
        % Jarrow-Turnbull (JT) model bootstrapping
        [datesCDS, survProbs, intensities] = bootstrapCDS_JT(datesCDS, spreadsCDS, recovery);
    else
        % Invalid flag: Display error message and exit function
        disp('Invalid flag. Please use 1 (Approximate), 2 (Exact), or 3 (JT).');
        return;
    end
end
