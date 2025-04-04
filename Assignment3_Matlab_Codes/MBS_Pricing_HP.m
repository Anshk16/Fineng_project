function exact_sol = MBS_Pricing_HP(Kd, Ku, recovery, I, rho, p, discounts, dates)
    % This function computes the exact solution for MBS (Mortgage-Backed Securities) pricing
    % using the Hull-White (HP) approach with a binomial expansion.
    %
    % Inputs:
    % - Kd: Lower bound of the loss range
    % - Ku: Upper bound of the loss range
    % - recovery: Recovery rate (as a decimal, e.g., 0.4 for 40%)
    % - I: Number of discretization steps in the binomial model
    % - rho: Correlation parameter
    % - p: Default probability for a single entity
    % - discounts: Discount factors for pricing
    % - dates: Corresponding dates for discount factors
    %
    % Output:
    % - exact_sol: Computed MBS price using the Hull-White exact approach

    % Define function P(y) which represents the default probability given y
    P = @(y) normcdf((norminv(p) - sqrt(rho) * y) / sqrt(1 - rho));
    
    % Define the standard normal probability density function
    phi = @(y) normpdf(y);
    
    % Define lower and upper boundaries for the loss range
    d = Kd / (1 - recovery);
    u = Ku / (1 - recovery);

    % Loss function L(z), constrained between [d, u]
    L = @(z) min(max(z - d, 0), u - d) / (u - d);

    % Initialize summation variable
    tot_sum = 0;

    % Compute the expected loss through binomial expansion
    for m = 0:I
        % Define the function to be integrated over the normal distribution
        integrand = @(y) phi(y) .* nchoosek(I, m) .* (P(y).^m) .* (1 - P(y)).^(I - m);
        
        % Compute the integral numerically over the standard normal range [-6,6]
        pd = quadgk(integrand, -6, 6); 
        
        % Accumulate the weighted loss values
        tot_sum = tot_sum + L(m / I) * pd;
    end

    % Compute the final solution
    exact_sol = tot_sum;

    % Interpolate discount factor for the specific date range
    date = datetime('02-Feb-2026'); % Target date for discounting
    today = datetime('02-Feb-2023'); % Current date
    discounts = interpolation_vector(dates, discounts, date, today);

    % Apply discounting to obtain present value
    exact_sol = discounts * (1 - exact_sol);

end
