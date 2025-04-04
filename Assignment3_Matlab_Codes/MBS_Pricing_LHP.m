function LHP_sol = MBS_Pricing_LHP(Kd, Ku, recovery, I, rho, p, discounts, dates)
    % This function computes the LHP (Large Homogeneous Portfolio) approximation 
    % for MBS (Mortgage-Backed Securities) pricing.
    %
    % Inputs:
    % - Kd: Lower bound of the loss range
    % - Ku: Upper bound of the loss range
    % - recovery: Recovery rate (as a decimal, e.g., 0.4 for 40%)
    % - I: Number of discretization steps in the binomial model (not used in LHP)
    % - rho: Correlation parameter
    % - p: Default probability for a single entity
    % - discounts: Discount factors for pricing
    % - dates: Corresponding dates for discount factors
    %
    % Output:
    % - LHP_sol: Computed MBS price using the LHP approximation

    % Define lower and upper boundaries for the loss range
    d = Kd / (1 - recovery);
    u = Ku / (1 - recovery);
    
    % Loss function L(z), constrained between [d, u]
    L = @(z) min(max(z - d, 0), u - d) / (u - d);

    % Inverse function of P(z), the probability transformation
    inv_P = @(z) (norminv(p) - sqrt(1 - rho) .* norminv(z)) ./ sqrt(rho);
    
    % Derivative of the transformation P(z)
    der_P = @(z) 1 ./ normpdf(norminv(z)) .* sqrt((1 - rho) / rho);
    
    % Integrand function for numerical integration
    integrand = @(z) normpdf(inv_P(z)) .* der_P(z);
    
    % Final integrand including the loss function
    integrand1 = @(z) L(z) .* integrand(z);
    
    % Compute the integral over [0,1] to estimate expected loss
    LHP_sol = integral(integrand1, 0, 1);

    % Apply discounting
    date = datetime('02-Feb-2026'); % Target date for discounting
    today = datetime('02-Feb-2023'); % Current date
    discounts = interpolation_vector(dates, discounts, date, today);

    % Compute present value
    LHP_sol = discounts * (1 - LHP_sol);

end
