function approx_sol = MBS_Pricing_KL(Kd, Ku, recovery, I, rho, p, discounts, dates)
    % This function computes an approximate solution for MBS (Mortgage-Backed Securities) pricing
    % using the Large Deviation (KL) approach.
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
    % - approx_sol: Computed MBS price using the KL approximation

    % Define lower and upper boundaries for the loss range
    d = Kd / (1 - recovery);
    u = Ku / (1 - recovery);
    
    % Loss function L(z), constrained between [d, u]
    L = @(z) min(max(z - d, 0), u - d) / (u - d);
    
    % Default probability function P(y) given y
    P = @(y) normcdf((norminv(p) - sqrt(rho) * y) / sqrt(1 - rho));

    % Kullback-Leibler (KL) divergence function
    K = @(z, x) z .* log(z ./ x) + (1 - z) .* log((1 - z) ./ (1 - x));
    
    % Define standard normal probability density function
    phi = @(y) normpdf(y);

    % First coefficient for normalization
    C1 = @(z) sqrt(I ./ (2 * pi * (1 - z) .* z));

    % Compute D(y) before using it
    integrand1 = @(z, y) C1(z) .* exp(-I * K(z, P(y)));
    D = @(y) arrayfun(@(yy) integral(@(z) integrand1(z, yy), 0, 1), y);
    
    % Define normalization factor C(z, y)
    C = @(z, y) C1(z) ./ D(y);
    
    % Integrand for the integral over z
    integrand2 = @(y, z) L(z) .* C(z, y) .* exp(-I .* K(z, P(y)));
    
    % Second integral over z
    integrand3 = @(y) phi(y) .* arrayfun(@(yy) quadgk(@(z) integrand2(yy, z), 0, 1), y);

    % Compute the final result using numerical integration over y
    approx_sol = quadgk(integrand3, -6, 6);

    % Apply discounting
    date = datetime('02-Feb-2026'); % Target date for discounting
    today = datetime('02-Feb-2023'); % Current date
    discounts = interpolation_vector(dates, discounts, date, today);

    % Compute present value
    approx_sol = discounts * (1 - approx_sol);

end
