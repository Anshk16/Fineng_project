function optionPrice = EuropeanOptionMC(F0, K, B, T, sigma, N, flag)
% European option price with Monte Carlo simulation
% F0: Initial asset price
% K: Strike price
% B: Discount factor
% T: Time to maturity
% sigma: Volatility
% N: Number of simulations
% flag: 1 for call option, -1 for put option (not used in this implementation)

% Generate N random samples from the standard normal distribution
x = randn(N, 1);

% Simulate asset prices at maturity using vectorized operations
assetPrices = F0 * exp((-0.5 * sigma^2) * T + sigma * sqrt(T) * x);

% Calculate payoffs for the call option (vectorized)
payoffs = max(assetPrices - K, 0);

% Compute the option price as the discounted average payoff
optionPrice = B * mean(payoffs);
end