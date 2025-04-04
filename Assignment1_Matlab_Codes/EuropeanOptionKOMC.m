function optionPrice = EuropeanOptionKOMC(F0, K, KO, B, T, sigma, N)
% European call option price with knock-out (KO) feature using Monte Carlo simulation
% F0: Initial asset price
% K: Strike price
% KO: Knock-out barrier
% B: Discount factor
% T: Time to maturity
% sigma: Volatility
% N: Number of simulations

% Generate N random samples from the standard normal distribution
x = randn(N, 1);

% Simulate asset prices at maturity using vectorized operations
assetPrices = F0 * exp((-0.5 * sigma^2) * T + sigma * sqrt(T) * x);

% Calculate payoffs for the call option with knock-out condition
% Payoff is max(y - K, 0) if y < KO, otherwise 0
payoffs = max(assetPrices - K, 0) .* (assetPrices < KO);

% Compute the option price as the discounted average payoff
optionPrice = B * mean(payoffs);
end