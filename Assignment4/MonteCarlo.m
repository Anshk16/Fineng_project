function C_price = MonteCarlo(montecarlo_simulations, delta, sigma, eta, discount, x_moneyness, k, F0)


K = F0 ./ exp(x_moneyness);  % strike price with log-moneyness and forward price 
laplace_exponent = @(w) - delta / k * log(1 + k * w * sigma ^ 2); % laplace exponent with alpha equal to 0
rng(30)

% Computation of the two i.d. random variables 
alpha = delta/k;
beta = alpha;

G = gamrnd(alpha, 1/beta, montecarlo_simulations, 1); % Gamma distributed random variables ( (montecarlo_simulations,1) matrix)
% the explanation behind the choice of a gamma(alpha,beta) with alpha = beta = delta/k
% is that we need a positive r.v. with unitary mean and variance equal to
% vega/delta

g = randn(montecarlo_simulations,1); % standard normal

%The dynamics of ft up to a variation delta of the time (our delta) can be expressed as
ft = sqrt(delta) * sigma * sqrt(G).*g - (1/2 + eta) * delta * sigma^2 * G - laplace_exponent(eta);

% Using the Lèvy model for the return (having F0 and ft)
Ft = exp(ft) .* F0; 

C_price =  discount * mean(max(Ft - K, 0), 1);


