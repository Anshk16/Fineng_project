function S = simulateBPaths(S0, F0, Delta, Nsteps, Nsim, Discounts, Sigma, DividendYield)

F = zeros(Nsim, Nsteps+1);
F(:,1) = F0;

ForwardDiscounts_1y_2y = Discounts(end)./ Discounts(end-1);
ForwardZR_1y_2y = -log(ForwardDiscounts_1y_2y)./Delta(2);

rng(30)
g = randn(Nsim,1);

F(:,2) = F0 .* exp(  - 0.5 * Sigma^2 * Delta(1) + Sigma * sqrt(Delta(1)) .* g);
F02 = F(:,2) .* (exp((ForwardZR_1y_2y - DividendYield) * Delta(2)));
F(:,3) = F02 .* exp(  - 0.5 * Sigma^2 * Delta(2) + Sigma * sqrt(Delta(2)) .* g);

S = zeros(Nsim, Nsteps+1);
S(:,1) = S0;
S(:,2:end) = F(:,2:end);

end


