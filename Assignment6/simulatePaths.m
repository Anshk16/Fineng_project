function S = simulatePaths(Model, S0, F0, Delta, Nsteps, Nsim)



F = zeros(Nsim, Nsteps+1);
F(:,1) = F0;

rng(30)

for i = 1:Nsteps

    if strcmp(Model.name, 'VG')
        LaplaceExponent  = @(w) - Delta(i) / Model.k * log(1 + Model.k * w * Model.sigma ^ 2); % VG Laplace exponent
    elseif strcmp(Model.name, 'NIG')
        LaplaceExponent  = @(w)  Delta(i) / Model.k * (1 - sqrt(1 + 2 * Model.k * w * Model.sigma ^ 2)); % NIG Laplace exponent
    else
        disp('error')
        return
    end

    alpha = Delta(i)/Model.k;
    beta = alpha;
    deltaG = gamrnd(alpha, 1/beta, Nsim,1);  % Gamma distributed
    g = randn(Nsim,1);                       % Normal distributed
    ft = sqrt(Delta(i)) * Model.sigma * sqrt(deltaG).*g - (1/2 + Model.eta) * Delta(i) * Model.sigma^2 * deltaG - LaplaceExponent(Model.eta);

    F(:,i+1) = F(:,i) .* exp(ft);
end

S = zeros(Nsim, Nsteps+1);
S(:,1) = S0;
S(:,2:end) = F(:,2:end);

end

