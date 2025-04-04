function optionPrice = EuropeanOptionKOCRR(F0,K,KO,B,T,sigma,N)

dt = T/N; % time step
u = exp(sigma*sqrt(dt)); % up factor
d = 1/u; % down factor
q = (1-d)/(u-d); % probability of up movement

r = -log(B) / T;

B_dt = exp(-r*dt);

nodes = F0*u.^(-N:2:N);
payoff = zeros(N+1,1);

for i=1:length(nodes)
    if nodes(i) > KO
        payoff(i) = 0;
    else
        payoff(i) = max(nodes(i)-K,0);
    end
end

for i=N:-1:1
    for j=1:i
        payoff(j) = B_dt*((1-q)*payoff(j)+q*payoff(j+1));
    end
end

optionPrice = B_dt*payoff(1);
end