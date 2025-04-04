function errAnti = AntitheticERR(F0,K,M,T,sigma)

comp1 = [];

Z = randn(M,1);

comp1 = max(F0*exp(-(sigma^2)/2*T+sigma*sqrt(T).*Z)-K,0);

comp2 = max(F0*exp(-(sigma^2)/2*T-sigma*sqrt(T).*Z)-K,0);
    
comp = (comp1 + comp2)/2;

errAnti = std(comp)/sqrt(M);


