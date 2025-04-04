function optionPrice=EuropeanOptionCRR(F0,K,B,T,sigma,N,flag)
%European option price with CRR formula
%

dt = T/N;
dx = sigma*sqrt(dt);
u = exp(dx);
d = 1/u;
q = (1-d)/(u-d);
s = [];
for i=0:N
    s = [s F0*u^(N-2*i)];
end

c = max(s - K,0);

for i=[1:N]
   for j=1:(N-i+1)
        appo = c;
        c(j) = (q*appo(j)+(1-q)*appo(j+1));

   end
end
optionPrice = c(1)*B;