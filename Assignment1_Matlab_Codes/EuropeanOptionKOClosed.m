function OptionPriceClosed = EuropeanOptionKOClosed(F0,K,KO,B,T,sigma)

r = -log(B) / T;

c_K = EuropeanOptionClosed(F0,K,B,T,sigma,1);
c_KO = EuropeanOptionClosed(F0,KO,B,T,sigma,1);

d2 = (log(F0*B/KO)+(r-sigma^2/2)*T)/(sigma*sqrt(T));
digital = (KO-K)*B*normcdf(d2);

OptionPriceClosed = c_K - c_KO - digital;

end   