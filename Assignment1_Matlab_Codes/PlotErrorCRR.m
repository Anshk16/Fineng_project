function [M,errCRR] = PlotErrorCRR(F0,K,B,TTM,sigma)


pricingMode = 1;
esatto = EuropeanOptionPrice(F0,K,B,TTM,sigma,pricingMode);

errCRR = [];
i = [1:10];
M = 2.^i;
for j=M
    pricingMode = 2; % 1 ClosedFormula, 2 CRR, 3 Monte Carlo
    OptionPrice = EuropeanOptionPrice(F0,K,B,TTM,sigma,pricingMode,j);
    errCRR = [errCRR, abs(OptionPrice-esatto)];
end


