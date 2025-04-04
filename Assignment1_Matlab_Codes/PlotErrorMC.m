function [M,errMC] = PlotErrorMC(F0,K,B,TTM,sigma)

errMC = [];
i = [1:20];
M = 2.^i;
for j=M
    comp = [];
    for i = 1:j
        comp = [comp, max(F0*exp(-(sigma^2)/2*TTM+sigma*sqrt(TTM)*randn)-K,0)];
    end
    errMC = [errMC, std(comp)/sqrt(j)];
end


