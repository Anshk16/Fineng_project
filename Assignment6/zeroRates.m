function [zRates] = zeroRates(dates,discounts)
zRates=-log(discounts)./yearfrac(dates(1),dates,3);
zRates=zRates*100;
end

