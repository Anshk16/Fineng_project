function vega = CapVega(LiborRates, Strikes, Vol, DeltaFix, DeltaMoving, Discounts,SwapRate)


vega = 0;

for i=1:length(DeltaFix)-3
    
    IntVol = spline(Strikes, Vol(i,:), SwapRate);

    d1 = log(LiborRates(i)/SwapRate)/(IntVol*sqrt(DeltaFix(i))+0.5*IntVol*sqrt(DeltaFix(i)));

    vega = vega + LiborRates(i) * DeltaMoving(i+1) * Discounts(i+2) * normpdf(d1) * sqrt(DeltaFix(i)); 

end

end