function Price = DigitalPrice(LiborRate, Strikes, K, Vol, DeltaMoving, DeltaFix, Discount, flag)
    
NewVol = spline(Strikes, Vol, K);

d1_L = log(LiborRate./K)./(sqrt(DeltaFix).*NewVol) + 1/2*NewVol.*sqrt(DeltaFix);
d2_L = d1_L - sqrt(DeltaFix).* NewVol;



if flag == "Black"

    Price = Discount*normcdf(d2_L);

elseif flag == "Digital"
    Index = find(Strikes(1:end-1) <= K  & Strikes(2:end) >= K,1);
    if isempty(Index)
        error('No Digital Risk');
    end

    NewSigma = (Vol(Index+1)-Vol(Index))/(Strikes(Index+1)-Strikes(Index));

    Price = Discount*normcdf(d2_L) - NewSigma*CapletVega(LiborRate, K, NewVol, DeltaFix, DeltaMoving, Discount);
end

end