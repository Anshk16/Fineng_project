function Cap = CapPrice(Discounts, DeltaFix, DeltaMoving, LiborRate, flag, Strike, Vol)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Discount : vector of the discounts B(t0, ti + 1)
% DeltaFix : vector of the distances between the t0 and ti
% DeltaMoving : vector of the distances between ti and ti+1
% LiborRate : vector of the LIBOR rates
% Strike : scalar of the strike
% Vol : scalar of the flat volatility
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

 if flag == 1 %LMM
    Caplet = CapletPriceLMM(Discounts,DeltaFix,DeltaMoving,LiborRate, Strike,Vol);
 elseif flag == -1 %BMM
    Caplet = CapletPriceBMM(Discounts,DeltaFix,DeltaMoving,LiborRate, Strike,Vol);
 else
     error('Choose the Market Model')
 end

Cap=sum(Caplet);

end