function Caplet = CapletPriceLMM(Discount,DeltaFix,DeltaMoving,LiborRate, Strike,Vol)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Discount : scalar of the discount B(t0, ti + 1)
% DeltaFix : scalar of the distance between the t0 and ti
% DeltaMoving : scalar of the distance between ti and ti+1
% LiborRate : scalar of the LIBOR rate
% Strike : scalar of the strike
% Vol : scalar of the volatility
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% defining variables d1 and d2

d1_L = log(LiborRate./Strike)./(sqrt(DeltaFix).*Vol) + 1/2*Vol*sqrt(DeltaFix);
d2_L = d1_L - sqrt(DeltaFix)*Vol;

Caplet = DeltaMoving.*Discount.*(LiborRate.*normcdf(d1_L) - Strike*normcdf(d2_L));
end