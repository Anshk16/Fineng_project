function Caplet = CapletPriceBMM(Discount,DeltaFix,DeltaMoving,LiborRate, Strike,Vol)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Discount : scalar of the discount B(t0, ti + 1)
% DeltaFix : scalar of the distance between the t0 and ti
% DeltaMoving : scalar of the distance between ti and ti+1
% LiborRate : scalar of the LIBOR rate
% Strike : scalar of the strike
% Vol : scalar of the volatility
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% defining variables d1 and d2

d1_B = log((1 + DeltaMoving .* LiborRate) ./ (1 + DeltaMoving .* Strike)) ./ (sqrt(DeltaFix) * Vol) + 0.5 * Vol * sqrt(DeltaFix);
d2_B = d1_B - Vol * sqrt(DeltaFix);

Caplet = Discount.*((1 + DeltaMoving .* LiborRate) .* normcdf(d1_B) - (1 + DeltaMoving .* Strike) .* normcdf(d2_B));
end