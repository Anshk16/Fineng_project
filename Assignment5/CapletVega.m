function Vega = CapletVega(LiborRate, K, Vol, DeltaFix, DeltaMoving, Discount)


    d1 = (log(LiborRate./K) + 0.5 * Vol^2 * DeltaFix) / (Vol * sqrt(DeltaFix));
    Vega = Discount * DeltaMoving * LiborRate * normpdf(d1) * sqrt(DeltaFix);

end
