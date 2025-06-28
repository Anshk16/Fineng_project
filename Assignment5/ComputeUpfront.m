function Upfront = ComputeUpfront(LiborRates, Strikes, SpotVol, DeltaMoving, DeltaFix, Discounts, SPOL, flag)


LenPayDates = length(SpotVol)+1;

Strike_0y_3y = 4.3 * 1e-2;
Strike_3y_6y = 4.6 * 1e-2;
Strike_6y_10y = 5.2 * 1e-2;

FirstQuarterCoupon = 3*1e-2; %First coupon 


eur3m = LiborRates; %default values, instead there is the if condition



BPV = sum(DeltaMoving(1:LenPayDates).*Discounts(2:LenPayDates+1));

PaymentA = BPV*SPOL + 1 - Discounts(LenPayDates+1);


coupon = zeros(LenPayDates,1);
coupon(1) = Discounts(2)*FirstQuarterCoupon;


for i=2:LenPayDates-1
    
    if i <= 12
        K = Strike_0y_3y;
    elseif 12 < i && i <= 24
        K = Strike_3y_6y;
    else
        K = Strike_6y_10y;
    end

    Vol = spline(Strikes, SpotVol(i,:), K);

    Cap = CapPrice(Discounts(3:i+2), DeltaFix(1:i), DeltaMoving(2:i+1), eur3m(1:i), 1, K, Vol);

    Digital = DigitalPrice(eur3m(i), Strikes, K, SpotVol(i,:), DeltaMoving(i+1), DeltaFix(i), Discounts(i+2), flag);
    
    DeltaD = Discounts(i+1)-Discounts(i+2);
   
    coupon(i) = 0.011*Discounts(i+1) - 0.009 * Digital - Cap + DeltaMoving(i) * DeltaD;

end


Upfront = PaymentA - sum(coupon);

end