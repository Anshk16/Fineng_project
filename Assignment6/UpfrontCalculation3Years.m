function Upfront = UpfrontCalculation3Years(S_Paths, Strike, Discounts, SPOL, Delta, DeltaFloat, FloatingDiscounts)

S1 = S_Paths(:,2); % after 1 year
S2 = S_Paths(:,3); % after 2 years

P1 = mean(S1 < Strike);
P2 = mean((S1 > Strike & S2 < Strike));
P3 = mean((S1 > Strike & S2 > Strike));


Coupon1 = Delta(1) * P1 * 0.06;
Coupon2 = Delta(2) * P2 * 0.06;
Coupon3 = Delta(3) * P3 * 0.02;

% Basis Point Value (BPV) on floating payment dates
BPV = sum(DeltaFloat(1:4).*FloatingDiscounts(2:5) + P2 * DeltaFloat(5:8).*FloatingDiscounts(6:9) ...
          + P3 * DeltaFloat(9:end).*FloatingDiscounts(10:end));


Upfront = 1 - Discounts(4) * (1 + Coupon3) + SPOL * BPV - Coupon1 * Discounts(2) - Coupon2 * Discounts(3);

end