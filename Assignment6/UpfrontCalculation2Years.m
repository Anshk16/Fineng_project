function Upfront = UpfrontCalculation2Years(S_Paths, Strike, Discounts, SPOL, Delta, DeltaFloat, FloatingDiscounts)

S1 = S_Paths(:,2); % after 1 year

P1 = mean(S1 < Strike);
P2 = mean(S1 > Strike);

Coupon1 = Delta(1) * P1 * 0.06;
Coupon2 = Delta(2) * P2 * 0.02;

% Basis Point Value (BPV) on floating payment dates
BPV = sum(DeltaFloat(1:4).*FloatingDiscounts(2:5) + P2 * DeltaFloat(5:end).*FloatingDiscounts(6:end));


Upfront = 1 - Discounts(3) * (1 + Coupon2) + SPOL * BPV - Coupon1 * Discounts(2);

end