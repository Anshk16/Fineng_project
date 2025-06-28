function SpotVol = CalibrationSpotVol(PaymentDates, DeltaFix, DeltaMoving, Discounts, LiborRates, Frequency, FlatVol_Strikes, FlatVol, Maturity, flag)

IBDayCount = 3; % yearfrac Act/365

LenStrikes = length(FlatVol_Strikes);
LenPayDates = length(PaymentDates) - 2;

SpotVol = zeros(LenPayDates-1, LenStrikes); % aligning to calibration pattern

% First iteration

FirstIndex = Frequency * 1 - 1; %we need the first year for the first iteration

SpotVol(1:FirstIndex, :) = FlatVol(1, :) .* ones(length(1:FirstIndex), 1);

% Computing the first Cap Price

Cap1 = arrayfun(@(strike, vol) CapPrice(Discounts(3:5), DeltaFix(1:3), DeltaMoving(2:4), LiborRates(1:3), flag, strike, vol), FlatVol_Strikes, FlatVol(1, :));

% Start the algorithm
for i = 2:Maturity
    idx_cap1 = Frequency * (i-1);
    idx_cap2 = Frequency * i;

    Cap2 = arrayfun(@(strike, vol) CapPrice(Discounts(3:idx_cap2 +1), DeltaFix(1:idx_cap2-1), DeltaMoving(2:idx_cap2), LiborRates(1:idx_cap2-1), flag, strike, vol), FlatVol_Strikes, FlatVol(i,:));
    DeltaC = Cap2 - Cap1;
    Cap1 = Cap2;

    denominator = yearfrac(PaymentDates(idx_cap1), PaymentDates(idx_cap2), IBDayCount);
    Sigma_i = @(LastSigma, ActualPaymentDate) SpotVol(idx_cap1 - 1, :) + (yearfrac(PaymentDates(idx_cap1), ActualPaymentDate, IBDayCount) / denominator) .* (LastSigma - SpotVol(idx_cap1 - 1, :));
    SumCapletOld = @(LastSigma) 0;
    if flag == 1 %LMM
        for j = idx_cap1:idx_cap2-1
            SumCapletNew = @(LastSigma) SumCapletOld(LastSigma) + arrayfun(@(strike, vol) CapletPriceLMM(Discounts(j+2), DeltaFix(j), DeltaMoving(j+1), LiborRates(j), strike,vol), FlatVol_Strikes, Sigma_i(LastSigma, PaymentDates(j + 1)));
            SumCapletOld = SumCapletNew;
        end
    elseif flag == -1 %BMM
        for j = idx_cap1:idx_cap2-1         
            SumCapletNew = @(LastSigma) SumCapletOld(LastSigma) + arrayfun(@(strike, vol) CapletPriceBMM(Discounts(j+2), DeltaFix(j), DeltaMoving(j+1), LiborRates(j), strike, vol), FlatVol_Strikes, Sigma_i(LastSigma, PaymentDates(j + 1)));      
            SumCapletOld = SumCapletNew;
        end
    end
    FunSolve = @(LastSigma) SumCapletNew(LastSigma) - DeltaC;
    Last_Sigma = fsolve(FunSolve, FlatVol(i, :));
    SpotVol(idx_cap2-1, :) = Last_Sigma;

    for j = idx_cap1:idx_cap2-2
        SpotVol(j, :) = Sigma_i(Last_Sigma, PaymentDates(j+1));
    end

end
