function rates_shifted = ShiftRates(Dates, Rates, Today, Shift)

ii = NaN;
%depos
for jj = 1:length(Dates.depos)
    if Dates.depos(jj) == Today
        ii = jj;
        break
    end
end

if ~isnan(ii)
    rates_shifted = Rates;
    rates_shifted.depos(ii) = Rates.depos(ii) + Shift;
    return
end
%futures
for jj = 1:length(Dates.futures)
    if Dates.futures(jj,2) == Today
        ii = jj;
        break
    end
end

if ~isnan(ii)
    rates_shifted = Rates;
    rates_shifted.futures(ii) = Rates.futures(ii) + Shift;
    return
end
%swaps
for jj = 1:length(Dates.swaps)
    if Dates.swaps(jj) == Today
        ii = jj;
        break
    end
end

if ~isnan(ii)
    rates_shifted = Rates;
    rates_shifted.swaps(ii) = Rates.swaps(ii) + Shift;
    return
end

end

