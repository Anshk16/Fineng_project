function [dates, discounts] = bootstrap(datesSet, ratesSet)
%% Discount factors of Depos
% Libor rate
L_depos = mean(ratesSet.depos, 2);
today = datesSet.settlement;
y_frac_depos = yearfrac(today, datesSet.depos, 2); % mod = 2 for Act/360
% y_frac_depos_365 = yearfrac(today, datesSet.depos, 3);

discounts_depos = 1./(1 + y_frac_depos .*L_depos);
disp('Depos discounts:');
disp(discounts_depos(1:4));

%% Discount factors of Futures

y_frac_futurs = yearfrac(datesSet.futures(:,1), datesSet.futures(:,2), 2); % year fraction between sattle date and expiry date
mid_rates_futures = mean(ratesSet.futures, 2);

fwd_disc = 1./(1 + y_frac_futurs.*mid_rates_futures); % compute the fwd discounts

discounts_futures = zeros(7, 1); % preallocate the discounts of the futures we want to use

discounts_futures(1) = interpolation(datesSet.depos(3), datesSet.depos(4), discounts_depos(3), discounts_depos(4), datesSet.futures(1,1), fwd_disc(1), today);
discounts_futures(2) = interpolation(datesSet.depos(4), datesSet.futures(1,2), discounts_depos(4), discounts_futures(1), datesSet.futures(2,1), fwd_disc(2), today);
discounts_futures(3) = interpolation(datesSet.futures(1,2), datesSet.futures(2,2), discounts_futures(1), discounts_futures(2), datesSet.futures(3,1), fwd_disc(3), today);
discounts_futures(4) = discounts_futures(3) * fwd_disc(4);
discounts_futures(5) = discounts_futures(4) * fwd_disc(5);
discounts_futures(6) = interpolation(datesSet.futures(4,2), datesSet.futures(5,2), discounts_futures(4), discounts_futures(5), datesSet.futures(6,1), fwd_disc(6), today);
discounts_futures(7) = interpolation(datesSet.futures(5,2), datesSet.futures(6,2), discounts_futures(5), discounts_futures(6), datesSet.futures(7,1), fwd_disc(7), today);

disp('Futures discounts:');
disp(discounts_futures);

%% Discount factors of Swaps

% dt = datesSet.settlement
dt = datetime(2023, 02, 02);
datesSet_add = Add_dates(dt, 50); % all of the dates from sattle date to 50y 
swap_dates = datenum(datesSet_add); % all of the dates from sattle date to 50y in numbers
y_frac_swaps = yearfrac(today, swap_dates, 6);

% Perform spline interpolation
mid_rates_swaps = mean(ratesSet.swaps, 2);
interpolated_rates = spline(datesSet.swaps, mid_rates_swaps, swap_dates);

discounts_swaps = zeros(length(swap_dates)-1, 1);
discounts_swaps(1) = interpolation(datesSet.futures(3,2), datesSet.futures(4,2), discounts_futures(3), discounts_futures(4), swap_dates(2), 1, today);

for i = 2:(length(swap_dates)-1)
    x = sum((y_frac_swaps(2:i) - y_frac_swaps(1:i-1)) .* discounts_swaps(1:i-1));
    discounts_swaps(i) = (1 - interpolated_rates(i+1)*x)/(1 + (y_frac_swaps(i) - y_frac_swaps(i-1)) * interpolated_rates(i+1));
end

disp('Swaps discounts:');
disp(discounts_swaps);

%% Aggregation

dates = [today; datesSet.depos(1:4); datesSet.futures(1:7,2); swap_dates(3:end)]; % to add: datesSet_add
discounts = [1; discounts_depos(1:4); discounts_futures; discounts_swaps(2:end)]; % to add: discounts_swaps

% Ordina le date in ordine cronologico e ottieni gli indici di ordinamento
[dates_sorted, idx] = sort(dates);
% Riordina anche i discount usando gli stessi indici
discounts_sorted = discounts(idx);

dates = dates_sorted;
discounts = discounts_sorted;

end