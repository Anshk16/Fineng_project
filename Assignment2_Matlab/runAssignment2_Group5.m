% runAssignment2
%  group X, AY20ZZ-20ZZ
% Computes Euribor 3m bootstrap with a single-curve model
%
% This is just code structure: it should be completed & modified (TBM)
%
% to run:
% > runAssignment2_TBM

clear; close all; clc;

%% Settings
formatData='dd/mm/yyyy'; %Pay attention to your computer settings 

%% Read market data
% This fuction works on Windows OS. Pay attention on other OS.

[datesSet, ratesSet] = readExcelData('MktData_CurveBootstrap', formatData);

%% Bootstrap
% dates includes SettlementDate as first date
clc; format long

[dates, discounts] = bootstrap(datesSet, ratesSet); 

%% Compute Zero Rates

zRates = zeroRates(dates, discounts);


%% Plot Results

% Convert dates to datetime format
dates_dt = datetime(dates, 'ConvertFrom', 'datenum');

% Plot discount factors with improved style
figure;
plot(dates_dt, discounts, 'o', 'MarkerSize', 8, 'DisplayName', 'Original Data'); % Original data points
hold on;
plot(dates_dt, discounts, '-', 'LineWidth', 1.5, 'DisplayName', 'Interpolated Curve'); % Interpolated curve
hold off;

% Customize plot appearance
xlabel('Dates', 'FontSize', 12);
ylabel('Discount Factors', 'FontSize', 12);
title('Discount Factors vs. Dates', 'FontSize', 14);
legend;
grid on;

d = datetime(dates, 'ConvertFrom', 'datenum');
% Plot zero-rates with improved style
figure;
plot(d, zRates, 'o', 'MarkerSize', 8, 'DisplayName', 'Original Data'); % Original data points
hold on;
plot(d, zRates, '-', 'LineWidth', 1.5, 'DisplayName', 'Interpolated Curve'); % Interpolated curve
hold off;

% Customize plot appearance
xlabel('Dates', 'FontSize', 12);
ylabel('Zero Rates', 'FontSize', 12);
title('Zero Rates vs. Dates', 'FontSize', 14);
legend;
grid on;



%% Price of bond
n = 10^8;
price = price_bond(dates, discounts, ratesSet);

tot_price = price*n;

format long g
disp('Bond price:')
disp(tot_price)

%% Point 3
% Datas
setDate = datesSet.settlement;
fixedRate = 0.028175;
n_years = 7;

ratesSet_shifted = rates_shifter(ratesSet);

% Compute discounts with shifted rates using bootstrap
[dates, discounts_DV01] = bootstrap(datesSet, ratesSet_shifted);

fix_dates = fixedLegPaymentDates(setDate, n_years);

[DV01, BPV, DV01_z] = sensSwap(setDate, fix_dates, fixedRate, dates, discounts, discounts_DV01);
disp('DV01:')
disp(DV01)

disp('BPV:')
disp(BPV)

disp('DV01_z:')
disp(DV01_z)

MacD = sensCouponBond(setDate, fix_dates, fixedRate, dates, discounts);
disp('MacD:')
disp(MacD)

%% NPV 
today = datesSet.settlement;
start_date = datetime(2026, 09, 25);
n_years = 20;
c = [1.5e3 6e3];

NPV(1) = compute_NPV(dates, discounts, c(1), today, start_date, n_years);
NPV(2) = compute_NPV(dates, discounts, c(2), today, start_date, n_years);

disp('NPV with 1500 eur starting cash flow:')
disp(NPV(1));
disp('NPV with 6000 eur starting cash flow:');
disp(NPV(2));