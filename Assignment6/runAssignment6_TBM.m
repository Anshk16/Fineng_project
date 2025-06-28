% runAssignment6
%  group 9, AY2024-2025
% to run:
% > runAssignment6_TBM

clear all;
close all;
clc;

formatData='dd/mm/yyyy'; 
[datesSet, ratesSet] = readExcelData( "MktData_CurveBootstrap.xls", formatData);

% Bootstrap the discount curve
[dates, discounts] = bootstrap(datesSet, ratesSet);

%% Day count conventions
DepoDayCount = 2; % yearfrac Act/360
IBDayCount = 3; % yearfrac Act/365
SwapDateCount = 6; % yearfrac 30/360 European

%% Case Study 1

% volatility, skewness and kurtosis for the Variance Gamma

VG = struct('name','VG','sigma', 0.1366, 'k', 1.6034, 'eta', 5.4258);

% volatility, skewness and kurtosis for the Normal Inverse Gaussian

NIG = struct('name','NIG','sigma', 0.1064, 'k', 1.2231, 'eta', 12.367);


% parameters

load("cSelect.mat")
Today = dates(1);
Strike = 3200;
Trigger = 6*1e-2;
PrincipalAmount = 1e8;
SPOL = 1.5*1e-2;

Nsim = 100000;     % Number of Monte Carlo paths
Nsteps = 2;        % Annual steps (2 years)


S0 = cSelect.reference;
DividendYield = cSelect.dividend;
K = cSelect.strikes;
Vol = cSelect.surface;

Dates1y = BusinessDates(2, Today, "year", 1, 1); 
Disc1y = zRatesInterp(dates, discounts, Dates1y);
TTM = yearfrac(Dates1y(1),Dates1y(2),IBDayCount);
ZR = -log(Disc1y(2))./TTM;

F0 = S0*exp(TTM*(ZR-DividendYield));

%% Point A-B

% Setting payment dates (floating and fixed leg)
FloatingDates_2y = BusinessDates(9, Today, "month", 3, 1); 
PaymentDates_2y = BusinessDates(3, Today, "year", 1, 1);
ResetDates_2y = [Today; busdate(PaymentDates_2y(2:end) - 2, "previous")];

% Computing discounts
FloatingDiscounts_2y = zRatesInterp(dates, discounts, FloatingDates_2y);
PaymentDiscounts_2y = zRatesInterp(dates, discounts, PaymentDates_2y);
ResetDiscounts_2y = zRatesInterp(dates, discounts, ResetDates_2y);


DeltaFloat_2y = yearfrac(FloatingDates_2y(1:end-1),FloatingDates_2y(2:end),DepoDayCount);
DeltaPrev_2y = yearfrac(ResetDates_2y(1:end-1), ResetDates_2y(2:end), SwapDateCount);
DeltaFoll_2y = yearfrac(Today, PaymentDates_2y(2:end), SwapDateCount);

% Simulating Stoxx50's paths and computing the upfront of the certificate
% 2y
% (VG Model)
S_VG = simulatePaths(VG, S0, F0, DeltaPrev_2y, Nsteps, Nsim);
Upfront_VG_2y = UpfrontCalculation2Years(S_VG, Strike, PaymentDiscounts_2y, SPOL, DeltaFoll_2y, DeltaFloat_2y, FloatingDiscounts_2y);
% For the request b, we choose another model (NIG Model)
S_NIG = simulatePaths(NIG, S0, F0, DeltaPrev_2y, Nsteps, Nsim);
Upfront_NIG_2y = UpfrontCalculation2Years(S_NIG, Strike, PaymentDiscounts_2y, SPOL, DeltaFoll_2y, DeltaFloat_2y, FloatingDiscounts_2y);


% Print results
fprintf('--- Upfront Payment Comparison (2 Years)---\n');
fprintf('Model: Variance Gamma (VG)\n');
fprintf('Upfront Payment: %.4f\n', Upfront_VG_2y);

fprintf('\nModel: Normal Inverse Gaussian (NIG)\n');
fprintf('Upfront Payment: %.4f\n', Upfront_NIG_2y);


%% Point C

% Setting payment dates (floating and fixed leg)
FloatingDates_3y = BusinessDates(13, Today, "month", 3, 1); 
PaymentDates_3y = BusinessDates(4, Today, "year", 1, 1);
ResetDates_3y = [Today; busdate(PaymentDates_3y(2:end) - 2, "previous")];

% Computing discounts
FloatingDiscounts_3y = zRatesInterp(dates, discounts, FloatingDates_3y);
PaymentDiscounts_3y = zRatesInterp(dates, discounts, PaymentDates_3y);


DeltaFloat_3y = yearfrac(FloatingDates_3y(1:end-1),FloatingDates_3y(2:end),DepoDayCount);
DeltaFoll_3y = yearfrac(Today, PaymentDates_3y(2:end), SwapDateCount);


% Computing the upfront of the certificate 3y
% The simulation of the underlying is above
Upfront_VG_3y = UpfrontCalculation3Years(S_VG, Strike, PaymentDiscounts_3y, SPOL, DeltaFoll_3y, DeltaFloat_3y, FloatingDiscounts_3y);
Upfront_NIG_3y = UpfrontCalculation3Years(S_NIG, Strike, PaymentDiscounts_3y, SPOL, DeltaFoll_3y, DeltaFloat_3y, FloatingDiscounts_3y);


% Print results
fprintf('\n--- Upfront Payment Comparison (3 Years)---\n');
fprintf('Model: Variance Gamma (VG)\n');
fprintf('Upfront Payment: %.4f\n', Upfront_VG_3y);

fprintf('\nModel: Normal Inverse Gaussian (NIG)\n');
fprintf('Upfront Payment: %.4f\n', Upfront_NIG_3y);

%% Point E

Delta = yearfrac(ResetDates_2y(1:end-1), ResetDates_2y(2:end), SwapDateCount);
SigmaB = interp1(K, Vol, Strike, 'spline');
S_B = simulateBPaths(S0, F0, Delta, Nsteps, Nsim, ResetDiscounts_2y, SigmaB, DividendYield);
Upfront_B_2y = UpfrontCalculation2Years(S_B, Strike, PaymentDiscounts_2y, SPOL, DeltaFoll_2y, DeltaFloat_2y, FloatingDiscounts_2y);

Error_B_VG = abs(Upfront_B_2y-Upfront_VG_2y);

fprintf('\n--- Upfront Payment Comparison (2 Years)---\n');
fprintf('Model: Variance Gamma (VG)\n');
fprintf('Upfront Payment: %.4f\n', Upfront_VG_2y);

fprintf('\nModel: Black Model\n');
fprintf('Upfront Payment: %.4f\n', Upfront_B_2y);

fprintf('\nError between Variance Gamma Model and Black Model\n');
fprintf('Error: %.6f\n', Error_B_VG);


%% Case study 2

% Parameters
today = dates(1);
Maturity = 10;
Strike = 5e-2;

% According to the contract, these are the dates when we can exercise the
% swaption

% Hull-White parameters
alpha = 0.1;
sigma = 8e-3;

% We cannot consider continuous time, so we take a discretization
% We take 52 weeks per year (times 10y)

TimeSteps = 520; % as higher as possible, to take dt smaller.        

BermudanPrice = BermudanSwaptionTree(today, Maturity, Strike, alpha, sigma, TimeSteps, dates, discounts);

fprintf('The Bermudan Price is: %.4f\n', BermudanPrice * 1e8);

%% point b
EuropeanPrice = EuropeanSwaptionTree(today, Maturity, Strike, alpha, sigma, TimeSteps, dates, discounts);

fprintf('The Bermudan Price is: %.4f\n', EuropeanPrice * 1e8);

%% point d

rates=ratesSet;
ddates=datesSet;

N=1;
PaymentDates = BusinessDates(41, today, "month", 3, 1); 

weights1=[1;3/4;1/2;1/4;0];
weights2=[0;1/4;1/2;3/4;1];
weights3=[3/4;1/2;1/4;0];

rates_bucket_2y=rates;
rates_bucket_2y.depos=rates.depos+1e-4;
rates_bucket_2y.futures=rates.futures+1e-4;
rates_bucket_2y.swaps(1:5,:)=rates.swaps(1:5,:)+weights1.*1e-4;

rates_bucket_6y=rates;
rates_bucket_6y.depos=rates.depos+1e-4;
rates_bucket_6y.futures=rates.futures+1e-4;
rates_bucket_6y.swaps(1:5,:)=rates.swaps(1:5,:)+weights2*1e-4;
rates_bucket_6y.swaps(6:9,:)=rates.swaps(6:9,:)+weights3*1e-4;

rates_bucket_10y=rates;
rates_bucket_10y.depos=rates.depos+1e-4;
rates_bucket_10y.futures=rates.futures+1e-4;
rates_bucket_10y.swaps(5:9,:)=rates.swaps(5:9,:)+weights2*1e-4;
rates_bucket_10y.swaps(10:end,:)=rates.swaps(10:end,:)+1e-4;


[~,discounts_2y]=bootstrap(ddates,rates_bucket_2y);
%BermudanPrice_2y=...
BermudanPrice_2y=BermudanSwaption(today, Maturity, Strike, alpha, sigma, TimeSteps, dates, discounts_2y);
Delta_2y=BermudanPrice_2y-BermudanPrice;

[~,discounts_6y]=bootstrap(ddates,rates_bucket_6y);
BermudanPrice_6y=BermudanSwaption(today, Maturity, Strike, alpha, sigma, TimeSteps, dates, discounts_6y);
Delta_6y=BermudanPrice_6y-BermudanPrice;

[~,discounts_10y]=bootstrap(ddates,rates_bucket_10y);
BermudanPrice_10y=BermudanSwaption(today, Maturity, Strike, alpha, sigma, TimeSteps, dates, discounts_10y);
Delta_10y=BermudanPrice_10y-BermudanPrice;

PaymentDates_bucket_2y=PaymentDates(1:4:9);
DeltaMoving_bucket_2y=yearfrac(PaymentDates_bucket_2y(1:end-1),PaymentDates_bucket_2y(2:end),6);
interpolated_discounts_bucket_2y=zRatesInterp(dates,discounts,PaymentDates_bucket_2y);
BPV_bucket_2y=sum(DeltaMoving_bucket_2y.*interpolated_discounts_bucket_2y(2:end));
SwapRate_bucket_2y=(1-interpolated_discounts_bucket_2y(end))/BPV_bucket_2y;

interpolated_discounts_bucket_2y=zRatesInterp(dates,discounts_2y,PaymentDates_bucket_2y);
BPV_bucket_2y=sum(DeltaMoving_bucket_2y.*interpolated_discounts_bucket_2y(2:end));
Sensitivites_Swap_2y=1-interpolated_discounts_bucket_2y(end)-SwapRate_bucket_2y*BPV_bucket_2y;

PaymentDates_bucket_6y=PaymentDates(1:4:25);
DeltaMoving_bucket_6y=yearfrac(PaymentDates_bucket_6y(1:end-1),PaymentDates_bucket_6y(2:end),6);
interpolated_discounts_bucket_6y=zRatesInterp(dates,discounts,PaymentDates_bucket_6y);
BPV_bucket_6y=sum(DeltaMoving_bucket_6y.*interpolated_discounts_bucket_6y(2:end));
SwapRate_bucket_6y=(1-interpolated_discounts_bucket_6y(end))/BPV_bucket_6y;

interpolated_discounts_bucket_6y=zRatesInterp(dates,discounts_6y,PaymentDates_bucket_6y);
BPV_bucket_6y=sum(DeltaMoving_bucket_6y.*interpolated_discounts_bucket_6y(2:end));
Sensitivites_Swap_6y=1-interpolated_discounts_bucket_6y(end)-SwapRate_bucket_6y*BPV_bucket_6y;

PaymentDates_bucket_10y=PaymentDates(1:4:end);
DeltaMoving_bucket_10y=yearfrac(PaymentDates_bucket_10y(1:end-1),PaymentDates_bucket_10y(2:end),6);
interpolated_discounts_bucket_10y=zRatesInterp(dates,discounts,PaymentDates_bucket_10y);
BPV_bucket_10y=sum(DeltaMoving_bucket_10y.*interpolated_discounts_bucket_10y(2:end));
SwapRate_bucket_10y=(1-interpolated_discounts_bucket_10y(end))/BPV_bucket_10y;

interpolated_discounts_bucket_10y=zRatesInterp(dates,discounts_10y,PaymentDates_bucket_10y);
BPV_bucket_10y=sum(DeltaMoving_bucket_10y.*interpolated_discounts_bucket_10y(2:end));
Sensitivites_Swap_10y=1-interpolated_discounts_bucket_10y(end)-SwapRate_bucket_10y*BPV_bucket_10y;

rates_shifted=rates;
rates_shifted.depos=rates.depos+1e-4;
rates_shifted.futures=rates.futures+1e-4;
rates_shifted.swaps=rates.swaps+1e-4;
[~,discounts_shifted]=bootstrap(ddates,rates_shifted);

DV01_2y=Delta_2y;
DV01_6y=Delta_6y+Delta_2y;
DV01_10y=Delta_10y+Delta_6y+Delta_2y;

x_10y=-DV01_10y*N/Sensitivites_Swap_10y;
x_6y=-(Sensitivites_Swap_10y*x_10y+DV01_6y*N)/Sensitivites_Swap_6y;
x_2y=-(Sensitivites_Swap_10y*x_10y+Sensitivites_Swap_6y*x_6y+DV01_2y*N)/Sensitivites_Swap_2y;