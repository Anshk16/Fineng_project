% runAssignment3
%  group 9, AY2024-2025
% to run:
% > runAssignment4_TBM

clear all;
close all;
clc;

%% Settings
formatData='dd/mm/yyyy'; %Pay attention to your computer settings 

%% Read market data
% This fuction works on Windows OS. Pay attention on other OS.

[datesSet, ratesSet] = readExcelData('MktData_CurveBootstrap', formatData);

%% Bootstrap
% dates includes SettlementDate as first date

[dates, discounts]=bootstrap(datesSet, ratesSet); % TBC

%% Case study 1: Certificate Pricing
% parameters

% Day count conventions
DepoDayCount = 2; % yearfrac Act/360
IBDayCount = 3; % yearfrac Act/365
SwapDateCount = 6; % yearfrac 30/360 European

% Volatility parameters for the two assets
sigma1 = 16.2*1e-2; % Volatility for ENEL
sigma2 = 20*1e-2; % Volatility for AXA
correlation = 45*1e-2; % Correlation between the two assets (ENEL, AXA)

div_e = 2.5*1e-2; % dividend yields ENEL
S0_e = 100; % stock prices ENEL @ 31^st January 2023
div_cs = 2.9*1e-2; % dividend yields AXA
S0_cs = 200;  % stock prices AXA @ 31^st January 2023
upfront = 3*1e-2; % upfront at Certificate issue


protection = 90*1e-2; 
notional = 1e8;
alpha = 110*1e-2; % partecipation coefficient


% Other useful variables
today = dates(1); % valuation date
certificate_dates = BusinessDates(17,today,"month",3, 1); % party A payment dates: Quarterly, subject to Following Business Convention

expiry = certificate_dates(end); % Certificate maturity date
certificate_discounts = zRatesInterp(dates,discounts,certificate_dates); % discount factors for certificate payment dates


basket_dates = BusinessDates(5,today,"year",1, 1); % dates that we need in order to compute the basket
year_frac_basket_dates = yearfrac(today,basket_dates(2:end),IBDayCount);
basket_discounts = zRatesInterp(dates,discounts,basket_dates);
basket_zero_rates = -log(basket_discounts(2:end))./year_frac_basket_dates;



% Basis Point Value (BPV) on floating payment dates
delta_float = yearfrac(certificate_dates(1:end-1),certificate_dates(2:end),DepoDayCount);
BPV = sum(delta_float.*certificate_discounts(2:end));


% Monte Carlo simulation setup (with Cholesky)
montecarlo_simulations = 1e7;
corr_mat = [1 correlation
            correlation 1];           
rng(30)
g = randn(montecarlo_simulations,2); 
C = chol(corr_mat); % Cholesky decomposition
g = g*C;


% Initialize variables 
previousE_e = S0_e;
previousE_cs = S0_cs;
S_basket = zeros(montecarlo_simulations,1);

% Monte Carlo simulation loop (4 years) and computation of the basket
for i = 1:4

    % Simulate ENEL stock price using Geometric Brownian Motion

    E_e = S0_e * exp((basket_zero_rates(i) - div_e - 0.5*sigma1^2) * year_frac_basket_dates(i) + sigma1 * sqrt(year_frac_basket_dates(i)) * g(:,1));
    
    % Simulate AXA stock price using Geometric Brownian Motion

    E_cs = S0_cs * exp((basket_zero_rates(i) - div_cs - 0.5*sigma2^2) * year_frac_basket_dates(i) + sigma2 * sqrt(year_frac_basket_dates(i)) * g(:,2));
    
    yearly_perf = 0.5*(E_e./previousE_e) + 0.5*(E_cs./previousE_cs);
    S_basket = S_basket + 0.25 * yearly_perf; % 25% weight per year

    % Update previous prices for next iteration
    previousE_e = E_e;
    previousE_cs = E_cs;

end

%Pays at expiry date the participation coefficient of the performance, if
%positive, of an equally weighted basket of ENEL S.p.A. and AXA S.A.

coupon = alpha * max(S_basket - protection, 0);
expected_coupon = mean(coupon);

% Calculate Spread over LIBOR (SPOL)
SPOL = (certificate_discounts(end) * (expected_coupon + protection)  + upfront - 1) / BPV; % compute thanks to NPV equal to 0
SPOL_bps = SPOL * 10000; % convert to basis points


disp(['Spread over Libor: ', num2str(SPOL_bps), ' bps']);

%% Exercise 2 : Pricing Digital Option

load('cSelect.mat')
%Parameters
Pricing_date = dates(1);
Notional = 10e6;
Digital_payoff = 0.07 * Notional;
Expiry = 1; %after 1y we have expiry of the contract
S_0 = cSelect.reference;
dist = cSelect.dividend; %dividend
K = S_0; %the strike is equal to the underlying at time 0, ATM
sigma = interp1(cSelect.strikes, cSelect.surface, K); %we want our volatility interpolated

%The digital call formula in Black approach is : Price = Discount * N(d2)
% Discount computation

Expiry_date = addtodate(Pricing_date, Expiry, "year"); %Date of the expiry
delta_time = yearfrac(Pricing_date, Expiry_date, IBDayCount); %1y of delta time
Discount = zRatesInterp(dates,discounts,Expiry_date);

r = - log(Discount) / delta_time; %interest rate
% N(d2) computation

F_0 = S_0 * exp(delta_time * (r - dist)); %computing forward price at time 0

d2 = log(F_0 / K) / sqrt(delta_time * sigma^2) - 0.5 * sqrt(delta_time * sigma^2);

%% Black price

Price_Black = Discount * normcdf(d2) * Digital_payoff;

fprintf('The price with Black approach is:      %.2f\n', Price_Black)

%% Implied volatility approach

Black_term = Price_Black;

% slope impact on the formula

epsilon = 1e-12; %we need this number close to 0, we are computing incremental ratio

sigma_plus = interp1(cSelect.strikes, cSelect.surface, K + epsilon);
sigma_minus = interp1(cSelect.strikes, cSelect.surface, K - epsilon);

slope_impact = (sigma_plus - sigma_minus) / (2 * epsilon);

%we can see that the third term is the vega, so we are using a MATLAB
%function for that

vega = blsvega(S_0, K, r, delta_time, sigma, dist);

%Now we compute the price according to the Implied Volatility approach

Price_IV_approach = Black_term - slope_impact * vega * Digital_payoff;
fprintf('The price in Implied Volatility approach is:   %.2f\n', Price_IV_approach)

%% Plotting

% We are plotting our volatility according to the strike we have
figure(1)
plot(cSelect.strikes, cSelect.surface,K, sigma, 'o');
xlabel('Strikes')
ylabel('Volatilities')
legend('Implied Volatility Surface', 'Sigma')

%% Checking the difference between the two prices

error = abs(Price_IV_approach - Price_Black);
fprintf('The error between Black formula and IV Approach is:     %.2f\n', error)

% %% Pricing with different strikes 
% % this is a facultative point, if you want to plot it
% 
% %facultative, you can delete this part
% strikes = cSelect.strikes;
% volatilities = cSelect.surface;
% 
% d2 = log(F_0 ./ strikes) ./ sqrt(delta_time) .* volatilities - 0.5 * sqrt(delta_time) * volatilities;
% 
% d1 = d2 + sqrt(delta_time) * volatilities;
% 
% multiPrice_Black = Discount * normcdf(d2) * Digital_payoff;
% 
% epsilon = 1e-12;
% 
% sigma_plus = interp1(cSelect.strikes, cSelect.surface, strikes + epsilon);
% sigma_minus = interp1(cSelect.strikes, cSelect.surface, strikes - epsilon);
% 
% slope_impact = (sigma_plus - sigma_minus) / (4 * epsilon);
% 
% %third term
% 
% vega = blsvega(S_0, strikes, r, delta_time, volatilities, dist);
% 
% multiPrice_IV_approach = Black_term - slope_impact .* vega * Digital_payoff;
% 
% error = abs(Price_IV_approach - Price_Black);
% max_error = max(error)
% min_error = min(error)
% % 
% % plot(cSelect.strikes, cSelect.surface,K, sigma, 'o');
% % legend('Implied Volatility Surface', 'Sigma')
% figure(1)
% plot(strikes, multiPrice_Black, strikes, multiPrice_IV_approach);
% legend('Price Black', 'Price IV Approach')
% figure(2)
% plot(volatilities, multiPrice_Black, volatilities, multiPrice_IV_approach)


%% Exercise 3: Pricing

%runPricingFourier, here we have the main for the different approaches
%Parameters
sigma = 0.2;
k = 1;
eta = 3;
t = 1;
x_min = -0.25;
x_max = 0.25;
grid_step = 0.01;
x_moneyness = x_min:grid_step:x_max;

Pricing_date = dates(1);
S_0 = cSelect.reference;
IBDayCount = 3; %Act/365
dist = cSelect.dividend; %dividend

%discount computation

Expiry_date = addtodate(Pricing_date, t, "year");
delta_time = yearfrac(Pricing_date, Expiry_date, IBDayCount);
discount = zRatesInterp(dates,discounts,Expiry_date);
r = - log(discount) / delta_time; %interest rate
F_0 = S_0 * exp(delta_time * (r - dist)); %forward price at time 0

% Parameters for integral computations
M = 15;
x_1 = -500;
dz = 0.0025;

montecarlo_simulations = 1e6; %Monte Carlo paramenter for how many simulations we want

alpha = 0; %VG, useful for characteristic function computations

figure(2)

%Pay attention: for the call price we take only the real part. The
%immaginary part is infinitesimal when we compute the integral

for i = 1:4
    flag = i; %switch method
    
    if i == 4
        C_price_Montecarlo = MonteCarlo(montecarlo_simulations, delta_time, sigma, eta, discount, x_moneyness, k, F_0);
        hold on
        plot(x_moneyness, C_price_Montecarlo)
    else
        C_price = real(call_price_integral(discount, F_0, k, sigma, eta, x_moneyness, delta_time, M, x_1, dz, flag, alpha));
        hold on
        plot(x_moneyness, C_price)
    end
end
xlabel('x moneyness')
ylabel('Call price')
legend('FFT method 1', 'FFT method 2', 'Quadratic method', 'MC method')

%% Facultative point

alpha = 1/3;

figure(3)

for i = 1:4
    flag = i; %switch method
    
    if i == 4
        C_price_Montecarlo = MonteCarlo(montecarlo_simulations, delta_time, sigma, eta, discount, x_moneyness, k, F_0);
        hold on
        plot(x_moneyness, C_price_Montecarlo)
    else
        C_price = real(call_price_integral(discount, F_0, k, sigma, eta, x_moneyness, delta_time, M, x_1, dz, flag, alpha));
        hold on
        plot(x_moneyness, C_price)
    end
end

xlabel('x moneyness')
ylabel('Call price alpha')
legend('FFT method 1 alpha', 'FFT method 2 alpha', 'Quadratic method alpha', 'MC method')


%% Case study 4: Volatility surface calibration

%Parameters
t = 1;
x_min = -0.25;
x_max = 0.25;
grid_step = 0.01;
sigma = cSelect.surface;


Pricing_date = dates(1);
S_0 = cSelect.reference;
d = cSelect.dividend;
K = cSelect.strikes;

Expiry_date = addtodate(Pricing_date, t, "year");
delta_time = yearfrac(Pricing_date, Expiry_date, IBDayCount);
discount = zRatesInterp(dates,discounts,Expiry_date);
r = - log(discount) / delta_time; %interest rate
F_0 = S_0 * exp(delta_time * (r - d)); %forward price at time 0

% Parameters for integral computations
M = 15;
x_1 = -500;
dz = 0.0025;
flag = 1;
alpha = 2/3;

% Price with third exercise methodology

Call_Price_Lewis = @(sigma, eta, k, x_moneyness) call_price_integral(discount, F_0, k, sigma, eta, x_moneyness, delta_time, M, x_1, dz, flag, alpha);

% Price with Black formula

Call_Price_Black = arrayfun(@(sigma, K) blkprice(F_0, K, r, delta_time, sigma), sigma, K);

% Least squared calibration

dist = @(sigma, eta, k, x_moneyness) sum((Call_Price_Black - real(Call_Price_Lewis(sigma, eta, k, x_moneyness))) .^2);

% We select the known parameters as the starting point of the MATLAB function
% 'fminsearch'

k = 1;
eta = 3;
x_moneyness = 0.10;

% We calibrate the parameters such that the function dist is minimized
optimal_parameters = [sigma, eta, k];
    
calibrated_par = fminsearch(@(calibrated_par) dist(calibrated_par(1), calibrated_par(2), calibrated_par(3), x_moneyness), optimal_parameters);

%The implied volatilities are the first values of our variable
%calibrated_par
imp_vol_cal = calibrated_par(1:length(sigma));

figure(4)

plot(K, imp_vol_cal, K, sigma)
xlabel('Strikes')
ylabel('Volatilities')
legend('Calibrated', 'Mkt')

% Calibrated skew

fprintf('The calibrated skew is:   %.2f\n', calibrated_par(length(sigma) + 1))

% Calibrated volatility of the volatility

fprintf('The calibrated volatility of the volatility is:   %.2f\n', calibrated_par(end))


%% We plot the calibration for different choices of x_moneyness
% 
% %Not suggest to run
% 
% % We calibrate the parameters such that the function dist is minimized
% optimal_parameters = [sigma, eta, k];
% 
% for x_moneyness = -0.15:0.05:0.15
%     calibrated_par = fminsearch(@(calibrated_par) dist(calibrated_par(1), calibrated_par(2), calibrated_par(3), x_moneyness), optimal_parameters);
% 
%     %The implied volatilities are the first values of our variable
%     %calibrated_par
%     imp_vol_cal = calibrated_par(1:length(sigma));
% 
%     plot(K, imp_vol_cal, K, sigma)
%     xlabel('Strikes')
%     ylabel('Volatilities')
%     legend('Calibrated', 'Mkt')
% end


