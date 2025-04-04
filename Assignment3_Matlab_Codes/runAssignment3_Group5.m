% runAssignment3
% Group 5, AY2024-2025
%
% To run:
% > runAssignment3_Group5
clear; close all; clc;
addpath(genpath(fullfile(pwd, 'Functions'))); % Add 'Functions' directory to the path

% Settings
formatData = 'dd/mm/yyyy';

% Read Market Data
[datesSet, ratesSet] = readExcelData('MktData_CurveBootstrap', formatData);

%% Bootstrap Discount Curve
% 'dates' includes the SettlementDate as the first date
clc; format long
[dates, discounts] = bootstrap(datesSet, ratesSet); 

%% Asset Swap Spread Calculation
CP = 1.01; % Clean Price
r = 0.048; % Fixed coupon rate
n_years = 6;
issue_date = datenum('31-Mar-2022'); % Bond issue date

% Compute Asset Swap Spread
s_asw = AssetSwapSpread(dates, discounts, CP, r, issue_date, n_years);
disp('Asset Swap Spread:')
disp(s_asw)

%% CDS Bootstrap
recovery = 0.4; % Recovery rate
n_years = 6;
settle_date = dates(1);
spreadsCDS  = [40 44 47 49 51 52]; % CDS spreads in basis points
datesCDS = Add_dates_pre(settle_date, n_years);

colors = ['r', 'g', 'b']; % Colors for different bootstrapping methods
hold on;

legend_labels = {'Without Accrual', 'With Accrual', 'JT'}; % Custom legend labels

for flag = 1:3
    [datesCDS, survProbs, intensities] = bootstrapCDS(dates, discounts, datesCDS, spreadsCDS, flag, recovery); %#ok<ASGLU>
    plot(datesCDS(2:end), survProbs, 'Color', colors(flag), 'LineWidth', 0.8);
end

hold off;
xlabel('CDS Dates');
ylabel('Survival Probability');
title('Survival Probability vs. CDS Maturities');

% Set the legend with custom labels
legend(legend_labels, 'Location', 'best');

grid on;
disp('-------------------------------------')

% Suppress output from bootstrapCDS
evalc('[datesCDS, survProbs, intensities] = bootstrapCDS(dates, discounts, datesCDS, spreadsCDS, 1, recovery);');

% Plot Intensities (Piecewise Constant Function)
figure;
hold on; grid on;

% Create x and y coordinates for the step plot
x_int = reshape([datesCDS(1:end-1) datesCDS(2:end)]', [], 1); % Duplicate dates
y_int = reshape([intensities intensities]', [], 1); % Duplicate intensities

plot(x_int, y_int, 'r', 'LineWidth', 2); % Stepwise plot of intensities

title('Default Intensities');
legend('Intensities (No Accrual)', 'Location', 'southeast'); % Legend in bottom-right
xlabel('Dates');
ylabel('Intensities');

hold off;

%% Credit Simulation 

bp = 0.0001; % Basis point conversion factor
lamb1 = 5 * bp; % Default intensity in the first regime
lamb2 = 9 * bp; % Default intensity in the second regime
teta = 4; % Change point between regimes
T = 30; % Simulation time horizon

% Run credit simulation
tau = CreditSimulation(lamb1, lamb2, teta, T);
disp('-------------------------------------')

%% MBS Pricing
clc;
format long;

% Parameter Definition
Kd = 0.05; % Lower loss threshold
Ku = 0.09; % Upper loss threshold
recovery = 0.2; % Recovery rate
rho = 0.4; % Correlation coefficient
p = 0.05; % Default probability per entity
notional = 1e9 * (Ku - Kd); % Notional amount of the MBS tranche

% Define the range for I (logarithmic scale)
I_values_full = logspace(1, log10(2e4), 10); % Values between 10 and 2*10^4
I_values_exact = floor(logspace(1, log10(200), 4)); % Values between 10 and 200 for the exact solution

% Initialize solution vectors
exact_sol = zeros(size(I_values_exact));
approx_sol = zeros(size(I_values_full));
LHP_sol = zeros(size(I_values_full));

% Disable all warnings during computation
warning('off', 'all');

% Compute exact MBS pricing solutions
for i = 1:length(I_values_exact)
    I = I_values_exact(i);
    exact_sol(i) = MBS_Pricing_HP(Kd, Ku, recovery, I, rho, p, discounts, dates);
end

% Re-enable warnings after execution
warning('on', 'all');

% Compute approximate and LHP solutions
for i = 1:length(I_values_full)
    I = I_values_full(i);
    approx_sol(i) = MBS_Pricing_KL(Kd, Ku, recovery, I, rho, p, discounts, dates);
    LHP_sol(i) = MBS_Pricing_LHP(Kd, Ku, recovery, I, rho, p, discounts, dates);
end

% Plot results with a logarithmic x-axis
figure;
semilogx(I_values_full, approx_sol * notional, 'rd-', 'LineWidth', 1.5, 'MarkerFaceColor', 'r'); hold on;
semilogx(I_values_full, LHP_sol * notional, 'b-', 'LineWidth', 1.5);
semilogx(I_values_exact, exact_sol * notional, 'go-', 'LineWidth', 1.5); % Exact solution only between 10 and 200

xlabel('I (log scale)');
ylabel('MBS Pricing Solutions');
title('Comparison of MBS Pricing Solutions vs. I');
legend('Approximate Solution', 'LHP Solution', 'Exact Solution (10-200)', 'Location', 'southeast'); % Legend in bottom-right
grid on;

%% MBS Pricing for Equity Tranche (KL Approach)
clc;
format long;

% Parameter Definition
Kd = 0.00; % Lower loss threshold for equity tranche
Ku = 0.05; % Upper loss threshold for equity tranche
recovery = 0.2; % Recovery rate
rho = 0.4; % Correlation coefficient
p = 0.05; % Default probability per entity
notional = 1e9 * (Ku - Kd); % Notional amount of the MBS equity tranche

% Define the range for I (logarithmic scale)
I_values_full = logspace(1, log10(2e4), 10); % Values between 10 and 2*10^4
I_values_exact = floor(logspace(1, log10(200), 4)); % Values between 10 and 200 for the exact solution

% Initialize solution vectors
exact_sol = zeros(size(I_values_exact));
equity_LHP_sol = zeros(size(I_values_full));

% Disable all warnings during computation
warning('off', 'all');

% Compute exact MBS pricing solutions (if applicable)
for i = 1:length(I_values_exact)
    I = I_values_exact(i);
    exact_sol(i) = MBS_Pricing_HP(Kd, Ku, recovery, I, rho, p, discounts, dates);
end

% Re-enable warnings after execution
warning('on', 'all');

% Compute LHP solutions for the equity tranche
for i = 1:length(I_values_full)
    I = I_values_full(i);
    equity_LHP_sol(i) = MBS_Pricing_LHP(Kd, Ku, recovery, I, rho, p, discounts, dates);
end

equity_KL_sol = MBS_Pricing_KL(Kd, Ku, recovery, I, rho, p, discounts, dates);
disp('Price Equity @I=100:')
disp(equity_KL_sol*notional)

% Plot results with a logarithmic x-axis
figure;
semilogx(I_values_full, equity_LHP_sol * notional, 'b-', 'LineWidth', 1.5); hold on;
semilogx(I_values_exact, exact_sol * notional, 'go-', 'LineWidth', 1.5); % Exact solution only between 10 and 200

xlabel('I (log scale)');
ylabel('MBS Pricing Solutions');
title('Comparison of MBS Pricing Solutions vs. I');
legend('LHP Solution', 'Exact Solution (10-200)', 'Location', 'southeast'); % Legend in bottom-right
grid on;
