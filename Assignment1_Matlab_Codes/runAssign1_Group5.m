% Assignment_1
% Group 5, AA2024-2025
%
%
clear all; close all; clc;
%% Pricing parameters
S0=1;
K=1.05;
r=0.025;
TTM=1/3; 
sigma=0.21;
flag=1; % flag:  1 call, -1 put
d=0.02;
n_contracts = 10^6;

%% Quantity of interest
B=exp(-r*TTM); % Discount

%% Pricing 
clc;
F0=S0*exp(-d*TTM)/B;     % Forward in G&C Model

rng("default"); % fix the seed for MC

% pricingMode: 1 ClosedFormula, 2 CRR, 3 Monte Carlo
prices = zeros(3,1);
M=100; % M = simulations for MC, steps for CRR;

Price_closed = EuropeanOptionPrice(F0,K,B,TTM,sigma,1,M,flag)
Price_CRR = EuropeanOptionPrice(F0,K,B,TTM,sigma,2,M,flag)
Price_MC = EuropeanOptionPrice(F0,K,B,TTM,sigma,3,10^2,flag)

%% Errors Rescaling 
clc;

% Plot Errors for CRR varying number of steps
[M, errCRR] = PlotErrorCRR(F0, K, B, TTM, sigma);

% Crea una figura per il plot
figure;
grid on;
hold on; % Permette di sovrapporre più plot

% Plot degli errori CRR in scala log-log
loglog(M, errCRR, 'b', 'LineWidth', 1.5); % Blu per CRR
x = logspace(log10(2), log10(1000), 1000); % Genera punti in scala logaritmica
loglog(x, 1 ./ x / 100, 'b--', 'LineWidth', 1.5); % Linea tratteggiata per 1/(x*100)

% Plot Errors for MC varying number of simulations N
[M, stdEstim] = PlotErrorMC(F0, K, B, TTM, sigma);

% Plot degli errori Monte Carlo in scala log-log
loglog(M, stdEstim, 'r', 'LineWidth', 1.5); % Rosso per MC
x = logspace(log10(2), log10(1000000), 1000); % Genera punti in scala logaritmica
loglog(x, 1 ./ sqrt(x) / 10, 'r--', 'LineWidth', 1.5); % Linea tratteggiata per 1/(sqrt(x)*10)

hold on;
yline(1e-4, 'k--', '1bp','LineWidth', 1.5);
hold off;

% Aggiungi legenda
legend('CRR Error', '1/(x*100)', 'MC Error', '1/(sqrt(x)*10)', '1bp', 'Location', 'best');

% Aggiungi etichette e titolo
xlabel('Number of Steps / Simulations');
ylabel('Error');
title('Convergence of CRR and MC Methods (Log-Log Scale)');

% Migliora la visualizzazione
set(gca, 'FontSize', 12); % Imposta la dimensione del font
set(gca, 'XScale', 'log'); % Imposta l'asse x in scala logaritmica
set(gca, 'YScale', 'log'); % Imposta l'asse y in scala logaritmica
hold off;


%% KO Option
clc;
format long

rng('default')
KO=1.4;
F0=S0*exp(-d*TTM)/B;     % Forward in G&C Model

OptionPriceKOClosed = EuropeanOptionKOClosed(F0,K,KO,B,TTM,sigma)
M=10^2; % M = simulations for MC, steps for CRR;
OptionPriceKOCRR = EuropeanOptionKOCRR(F0,K,KO,B,TTM,sigma,M)
M=10^4;
OptionPriceKOMC = EuropeanOptionKOMC(F0,K, KO,B,TTM,sigma,M)

%% KO Option Vega
clc;
price_space = 0.65:0.01:1.45;
v_closed = zeros(1,length(price_space));
v_crr = zeros(1,length(price_space));
v_mc = zeros(1,length(price_space));


for i=1:length(price_space)
    Fi = price_space(i)*B;
    v_crr(i) = VegaKO(Fi,K,KO,B,TTM,sigma,5000,1);
    v_mc(i) = VegaKO(Fi,K,KO,B,TTM,sigma,5000,2);
    v_closed(i) = VegaKO(Fi,K,KO,B,TTM,sigma,5000,3);
end


figure
subplot(1,2,1);
plot(price_space, v_closed);
hold on; grid on;
plot(price_space, v_crr);
legend('Closed','CRR');
title('Closed - CRR')

subplot(1,2,2);
plot(price_space, v_closed);
grid on; hold on;
plot(price_space, v_mc);
legend('Closed','MC');
title('Closed - MC')

%% Antithetic Variables
clc;
[M, stdEstim] = PlotErrorMC(F0, K, B, TTM, sigma);

errAnti = zeros(size(M)); % Inizializza il vettore di zeri con la stessa dimensione di M

for idx = 1:length(M)  % Itera sugli indici di M
    i = M(idx); % Estrai il valore di M all'indice idx
    errAnti(idx) = AntitheticERR(F0, K, i, TTM, sigma);
end


% Crea una figura per il plot
figure;
grid on;
hold on; % Permette di sovrapporre più plot

% Plot degli errori CRR in scala log-log
loglog(M, errAnti, 'b', 'LineWidth', 1.5); % Blu per CRR
x = logspace(log10(2), log10(1000), 1000); % Genera punti in scala logaritmica

% Plot degli errori Monte Carlo in scala log-log
loglog(M, stdEstim, 'r', 'LineWidth', 1.5); % Rosso per MC
x = logspace(log10(2), log10(1000000), 1000); % Genera punti in scala logaritmica

hold on;
yline(1e-4, 'k--', '1bp','LineWidth', 1.5);
hold off;

% Aggiungi legenda
legend('MC + AV Error', 'MC Error', '1bp');

% Aggiungi etichette e titolo
xlabel('Number of Steps / Simulations');
ylabel('Error');
title('Error: MC + AV vs MC');

% Migliora la visualizzazione
set(gca, 'FontSize', 12); % Imposta la dimensione del font
set(gca, 'XScale', 'log'); % Imposta l'asse x in scala logaritmica
set(gca, 'YScale', 'log'); % Imposta l'asse y in scala logaritmica
hold off;

%% Bermudan option price
clc; format long

N = 1000;
bermudan_price = BermudanOptionCRR(F0, K, B, TTM, sigma, d, N)
Price_CRR = EuropeanOptionPrice(F0,K,B,TTM,sigma,2,N,flag)

%% Bermudan Option: Dividend Yield
N = 100
clc; format long
d_step = 1000; % number of steps
di = linspace(0,0.05,d_step); % divide the interval [0,0.05] which is the dividend in d_step steps

eur_price = zeros(1,d_step);
bermudan_price = zeros(1,d_step);
for i=1:d_step
    Fi =S0*exp(-di(i)*TTM)/B;     % Forward in G&C Model
    eur_price(i) = EuropeanOptionCRR(Fi,K,B,TTM,sigma,N,1);
    bermudan_price(i) = BermudanOptionCRR(Fi,K,B,TTM,sigma,di(i),N);
end

figure;
title('European vs Bermudan price wrt dividend');
plot(di, eur_price);
hold on; 
grid on;
plot(di, bermudan_price);
legend('European','Bermudan');
xlabel('Dividend Yield');
ylabel('Price');
