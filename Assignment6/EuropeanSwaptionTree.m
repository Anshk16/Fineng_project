function price = EuropeanSwaptionTree(today, Maturity, Strike, alpha, sigma, TimeSteps, dates, discounts)
IBDayCount = 3;
dt = Maturity / TimeSteps;

muCap = 1 - exp(- alpha * dt);
sigmaCap = sigma * sqrt((1 - exp(- 2 * alpha * dt)) / (2 * alpha));
sigmaStar= (sigma / alpha) * sqrt(dt+(1/(2*alpha))*(1-exp(-2*alpha*dt))-(2/alpha)*(1-exp(-alpha*dt)));
sigma_function = @(s,t) sigma * (1 - exp(- alpha * (t - s))) / alpha;

AlgorithmDates = zeros(TimeSteps + 1,1);
AlgorithmDates(1) = today;
for i = 2:TimeSteps+1
    AlgorithmDates(i) = addtodate(AlgorithmDates(i - 1), 7, "day");
end

% PAY ATTENTION: This is an approx
AlgorithmDiscounts = zRatesInterp(dates, discounts, AlgorithmDates);
        
% Parameters for the tree
dx = sqrt(3) * sigmaCap;

% Searching the optimal l_max (the nearest possible to the lower bound of
% l_max which gives us a multiplier of Deltax)
l_max_down = (1 - sqrt(2/3)) / muCap;
x_max_down = l_max_down * dx;
x_max = ceil(x_max_down / dx) * dx;
l_max = x_max / dx;

flag = x_max / dx;
SwitchCycleFor = flag;

l = 0;
grid_x = zeros(2 * SwitchCycleFor + 1, TimeSteps);
% Building the tree
for i = 1:TimeSteps
    if i <= SwitchCycleFor % Approach A
        l = [l(1) + 1; l; l(end) - 1];
        grid_x(flag - i + 1:flag + i + 1, i) = l * dx; 
    else
        grid_x(:,i) = grid_x(:, i - 1); % We cannot go above l_max/x_max or dowm -l_max/-x_max
    end
end

DiscountedPayoff = @(payoff, discount) payoff .* discount;
% Probabilities for trinomial tree

% Approach A
pu = @(l) 1/2 * (1/3 - l*muCap + (l*muCap).^2);  
pm = @(l) 2/3 - (l*muCap).^2;      
pd = @(l) 1/2 * (1/3 + l*muCap + (l*muCap).^2);

% Approach B
pu_B = 1/2 * (1/3 -l_max*muCap + (l_max*muCap)^2);   
pm_B = -1/3 + 2 * l_max*muCap - (l_max*muCap)^2;         
pd_B = 1/2 * (7/3 - 3*l_max*muCap + (l_max*muCap)^2);  

% Approach C
pu_C = 1/2 * (7/3 - 3*l_max*muCap + ((l_max*muCap)^2));  
pm_C= -1/3 + 2 * l_max*muCap - (l_max*muCap)^2;   
pd_C = 1/2 * (1/3 - l_max*muCap + (l_max*muCap)^2);  

%% Computation of the discounts (not from bootstrap)
term1 = @(ti, tau) 0.5 * (exp(-2 * alpha * tau) - exp(-2 * alpha * yearfrac(today, ti, IBDayCount) - tau) + 1 - exp(-2 * alpha * yearfrac(today, ti, IBDayCount)));
term2 = @(ti, tau) 2 * (exp(-alpha * tau) - exp(-alpha * yearfrac(today, ti, IBDayCount) - tau) + 1 - exp(-alpha * yearfrac(today, ti, IBDayCount)));

integral = @(ti, tau, xi) exp(-xi * sigma_function(0, tau) / sigma - 0.5 * (sigma ^ 2 / alpha ^ 3) * (term1(ti, tau) - term2(ti, tau)));


%% Starting the backward induction
% Starting payoffs
LastIndex = TimeSteps - 1/dt; % we take the last year when we can exercise the swaption (9y)
prev_payoffs=SwaptionPayoff(LastIndex ,AlgorithmDates,grid_x(:, LastIndex),AlgorithmDiscounts,alpha,sigma,Strike, dt, today, term1, term2, integral, sigma_function);

count = 2; % useful when we use only A approach
for i = LastIndex:-1:1 % backward induction from year 9
        
    if i > SwitchCycleFor % We take into account all three approaches
        % Computation of the stochastic discount factor
        B=(AlgorithmDiscounts(i+1)/AlgorithmDiscounts(i))*integral(AlgorithmDates(i), dt, grid_x(:, i));
        D_B=@(deltax) B(1).*exp(-0.5*(sigmaStar^2)-(sigmaStar/sigmaCap)*(deltax+muCap*grid_x(1,i)));
        D_C=@(deltax) B(end).*exp(-0.5*(sigmaStar^2)-(sigmaStar/sigmaCap)*(deltax+muCap*grid_x(end,i)));
        D_A=@(deltax) B(2:end-1).*exp(-0.5*(sigmaStar^2)-(sigmaStar/sigmaCap)*(deltax+muCap*grid_x(2:end-1,i)));

        ExpValue_C = pu_C * DiscountedPayoff(prev_payoffs(1), D_C(0)) + pm_C * DiscountedPayoff(prev_payoffs(2), D_C(-dx)) + pd_C * DiscountedPayoff(prev_payoffs(3), D_C(-2 * dx));
        ExpValue_B = pd_B * DiscountedPayoff(prev_payoffs(end), D_B(0)) + pm_B * DiscountedPayoff(prev_payoffs(end - 1), D_B(dx)) + pu_B * DiscountedPayoff(prev_payoffs(end - 2), D_B(2 * dx));
        ExpValue_A = pd(grid_x(3:end, i) / dx) .* DiscountedPayoff(prev_payoffs(3:end), D_A(-dx)) + pm(grid_x(2:end - 1, i) / dx) .* DiscountedPayoff(prev_payoffs(2:end-1), D_A(0)) + pu(grid_x(1:end - 2, i) / dx) .* DiscountedPayoff(prev_payoffs(1:end-2), D_A(dx));
        prev_payoffs = [ExpValue_C; ExpValue_A; ExpValue_B]; % Updating the payoff

    else % We take only the approach A because we are not anymore on the boundes
        B=(AlgorithmDiscounts(i+1)/AlgorithmDiscounts(i))*integral(AlgorithmDates(i), dt, grid_x(count:end - count + 1, i));
        D=@(deltax) B.*exp(-0.5*(sigmaStar^2)-(sigmaStar/sigmaCap)*(deltax+muCap*grid_x(count:end - count + 1,i)));
        prev_payoffs = pd(grid_x(count + 1:end - count + 2, i) / dx) .* DiscountedPayoff(prev_payoffs(3:end), D(-dx)) + pm(grid_x(count:end - count + 1, i) / dx) .* DiscountedPayoff(prev_payoffs(2:end-1), D(0)) + pu(grid_x(count - 1:end - count, i) / dx) .* DiscountedPayoff(prev_payoffs(1:end-2), D(dx));
        count = count + 1;
    end
end

price = prev_payoffs;