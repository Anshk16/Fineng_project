function y = CreditSimulation(lamb1, lamb2, teta, T)
    %% Simulation of default time
    % This section simulates default times using an inhomogeneous exponential model.
    
    rng('default'); % Resets random number generator to MATLAB’s default seed
    M = 10^5; % Number of simulations
    u = rand(M, 1); % Generate M uniform random numbers

    % Allocate memory for tau (default times)
    tau = zeros(M,1);
    
    % Simulate default times using piecewise exponential distribution
    for i = 1:M
        y = -log(1 - u(i)) / lamb1;
        if y < teta
            tau(i) = y; % If within first regime, use lambda1
        else
            tau(i) = (-log(1 - u(i)) + lamb2 * teta - lamb1 * teta) / lamb2; % Second regime
        end
    end

    % Compute empirical survival probabilities
    probs = zeros(T, 1);
    for i = 1:T
        probs(i) = 1 - sum(tau <= i) / M; % Fraction of simulations surviving beyond i
    end

    % Compute theoretical survival probability using piecewise exponential model
    P = exp(-lamb1 * (1:T)) .* ((1:T) < teta) + exp(-lamb1 * teta - lamb2 * ((1:T) - teta)) .* ((1:T) >= teta);

    %% Plot the results correctly
    figure
    hold on
    grid on
    
    % Plot simulated probabilities
    plot(1:T, probs, 'b-', 'LineWidth', 2); % Simulated survival probability (blue solid line)
    
    % Plot theoretical survival probability
    plot(1:T, P, 'r--', 'LineWidth', 2); % Theoretical probability (red dashed line)
    
    % Labels and title
    xlabel('Time (T)');
    ylabel('Survival Probability');
    title('Credit Simulation vs Theoretical Survival Probability');
    
    % Legend
    legend('Simulated Survival Probability', 'Theoretical Probability', 'Location', 'Best');
    
    hold off

    %% Fit of the lambdas (Parameter Estimation)
    % This section estimates the parameters lambda1 and lambda2 by fitting 
    % the theoretical survival function to the simulated probabilities.

    % Define the theoretical survival model as a function of lambda1 and lambda2
    ft = fittype(@(lambda1, lambda2, t) ...
        exp(-lambda1 * t) .* (t < teta) + exp(-lambda1 * teta - lambda2 * (t - teta)) .* (t >= teta), ...
        'independent', 't', 'coefficients', {'lambda1', 'lambda2'});
    
    % Fit the model to the simulated survival probabilities
    fit_result = fit((1:T)', probs, ft, 'StartPoint', [0.5, 0.5]); % Initial guesses for λ₁ and λ₂

    % Extract estimated parameters
    lambda1_hat = fit_result.lambda1;
    lambda2_hat = fit_result.lambda2;

    % Compute 95% confidence intervals for estimated parameters
    ci = confint(fit_result, 0.95);
    lambda1_CI = ci(:,1); 
    lambda2_CI = ci(:,2);
    
    % Display estimated parameters and confidence intervals
    disp(['Estimated λ₁: ', num2str(lambda1_hat)]);
    disp(['Estimated λ₂: ', num2str(lambda2_hat)]);
    disp(['Confidence Interval for λ₁: [', num2str(lambda1_CI(1)), ', ', num2str(lambda1_CI(2)), ']']);
    disp(['Confidence Interval for λ₂: [', num2str(lambda2_CI(1)), ', ', num2str(lambda2_CI(2)), ']']);

end
