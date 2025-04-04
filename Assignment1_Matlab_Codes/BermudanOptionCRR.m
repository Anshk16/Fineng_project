function optionPrice = BermudanOptionCRR_prova(F0, K, B, T, sigma, d, N)

    % Ensure N is a multiple of 4
    N = 4 * floor(N / 4);
    dt = T / N;
    u = exp(sigma * sqrt(dt));
    d_tree = 1 / u; % Down factor
    q = 1 / (u + 1); % Risk-neutral probability

    r = -log(B) / T;
    B_dt = exp(-r * dt);

    % Initialize the asset prices at maturity

    num_elements = length(-N:2:N);
    Ft = zeros(1, num_elements); % Preallocazione del vettore

    index = 1; % Indice per riempire Ftt
    for i = -N:2:N
        Ft(index) = F0 * u^i; % Calcolo del valore
        index = index + 1; % Aggiorna l'indice
    end

    leavesCRR = max(Ft - K, 0);

    % Reduce the tree to the root for the Bermudan option
    for i = N-1:-1:0
        % Update the option price
        leavesCRR = B_dt * ((1 - q) * leavesCRR(1:end-1) + q * leavesCRR(2:end));
        
        % Check for early exercise points
        if mod(i, N/4) == 0
            % Compute the price of the forward at time t
            Fi = F0 * u.^(-i:2:i);

            n = length(-i:2:i);
            Fi = zeros(1, n); % Preallocazione del vettore

            j = 1; % Inizializzazione dell'indice del vettore
            for k = -i:2:i
                Fi(j) = F0 * u^k; % Calcolo dell'elemento
                j = j + 1; % Incremento dell'indice
            end


            Sti = Fi / exp((r - d) * (N - i) * dt);
            leavesCRR = max(leavesCRR, Sti - K);
        end
    end

    optionPrice = leavesCRR;
end