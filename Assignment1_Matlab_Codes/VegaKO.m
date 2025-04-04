function vega = VegaKO(F0, K, KO, B, T, sigma, N, flagNum)
    % Funzione per calcolare la vega di una call option con knock-out (KO)
    % Input:
    %   F0: Prezzo forward del sottostante
    %   K: Strike price della call option
    %   KO: Barriera di knock-out
    %   B: Payoff della digital call (solitamente 1)
    %   T: Tempo alla scadenza (in anni)
    %   sigma: Volatilità del sottostante
    %   N: Numero di simulazioni (non utilizzato in questo caso)
    %   flagNum: Flag per il metodo di calcolo (3 = closed formula)
    % Output:
    %   gamma: Vega della call option con knock-out

    % Inizializzazione dell'output
    vega = NaN;
    
    % Calcolo vega con CRR formula se flagNum = 1
    if flagNum == 1
        % Piccola variazione della volatilità per il calcolo empirico
        deltaSigma = 0.01;

        % Calcolo del prezzo della call con knock-out con la volatilità originale
        price_original = EuropeanOptionKOCRR(F0, K, KO, B, T, sigma - deltaSigma, N);

        % Calcolo del prezzo della call con knock-out con la volatilità perturbata
        price_perturbed = EuropeanOptionKOCRR(F0, K, KO, B, T, sigma + deltaSigma, N);

        % Calcolo della vega empirica
        vega = ((price_perturbed - price_original) / (2*deltaSigma))*deltaSigma;
    end
    
    % Calcolo vega con MC se flagNum = 2
    if flagNum == 2
        % Piccola variazione della volatilità per il calcolo empirico
        deltaSigma = 0.01;

        % Calcolo del prezzo della call con knock-out con la volatilità originale
        price_original = EuropeanOptionKOMC(F0, K, KO, B, T, sigma - deltaSigma, N);

        % Calcolo del prezzo della call con knock-out con la volatilità perturbata
        price_perturbed = EuropeanOptionKOMC(F0, K, KO, B, T, sigma + deltaSigma, N);

        % Calcolo della vega empirica
        vega = ((price_perturbed - price_original) / (2*deltaSigma))*deltaSigma;        
    end
    
    % Calcolo vega con la closed formula se flagNum = 3
    if flagNum == 3
        deltaSigma = 0.01;
        % Calcolo di d1 e d2 per la call con strike K
        d1_K = (log(F0/K) + (0.5*sigma^2)*T) / (sigma*sqrt(T));
        d2_K = d1_K - sigma*sqrt(T);

        % Calcolo di d1 e d2 per la call con strike KO
        d1_KO = (log(F0/KO) + (0.5*sigma^2)*T) / (sigma*sqrt(T));
        d2_KO = d1_KO - sigma*sqrt(T);
        
        r = -log(B)/T;

        % Calcolo di d1 e d2 per la digital
        d2_DigKO = (log(F0*B/KO) + (r - sigma^2/2) * T)/(sigma * sqrt(T));
        d1_DigKO = d2_DigKO + sigma * sqrt(T);

        % Vega della call bull spread (call con strike K - call con strike KO)
        vega_bull_spread = B * F0 * normpdf(d1_K) * sqrt(T) - B * F0 * normpdf(d1_KO) * sqrt(T);

        % Vega della digital call con strike KO
        vega_digital_call = -B * normpdf(d2_DigKO) * (d1_DigKO / sigma) * (KO - K);

        % Vega della call option con knock-out
        vega = (vega_bull_spread - vega_digital_call) * deltaSigma;
    end

end

