function datesSet_add = Add_dates(dt, n)
    % Converte dt in datetime se non lo è già
    if ~isa(dt, 'datetime')
        dt = datetime(dt, 'ConvertFrom', 'datenum');
    end

    % Inizializza un array di NaT per memorizzare le date
    datesSet_add = NaT(n+1, 1);

    % Genera la sequenza di date
    for i = 0:n
        datesSet_add(i+1) = dt + calyears(i); % Aggiunge i anni alla data iniziale
        
        % Controlla se la data cade nel weekend e la sposta al lunedì successivo
        if weekday(datesSet_add(i+1)) == 7
            datesSet_add(i+1) = datesSet_add(i+1) + days(2);  % Sabato → Lunedì
        elseif weekday(datesSet_add(i+1)) == 1
            datesSet_add(i+1) = datesSet_add(i+1) + days(1);  % Domenica → Lunedì
        end
    end
end
