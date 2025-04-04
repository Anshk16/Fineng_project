function ratesSet_shifted = rates_shifter(ratesSet)
%RATES_SHIFTER Shifts all numeric fields of a structure by adding 1 basis point.
%
%   ratesSet_shifted = rates_shifter(ratesSet) takes an input structure 
%   (ratesSet) containing various fields, identifies the numeric fields, 
%   and increases their values by 1 basis point (0.0001). The function 
%   preserves the original structure format and returns the modified 
%   structure.
%
%   Input:
%   - ratesSet: A structure containing various fields, some of which may 
%     be numeric values representing interest rates or financial data.
%
%   Output:
%   - ratesSet_shifted: A modified version of the input structure, where 
%     all numeric fields have been incremented by 1 basis point (0.0001).

    % Create a copy of the original structure to maintain its format
    ratesSet_shifted = ratesSet;

    % Retrieve the field names of the structure
    fieldNames = fieldnames(ratesSet);

    % Iterate through each field of the structure
    for i = 1:numel(fieldNames)
        field = fieldNames{i};

        % Check if the field contains numeric values
        if isnumeric(ratesSet.(field))
            % Add 1 basis point (0.0001) to the numeric values
            ratesSet_shifted.(field) = ratesSet.(field) + 1e-4;
        end
    end
end
