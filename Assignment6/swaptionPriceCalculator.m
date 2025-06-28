function [price, delta] = swaptionPriceCalculator(S0, strike, refDate, expiry, underlyingExpiry, sigmaBlack, freq, dates, discountFactors, swaptionType, computeDelta)
% swaptionPriceCalculator  Calculate Black swaption price (and delta) in MATLAB
%
%   [price, delta] = swaptionPriceCalculator(S0, strike, refDate, expiry,
%       underlyingExpiry, sigmaBlack, freq, dates, discountFactors,
%       swaptionType, computeDelta)
%
% Inputs:
%   S0               - Forward swap rate (scalar)
%   strike           - Swaption strike rate (scalar)
%   refDate          - Valuation date (datetime)
%   expiry           - Swaption expiry date (datetime)
%   underlyingExpiry - Underlying swap termination date (datetime)
%   sigmaBlack       - Implied volatility (scalar)
%   freq             - Fixed leg payment frequency per year (integer)
%   dates            - Vector of curve dates (datetime array)
%   discountFactors  - Corresponding discount factors P(0,T) (numeric vector)
%   swaptionType     - 'receiver' or 'payer' (string)
%   computeDelta     - Boolean flag; compute delta if true
%
% Outputs:
%   price - Swaption price
%   delta - Swaption delta (if computeDelta = true)

% Time to expiry fraction (ACT/365)
x = yearfrac(refDate, expiry, 3);

%  Black parameters
d1 = (log(S0/strike) / (sigmaBlack * sqrt(x))) + 0.5 * sigmaBlack * sqrt(x);
d2 = d1 - sigmaBlack * sqrt(x);

%  payment schedule for underlying swap (annual steps of 1/freq years)
step = calmonths(12/freq);
discountDates = expiry:step:underlyingExpiry;

%  Interpolate discount factors at those dates
B0_T = interp1(dates, discountFactors, discountDates, 'linear', 'extrap');

% forward-start discounts B(Te, Ti)
B0_Te = B0_T(1);
B_Te_T = B0_T ./ B0_Te;

% time periods (30/360) and BPV
deltas = yearfrac(discountDates(1:end-1), discountDates(2:end), 0);
DFs    = B_Te_T(2:end);
bpv    = sum(deltas .* DFs);

% Option payoff multiplier based on type
switch lower(swaptionType)
    case 'receiver'
        price = bpv * ( strike * normcdf(-d2) - S0 * normcdf(-d1) );
    case 'payer'
        price = bpv * ( S0 * normcdf(d1) - strike * normcdf(d2) );
    otherwise
        error('swaptionType can be ''receiver'' or ''payer''.');
end

% delta for later
if computeDelta
    switch lower(swaptionType)
        case 'receiver'
            delta = -bpv * normcdf(-d1);
        case 'payer'
            delta =  bpv * normcdf(d1);
    end
else
    delta = [];
end
end