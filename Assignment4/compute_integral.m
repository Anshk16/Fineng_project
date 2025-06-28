function price = call_price_integral(discount, F_0, k, sigma, eta, x_moneyness, delta, M, x_1, dz, flag, alpha)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Imputs
% k: volatility of the volatility
% sigma: implied volatility
% eta: skew
% x_moneyness: grid where we do the integral
% delta: yearfrac between value date and expiry date
% M,x_1,dz: parameters for FFT method
% flag: method choice
% alpha: parameter we use for laplace exponential
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

imm = 1i; %immaginary number

%Laplace exponential

if alpha == 0 %VG mode
    laplace_exponent = @(w) - delta / k * log(1 + k * w * sigma ^ 2);
     
elseif alpha > 0 && alpha < 1 
    laplace_exponent = @(w) (delta / k)*(1 - alpha) / alpha * (1 - (1 + w * k * sigma^2 / (1 - alpha)) .^ alpha);
end

%Characteristic function

char_func = @(csi) exp(-imm*csi*laplace_exponent(eta)).*exp(laplace_exponent((csi.^2+imm*(1+2*eta)*csi)/2));

% Integral computation for different methods
if flag == 1 %FFT method with x_1 and M
    integral = compute_FFT_1(x_moneyness, x_1, char_func, M);
elseif flag == 2 %FFT method with dz and M
    integral = compute_FFT_2(x_moneyness, dz, char_func, M);
elseif flag == 3 %quadratic method
    integral = compute_quad(x_moneyness, char_func, x_1);
end

price = discount * F_0 * (1 - exp(-x_moneyness / 2) .* integral);
