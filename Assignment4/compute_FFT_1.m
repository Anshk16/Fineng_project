function integral_FFT = compute_FFT_1(x_moneyness, x_1, phi, M)

%Let us compute all parameters
N = 2^M;
x_N = -x_1;
dx = (x_N - x_1)/(N-1);
dz = 2*pi/(N*dx);
z_1 = -dz*(N-1)/2;
z_N = -z_1;
x = x_1:dx:x_N;
z = z_1:dz:z_N;
imm = 1i;

%Computing integral using Lewis formula
lewis_integrand = @(csi) 1/(2*pi).*phi(-csi-imm/2)./(csi.^2+1/4);

% Compute the Fourier transform
lewis_integrand_final = lewis_integrand(x).*exp(-imm*z_1*dx * (0:N-1));
f_hat = dx*exp(-imm*x_1*z).*fft(lewis_integrand_final, N);

% Compute integral
integral_FFT = interp1(z,f_hat,x_moneyness,'spline');