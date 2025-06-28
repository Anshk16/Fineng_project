function integral_quad = compute_quad(x_moneyness, phi, x_1)

%check if everything is ok
imm = 1i;
x_N = -x_1;
f = @(csi,y) 1/(2*pi)*phi(-csi-imm/2)./(csi.^2+1/4).*exp(-imm*csi*y);
integral_quad = zeros(size(x_moneyness));

for i = 1:length(x_moneyness)
    y = x_moneyness(i);
    integral_quad(i) = quadgk(@(csi) f(csi,y),x_1, x_N);
end